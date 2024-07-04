from typing import List
import os
from datetime import datetime

import click
import yaml
import json
import torch as pt
import cv2
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import Pix2StructImageProcessor, Pix2StructForConditionalGeneration, Pix2StructVisionModel

from guiagents.utils import (
    load_video_data,
    load_video_paths,
    load_mouse_tracking,
    quantize_xy,
    create_experiment_checkpoint
)

EXPERIMENT_NAME='train_oc'

class SimpleDataset(Dataset):
    def __init__(
            self,
            meta_data_path,
            videos_dir,
            transcripts_dir,
            mouse_tracking_dir,
            lags:List[int]=list(range(1,5)),
            dxdy_bin_sizes=(0.2,0.2),
            xy_bin_sizes=(0.1,0.1),
            mu=5,
            mouse_tracking_conf_tresh = 0.7,
            frames_per_step=3,
            output_image_size=(720, 1280),
            transforms=None,
            mask_vid_ids=None
        ):

        """causal_skip_frames corresponds to the number of frames at the beggining of each video that cannot
        be used to train since they do not have enough frames before them to run the model"""
        self.vid_df = load_video_data(meta_data_path)
        self.lags = lags
        self.video_paths = load_video_paths(videos_dir)
        self.mouse_tracking_conf_tresh = mouse_tracking_conf_tresh
        self.frames_per_step = frames_per_step
        self.mouse_tracking_dir = mouse_tracking_dir
        self.output_image_size = output_image_size
        self.resize = v2.Resize(self.output_image_size)
        self.transforms = transforms
        self.mouse_tracking = load_mouse_tracking(mouse_tracking_dir, mouse_tracking_conf_tresh, np.max(lags), frames_per_step=3)
        self.transcripts_dir = transcripts_dir
        self.dxdy_bin_sizes = dxdy_bin_sizes
        self.xy_bin_sizes = xy_bin_sizes
        self.mu = mu

        if mask_vid_ids:
            self.mask_vid_ids=set(mask_vid_ids)
        else:
            self.mask_vid_ids=set()

        # resolving misshapen videos frames
        misshapen = set(self.vid_df.query('~((frame_height == @output_image_size[0]) & (frame_width == @output_image_size[1]))').index)

        # resolve missing transcripts
        avail_transcripts = set([
            os.path.splitext(os.path.basename(path))[0]
            for path in glob(os.path.join(self.transcripts_dir, "*.json"))
        ])

        unavail_transcripts = set(self.vid_df.index).difference(avail_transcripts)

        self.mask_vid_ids = self.mask_vid_ids.union(misshapen)
        self.mask_vid_ids = self.mask_vid_ids.union(unavail_transcripts)
        self._calculate_runnable_frames(self.mouse_tracking)
    
    # TODO remove
    def _load_mouse_tracking(self,):
        max_lag = np.max(self.lags)
        confidence_treshold = self.mouse_tracking_conf_tresh
        tracking_files = glob(os.path.join(self.mouse_tracking_dir, "*.csv"))
        #video_ids = [os.path.splitext(os.path.basename(path))[0] for path in tracking_files]
        data = dict()
        for path in tracking_files:
            video_id = os.path.splitext(os.path.basename(path))[0]
            frame_height = self.vid_df.loc[video_id, 'frame_height']
            frame_width = self.vid_df.loc[video_id, 'frame_width']
            df = pd.read_csv(path, index_col='frame').query("conf>@confidence_treshold")
            if(len(df) <= max_lag):
                continue
            is_subsq = (df.index.diff()==self.frames_per_step)
            lags_avail = np.concatenate([[False]*(max_lag),np.convolve(is_subsq, np.ones(max_lag+1), mode='valid') >= max_lag+1])
            df['is_subsq'] = is_subsq
            df['lags_avail'] = lags_avail
            bboxes = BoundingBoxes(
                pt.tensor(df[['x', 'y', 'width', 'heigth']].to_numpy()),
                format=BoundingBoxFormat.XYWH,
                canvas_size=(frame_height, frame_width)
            )
            bboxes = self.resize(bboxes)
            if self.transforms:
                bboxes = self.transforms(bboxes)
            df.loc[:, ['x', 'y', 'width', 'heigth']] = bboxes.numpy()
            df['dx'] = df['x'].diff()
            df.loc[~df['is_subsq'], 'dx'] = None
            df['dy'] = df['y'].diff()
            df.loc[~df['is_subsq'], 'dy'] = None
            data[video_id] = df
        return data

    def _load_transcript(self, video_id):
        with open(os.path.join(self.transcripts_dir, f"{video_id}.json"), 'r') as f:
            transcript = json.load(f)
        intervals = pd.DataFrame(transcript['transcript'], columns=['tStartMs', 'dDurationMs', 'text'])
        segs = list(map(lambda x: x.get('segs', None), transcript['transcript']))
        is_generated = transcript['generated']
        return intervals, segs, is_generated

    def _calculate_runnable_frames(self, mouse_tracking):
        filtered_mouse_tracking = filter(lambda x: not x[0] in self.mask_vid_ids, mouse_tracking.items())
        self._runnable_frames = dict(map(lambda x: (x[0],x[1][x[1]['lags_avail']].index), filtered_mouse_tracking))
        frame_counts = map(lambda x: (x[0], len(x[1])), self._runnable_frames.items())
        self._num_runnable_frames = pd.Series(dict(frame_counts))

    def _find_relative_indexes(self, idx):
        iter_cumsum = self._num_runnable_frames.cumsum().shift(1, fill_value=0)
        video_index = iter_cumsum.searchsorted(idx, side='right') - 1
        video_id = iter_cumsum.index[video_index]
        relative_iter = idx - iter_cumsum[video_id]
        relative_frame = self._runnable_frames[video_id][relative_iter]
        return video_id, relative_iter, relative_frame
    
    def __getitem__(self, idx, get_screens=True):
        vid_id, rel_iter, rel_frame = self._find_relative_indexes(idx)
        vid_path = self.video_paths[vid_id]
        mt = self.mouse_tracking[vid_id] # mouse tracking
        mt_ord = mt.index.get_loc(rel_frame)
        intervals, _, is_caption_generated = self._load_transcript(vid_id)
        rel_time = rel_frame / self.vid_df.loc[vid_id, 'fps'] * 1000 # in ms
        frame_height = self.vid_df.loc[vid_id, 'frame_height']
        frame_width = self.vid_df.loc[vid_id, 'frame_width']

        mt_lags = (mt.iloc[[mt_ord] + [mt_ord - lag for lag in self.lags]]).copy()

        # normalize dx and dy
        mt_lags.loc[:, 'dx'] /= frame_width
        mt_lags.loc[:, 'dy'] /= frame_height
        mt_lags.loc[:, 'x'] /= frame_width
        mt_lags.loc[:, 'y'] /= frame_height

        # quantize
        mt_lags['dx_bin'] = mt_lags['dx'].apply(lambda x : quantize_xy(x, 1., self.dxdy_bin_sizes[0], self.mu))
        mt_lags['dy_bin'] = mt_lags['dy'].apply(lambda y : quantize_xy(y, 1., self.dxdy_bin_sizes[1], self.mu))

        label = pt.tensor((int(mt_lags.iloc[0]['dx_bin']), int(mt_lags.iloc[0]['dy_bin'])))
        mt_lags_dxdy = pt.tensor(mt_lags.iloc[1:][['dx_bin', 'dy_bin']].to_numpy())

        # quantize positions
        bins_x = np.arange(0,1,self.xy_bin_sizes[0])
        bins_y = np.arange(0,1,self.xy_bin_sizes[1])
        bins_x = np.arange(0,1+self.xy_bin_sizes[0], self.xy_bin_sizes[0])
        bins_y = np.arange(0,1+self.xy_bin_sizes[1], self.xy_bin_sizes[1])
        mt_lags['x_bin'] = pd.cut(mt_lags['x'], bins=bins_x, labels=range(len(bins_x)-1)).astype(int)
        mt_lags['y_bin'] = pd.cut(mt_lags['y'], bins=bins_y, labels=range(len(bins_y)-1)).astype(int)
        mt_lags_xy = pt.tensor(mt_lags.iloc[1:][['x_bin', 'y_bin']].to_numpy())

        screens = None
        if get_screens:
            cap = cv2.VideoCapture(vid_path)
            screen_imgs = []
            for lag in self.lags:
                cap.set(cv2.CAP_PROP_POS_FRAMES, rel_frame - lag*self.frames_per_step)
                _, screen_img = cap.read()
                screen_imgs.append(pt.tensor(cv2.cvtColor(screen_img, code=cv2.COLOR_BGR2RGB))/255.0)
            screens = pt.stack(screen_imgs)
            cap.release()
        
        captions = intervals.query('tStartMs < @rel_time and @rel_time < (tStartMs+dDurationMs)')['text']
        caption = "" if len(captions) == 0 else captions.iloc[0] # if at that time there is no subtitle then put nothing
        #caption = intervals[(intervals['tStartMs'] < rel_time) and (rel_time < (intervals['tStartMs'] + intervals['dDurationMs']))]
        
        return vid_id, rel_frame, *((screens,) if get_screens else ()), caption, is_caption_generated, mt_lags_xy, mt_lags_dxdy, label

    def __len__(self):
        return sum(self._num_runnable_frames)


class PolicyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._num_dx_bins = 11
        self._num_dy_bins = 11
        self._num_x_bins = 10
        self._num_y_bins = 10
        self.vision_model = Pix2StructVisionModel.from_pretrained(config['vision_model_path'])
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.vision_model.config.hidden_size, nhead=8, batch_first=True),
            num_layers=1
        )
        self.decoder_tgt = nn.Parameter(pt.zeros(1,self.vision_model.config.hidden_size))
        self.linear = nn.Linear(in_features=768+(10+10+11+11)*4)
        self.lin_out_x = nn.Linear(in_features=512, out_features=self._num_dx_bins)
        self.lin_out_y = nn.Linear(in_features=512, out_features=self._num_dy_bins)
        #self.out_projection = nn.Sequential(
        #   nn.Linear(in_features=768+(10+10+11+11)*4, out_features=512),
        #   nn.ELU(),
        #   nn.Linear(in_features=512, out_features=11+11),
        #)

    def forward(self, flattened_patches, attention_mask, dxdy_lags, xy_lags):
        dx = F.one_hot(dxdy_lags[:,:,0].long(), num_classes=self._num_dx_bins)
        dy = F.one_hot(dxdy_lags[:,:,1].long(), num_classes=self._num_dy_bins)
        x = F.one_hot(xy_lags[:,:,0].long(), num_classes=self._num_x_bins)
        y = F.one_hot(xy_lags[:,:,1].long(), num_classes=self._num_y_bins)
        mouse = pt.concat((dx, dy, x, y), dim=-1).flatten(start_dim=1)
        vision_out = self.vision_model(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask
        )
        vision_hidden = vision_out.last_hidden_state
        tgt = self.decoder_tgt.expand((*vision_hidden.shape[:-2], *self.decoder_tgt.shape[-2:]))
        # tgt_mask = pt.full((1,1), fill_value=-1e10) do i need this mask?
        decoded_screen = self.decoder(tgt=tgt, memory=vision_hidden)
        decoded_screen = decoded_screen.squeeze(dim=1)
        features = pt.concat([decoded_screen, mouse], dim=-1)
        features = F.elu(self.linear(features))
        return self.lin_out_x(features), self.lin_out_y(features)


@click.command()
@click.option('--config', type=click.File('r'))
def script(config):

    # Initialize experiment
    if not config:
        config = os.path.join("..", "configs", f"{EXPERIMENT_NAME}.yaml")
    print(config)
    #with open(config, 'r') as conf_file:
    config = yaml.safe_load(config)
    if not os.path.exists(config['run']['store_dir']):
        raise ValueError(f"Directory in configuration: config.run.store_dir: {config['run']['store_dir']} does not exist.")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_tag = f"{config['run']['tag']}_{timestamp}"
    run_dir = os.path.join(config['run']['store_dir'], run_tag)

    create_experiment_checkpoint(
        exp_name=EXPERIMENT_NAME,
        exp_dir=run_dir,
    )

    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoints_dir)

    tensorboard_dir = os.path.join(run_dir, 'runs')
    os.makedirs(tensorboard_dir)

    # Device
    device = "cuda" if pt.cuda.is_available() else "cpu"

    # Tensorboard
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Dataset
    full_dataset = SimpleDataset(
        meta_data_path=config['data']['meta_data_path'],
        videos_dir=config['data']['videos_dir'],
        transcripts_dir=config['data']['transcripts_dir'],
        mouse_tracking_dir=config['data']['mouse_tracking_dir'],
        lags=list(range(1,5)),
        dxdy_bin_sizes=(0.2,0.2),
        xy_bin_sizes=(0.1,0.1),
        mouse_tracking_conf_tresh = 0.7,
        frames_per_step=3,
        output_image_size=(720, 1280),
        transforms=None
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, test_data = random_split(full_dataset, [train_size, val_size])

    # Create dataloader
    train_dl = DataLoader(train_data, batch_size=4, shuffle=True, drop_last=True)
    val_dl = DataLoader(train_data, batch_size=4, shuffle=False, drop_last=False)

    # processor
    processor = Pix2StructImageProcessor.from_pretrained("google/pix2struct-docvqa-base")

    # Model
    model = PolicyModel(config=config['model']).to(device)

    # Create optimizer

    optimizer = pt.optim.Adam(
        [model.parameters],
        lr=config['train']['lr'],
        weight_decay=config['train']['w_decay']
    )

    crit_x = nn.CrossEntropyLoss() # no weights for now
    crit_y = nn.CrossEntropyLoss() # no weights for now

    loss_sum = 0
    step = 0
    for epoch in range(config['train']['num_epoch']):
        for batch_i, (
                vid_id,
                rel_frame,
                screens,
                caption,
                is_caption_generated,
                mt_lags_xy,
                mt_lags_dxdy,
                label
            ) in enumerate(train_dl):
            model.train()

            visual_inputs = processor(images=screens, header_text=caption, return_tensors="pt", is_vqa=True)
            screens = screens.to(device)
            flattened_patches = visual_inputs['flattened_patches'].to(device)
            visual_attention_mask = visual_inputs['attention_mask'].to(device)
            mt_lags_xy = mt_lags_xy.to(device)
            mt_lags_dxdy = mt_lags_dxdy.to(device)

            x_pred, y_pred = model(
                flattened_patches=flattened_patches,
                attention_mask=visual_attention_mask,
                mt_lags_dxdy=mt_lags_dxdy,
                mt_lags_xy=mt_lags_xy,
            )

            loss_x = crit_x(x_pred, label)
            loss_y = crit_y(y_pred, label)
            loss = loss_x + loss_y
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            if step % config['train']['logging_step'] == 0:
                loss_sum /= config['train']['logging_step']*config['train']['batch_size']
                writer.add_scalar('train/loss', loss_sum, step)
                loss_sum = 0 

            # Validation
            if step % config['train']['validation_step'] == 0:
                model.eval()
                with pt.no_grad():
                    val_loss = 0
                    for batch_i, (
                            vid_id,
                            rel_frame,
                            screens,
                            caption,
                            is_caption_generated,
                            mt_lags_xy,
                            mt_lags_dxdy,
                            label
                        ) in enumerate(val_dl):

                        visual_inputs = processor(images=screens, header_text=caption, return_tensors="pt", is_vqa=True)
                        screens = screens.to(device)
                        flattened_patches = visual_inputs['flattened_patches'].to(device)
                        visual_attention_mask = visual_inputs['attention_mask'].to(device)
                        mt_lags_xy = mt_lags_xy.to(device)
                        mt_lags_dxdy = mt_lags_dxdy.to(device)

                        x_pred, y_pred = model(
                            flattened_patches=flattened_patches,
                            attention_mask=visual_attention_mask,
                            mt_lags_dxdy=mt_lags_dxdy,
                            mt_lags_xy=mt_lags_xy,
                        )

                        loss_x = crit_x(x_pred, label)
                        loss_y = crit_y(y_pred, label)
                        loss = loss_x + loss_y
                        val_loss += loss.item()
                
                val_loss /= config['val']['batch_size'] * len(val_dl)
                writer.add_scalar('validation/loss', val_loss, step)

            # Save checkpoint
            if step % config['train']['validation_step'] == 0:
                this_checkpoint_dir = os.path.join(checkpoints_dir, f"{step}")
                os.makedirs(this_checkpoint_dir)
                pt.save(model.state_dict(), os.path.join(this_checkpoint_dir, 'model.pt'))
                pt.save(optimizer.state_dict(), os.path.join(this_checkpoint_dir, 'optim.pt'))
            
            step += 1
        
        

if __name__ == "__main__":
    script()