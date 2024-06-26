import os
from datetime import datetime

import click
import yaml
import torch as pt
from torch import nn
from torch.utils.data import Dataset, DataLoader

from guiagents.utils import create_experiment_checkpoint

EXPERIMENT_NAME='train_oc'

class BaseCaptionStrategy:
    pass

class SimpleCaptionStrategy(BaseCaptionStrategy):
    pass

class SimpleDataset(Dataset):
    def __init__(
            self,
            meta_data_path,
            videos_dir,
            transcripts_dir,
            mouse_tracking_dir,
            #causal_skip_frames=5,
            lags:List[int]=list(range(1,5)),
            xy_foveated_binning_size=(11,11),
            mouse_tracking_conf_tresh = 0.7,
            frames_per_step=3,
            caption_strategy = None
        ):

        """causal_skip_frames corresponds to the number of frames at the beggining of each video that cannot
        be used to train since they do not have enough frames before them to run the model"""
        self.vid_df = load_video_data(meta_data_path)
        self.video_frame = (self.vid_df['frame_count'] - self.causal_skip_frames).cumsum().shift(1, fill_value=0)
        self.lags = lags
        self.video_paths = load_video_paths(videos_dir)
        self.mouse_tracking = load_mouse_tracking(
            mouse_tracking_dir,
            confidence_treshold=mouse_tracking_conf_tresh,
            max_lag=np.max(lags),
            frames_per_step=frames_per_step
        )
        self._calculate_runnable_frames(self.mouse_tracking)
        self.mouse_tracking_dir = mouse_tracking_dir
        self.transcripts_dir = transcripts_dir
        self.frames_per_step = frames_per_step
        self._caption_strategy = None

        if not caption_strategy:
            self.caption_strategy = SimpleCaptionStrategy()
    
    @property
    def caption_strategy():
        return self._caption_strategy.desc()

    def _calculate_runnable_frames(self, mouse_tracking):
        self._runnable_frames = dict(map(lambda x: (x[0],x[1][x[1]['lags_avail']].index), self.mouse_tracking.items()))
        frame_counts = map(lambda x: (x[0], len(x[1])), self._runnable_frames.items())
        self._num_runnable_frames = pd.DataFrame(frame_counts, columns=['id', 'frames'])
        self._num_runnable_frames.set_index('id', inplace=True)

    def _find_relative_indexes(self, idx):
        iter_cumsum = self._num_runnable_frames.cumsum().shift(1, fill_value=0)
        video_index = iter_cumsum.searchsorted(idx, side='right') - 1
        video_id = iter_cumsum.index[video_index]
        relative_iter = idx - iter_cumsum[video_id]
        relative_frame = self._runnable_frames[video_id][relative_iter]
        return video_id, relative_iter, relative_frame

    # TODO: this can be a pytorch compiled funciton
    def _foveal_binning(self, x, y):
        pass

    def __getitem__(self, idx, get_screens=True):
        vid_id, rel_iter, rel_frame = self._find_relative_indexes(idx)
        vid_path = self.video_paths[vid_id]
        self.mouse_tracking[vid_id]
        if get_screens:
            cap = cv2.VideoCapture(vid_path)
            screen_imgs = []
            for lag in self.lags:
                cap.set(cv2.CAP_PROP_POS_FRAMES, rel_frame - lag*self.frames_per_step)
                _, screen_img = cap.read()
                screen_imgs.append(pt.torch(cv2.cvtColor(screen_img, code=cv2.COLOR_BGR2RGB)))
            screens = pt.stack(screen_imgs)

        return vid_id, rel_frame, screens,

    def __len__(self):
        return sum(self._num_runnable_frames)

class PolicyModel(nn.Module):

    def __init__(self, ):
        super.__init__()
    
    def forward(self,):
        pass


@click.command()
@click.option('--config', type=click.File('r'))
def script(config):

    # Initialize experiment
    if not config:
        config = os.path.join("..", "configs", f"{EXPERIMENT_NAME}.yaml")
    with open(config) as conf_file:
        config = yaml.safe_load(conf_file)
    if not os.path.exists(config['run']['store_dir']):
        raise ValueError(f"Directory in configuration: config.run.store_dir: {config['run']['store_dir']} does not exist.")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_tag = f"{config['run']['tag']}_{timestamp}"
    run_dir = os.path.join(config['run']['store_dir'], run_tag)
    create_experiment_checkpoint(
        exp_name=EXPERIMENT_NAME,
        exp_dir=run_dir,
    )

    # Create dataloader
    data_loader = None

    # Create optimizer
    trainable_parameters = None # TODO

    optimizer = pt.optim.Adam(
        trainable_parameters,
        lr=config['train']['lr'],
        weight_decay=config['train']['w_decay']
    )

    loss_sum = 0
    for batch_i, (batch_images, batch_mouse, batch_episode_id) in enumerate(data_loader):
        batch_loss = 0
        for image, action, episode_id in zip(batch_images, batch_actions, batch_episode_id):
            agent_action = agent._env_action_to_agent(action, to_torch=True, check_if_null=True)
            if agent_action is None:
                # Action was null
                continue

if __name__ == "__main__":
    script()