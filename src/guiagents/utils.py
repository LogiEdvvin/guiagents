import shutil
import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from glob import glob

def vid2array(vid_path):
    cap = cv2.VideoCapture(vid_path)
    ret = True
    frames = []
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            frames.append(img)
    video = np.stack(frames, axis=0) # dimensions (T, H, W, C)
    return video

def screen_pair_figure(img1, img2, text, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,10), constrained_layout=True)
    fig.suptitle(text, wrap=True)
    ax1.imshow(img1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.imshow(img2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.savefig(save_dir)
    plt.close(fig=fig)

def create_experiment_checkpoint(*,exp_name: str, exp_dir: str):
    configs_dir = os.path.join(exp_dir, "configs")
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=False)
    
    shutil.copy(f"{exp_name}.py", exp_dir)
    shutil.copy(os.path.join('configs', f'{exp_name}.yaml'), configs_dir)

    if os.path.exists(os.path.join('configs', 'secrets.yaml')):
        shutil.copy(os.path.join('configs', 'secrets.yaml'), configs_dir)
    else:
        print(os.path.join(configs_dir, 'secrets.yaml'))

    if os.path.exists(os.path.join('configs', '.gitignore')):
        shutil.copy(os.path.join('configs', '.gitignore'), configs_dir)

def load_video_data(data_path):
    if os.path.exists(data_path):
        vid_df = pd.read_json(data_path, orient='index')
        vid_df.index.name = 'id'
        vid_df['publish_date'] = pd.to_datetime(vid_df['publish_date'])
        return vid_df
    else:
        return None

def store_video_data(vid_df: pd.DataFrame, data_path):
    vid_df.publish_date = vid_df.publish_date.apply(lambda x : x.isoformat())
    vid_df.to_json(data_path, indent=4, orient="index")

def parse_video_file(video_path):
    match = re.match(r"\[(.+)\]_(.*)", os.path.basename(video_path))
    return {
        'id': match.group(1),
        'video_name': match.group(2)
    }

def load_video_paths(video_download_dir):
    downloaded_paths = glob(os.path.join(video_download_dir, "*.mp4"))
    return dict(map(lambda x: (parse_video_file(x)['id'], x), downloaded_paths))

def load_mouse_tracking(mouse_tracking_dir, confidence_treshold, max_lag, frames_per_step):
    tracking_files = glob(os.path.join(mouse_tracking_dir, "*.csv"))
    video_ids = [os.path.splitext(os.path.basename(path))[0] for path in tracking_files]
    data = []
    for path in tracking_files:
        df = pd.read_csv(path, index_col='frame').query("conf>@confidence_treshold")
        if(len(df) <= max_lag):
            continue
        is_subsq = (df.index.diff()==frames_per_step)
        lags_avail = np.concatenate([[False]*(max_lag),np.convolve(is_subsq, np.ones(max_lag+1), mode='valid') >= max_lag+1])
        df['is_subsq'] = is_subsq
        df['lags_avail'] = lags_avail
        df['dx'] = df['x'].diff()
        df.loc[~df['is_subsq'], 'dx'] = None
        df['dy'] = df['y'].diff()
        df.loc[~df['is_subsq'], 'dy'] = None
        data.append(df)
    return dict(zip(video_ids, data))
    