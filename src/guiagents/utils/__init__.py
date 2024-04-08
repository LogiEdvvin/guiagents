import shutil
import os

import numpy as np
from matplotlib import pyplot as plt
import cv2

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