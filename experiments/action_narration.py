import os
import yaml
import json
from datetime import datetime
import base64
from tqdm import tqdm
import glob
from typing import Optional, Dict, Callable, List, Any
import click
from openai import OpenAI
import cv2

from guiagents.utils import vid2array, screen_pair_figure, create_experiment_checkpoint

EXPERIMENT_NAME = "action_narration" # must be same as script name and config name

def default_img_processor(images):
    proc_images = []
    for img in images:
        jpeg_img = cv2.imencode('.jpeg', img)[1]
        b64_img = base64.b64encode(jpeg_img)
        proc_images.append(b64_img.decode('utf-8'))
    return proc_images

class ActionRecognitionPrompt:
    actions: Dict[str, str]

    def __init__(self, *, actions: Optional[Dict[str, str]] = None, img_processor: Optional[Callable[List[Any], List[str]]] = None, few_shot_examples: Optional[List[str]] = None):

        if actions:
            self.actions = actions
        
        if few_shot_examples:
            self.few_shot_examples = few_shot_examples 
        else:
            self.few_shot_examples = None
        
        if img_processor:
            self.img_processor = img_processor
        else:
            self.img_processor = default_img_processor

    def generate(self, images: List[Any]) -> List[Dict[str, str]]:
        action_tags = "\n".join([f"<{tag}> - {definition}" for tag, definition in self.actions.items()])
        system_message = f"You are a GUI usage narration and action recognition system. You respond using imperative sentences that match a users behaviour shown in provided image pairs. At the end of every imperative sentence you will choose a tag of the action performed using one of the provided action tags, the tag must be one of the provided ones. The action tags with their definitions are as follows: \n\n {action_tags} \n\n."
        user_message = f"You are given two images of a computer screen, the first one being before some actions being taken and the second one after. Which actions would the user need to do to reproduce the outcome. Focus on the position of the cursor or texting being typed out to determine the action. Remember there can be multiple actions between the two images."

        prompt = [
            {
            "role": "system",
            "content": system_message,
            }
        ]

        for example in self.few_shot_examples:
            prompt.extend([
                {
                "role": "user",
                "content": [
                    user_message,
                ],
                },
                {
                "role": "assistant",
                "content": example,
                }
            ])
        
        prompt.append(
            {
            "role": "user",
            "content": [
                user_message,
                *[{"image": img, "resize":768} for img in self.img_processor(images)],
            ],
            }
        )

        return prompt

@click.command()
@click.option('--config', type=click.File('r'))
@click.option('--api_key', type=str)
@click.option('--checkpoint/--no-checkpoint', default=True)
def experiment(config, api_key, checkpoint):

    if not config:
        config = os.path.join("configs", f"{EXPERIMENT_NAME}.yaml")
    with open(config) as conf_file:
        config = yaml.safe_load(conf_file)

    if not api_key:
        with open(os.path.join("configs", "secrets.yaml")) as secrets_file:
            secrets = yaml.safe_load(secrets_file)
        api_key = secrets['openai_api_key']

    client = OpenAI(api_key=api_key)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_tag = f"{config['run']['tag']}_{timestamp}"
    
    if not os.path.exists(config['run']['store_dir']):
        raise ValueError(f"Directory in configuration: config.run.store_dir: {config['run']['store_dir']} does not exist.")
    
    run_dir = os.path.join(config['run']['store_dir'], run_tag)
    
    if os.path.exists(run_dir):
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        run_tag = f"{config['run']['tag']}_{timestamp}"
        run_dir = os.path.join(config['run']['store_dir'], run_tag)

    os.makedirs(run_dir)
    create_experiment_checkpoint(
        exp_name=EXPERIMENT_NAME,
        exp_dir=run_dir,
    )

    # Create prompt template
    prompt_template = ActionRecognitionPrompt(
        actions=config['prompt']['action_set'],
        few_shot_examples=config['prompt']['few_shot_examples'],
    )

    all_videos = sum([
        glob.glob(os.path.join(config['data']['data_dir'], "*" + vid_ext)) for vid_ext in config['data']['extensions']
    ], [])

    # Load metadata
    metadata_dir = os.path.join(config['data']['data_dir'], config['data']['metadata_file'])
    with open(metadata_dir, 'r') as metadata_file:
        metadata = json.load(metadata_file)

    for vid_path in tqdm(all_videos[::-1]):
        vid_name = os.path.basename(vid_path).split('.')
        vid_name, vid_ext = ".".join(vid_name[:-1]), vid_name[-1]
        # Load important screens metadata
        try:
            frames = metadata[vid_name]['prompt_frames']
        except KeyError:
            raise ValueError(f"Video {vid_path} does not exist")

        frames = zip(frames, frames[1:])

        video = vid2array(vid_path=vid_path)

        for i, j in frames:
            img1 = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(video[j], cv2.COLOR_BGR2RGB)
            prompt = prompt_template.generate([img1, img2])
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=prompt,
                max_tokens=300,
            )
            text = response.choices[0].message.content
            text = "test"
            screen_pair_figure(img1, img2, text, os.path.join(run_dir, f"{vid_name}_{i}_{j}.jpeg"))

if __name__ == "__main__":
    experiment()