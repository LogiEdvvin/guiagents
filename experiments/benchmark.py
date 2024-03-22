import yaml
import logging
import logging.config

with open("../logging_config.yaml", "r") as log_conf_file:
    log_conf = yaml.safe_load(log_conf_file.read())
logging.config.dictConfig(log_conf)

import json
import os
from tqdm import tqdm
import click
from datetime import datetime
from PIL import Image
from guiagents.agents import SeeClickAgent

results_path = "./results.json"
data_path = "../data/mind2web/"
mind2web_imgs_dir = os.path.join(data_path, "screenshots")

@click.group(chain=True, invoke_without_command=True)
@click.option('--agent', '-a', multiple=True, type=str, help='Add an agent to be benchmarked if no agents provided then all will be used')
@click.pass_context
def benchmark(ctx, agent):
    ctx.ensure_object(dict)
    timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    ctx.obj['experiment_tag'] = f"benchmark_{timestamp}"
    ctx.obj['agents'] = []


@benchmark.command('m2w')
@click.option('--across', type=click.Choice(['task', 'website', 'domain', 'all']), default='all')
@click.pass_context
def mind2web(ctx, across):
    logger = logging.getLogger(__name__ + ".mind2web")

    # LOAD DATASET

    if(across == 'all'):
        across = ['task', 'website', 'domain']
    else:
        across = [across]

    m2w_data = []
    for section_name in across:
        section_data = json.load(open(os.path.join(data_path, f"test_{section_name}.json")))
        m2w_data.extend([(section_name, episode) for episode in section_data])

    agent = SeeClickAgent()

    results = []

    ep_num = 0

    for sect, episode in tqdm(m2w_data):
        task_desc = episode['confirmed_task']
        episode_id = episode['annotation_id']
        session = agent.create_session_from_task(task=task_desc)

        for step_num, step in enumerate(episode['actions']):
            if "bbox" not in step:
                logger.warning(f'No bbox found in step {step_num} of episode with id: {id}')
                continue

            # Retrieve Screen shot
            img_filename = episode_id + '-' + step['action_uid'] + '.jpg'
            img_path = os.path.join(mind2web_imgs_dir, img_filename)

            if not os.path.exists(img_path):
                logger.warning(f'No screenshot found in step {step_num} of episode with id: {id}')
                return None

            screenshot = Image.open(img_path)

            agent_action = agent.choose_action(
                session,
                screenshot = screenshot,
                screenshot_path = img_path
            )

            ref_click_x = (step['bbox']['x'] + step['bbox']['width'])/2
            ref_click_y = (step['bbox']['y'] + step['bbox']['height'])/2

            ref_action = {
                "type": step['operation']['op'],
                "value": step['operation']['value'],
                "x": ref_click_x,
                "y": ref_click_y
            }

            step_result = {
                "episode_id": episode_id,
                "action_uid": step["action_uid"],
                "agent": "SeeClickAgent", # TODO
                "step": step_num,
                "test_data_section": sect,
                "task": task_desc,
                "pred_action_type": agent_action['type'],
                "pred_action_value": agent_action.get('value', None),
                "pred_action_location": agent_action['x'],
                "pred_action_location": agent_action['y'],
                "ref_action_type": step['operation']['op'],
                "ref_action_value": step['operation']['value'],
                "ref_action_bbox_x": step['bbox']['x'],
                "ref_action_bbox_y": step['bbox']['y'],
                "ref_action_bbox_w": step['bbox']['width'],
                "ref_action_bbox_h": step['bbox']['height'],
            }

            results.append(step_result)

            agent.perform_action(session, ref_action) # apply action to session
        
        ep_num += 1
        if ep_num % 10 == 0:
            if (os.path.exists("./results.json")):
                with open("./results.json", "r") as res_file:
                    old_results = json.load(res_file)
                
                old_results.append(results)
            else:
                old_results = results
            
            with open("./results.json", "w") as res_file:
                json.dump(old_results, res_file)
        
        results = []
            

if __name__ == "__main__":
    benchmark(obj={})