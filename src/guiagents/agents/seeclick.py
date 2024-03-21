import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import ast
import logging
import torch

logger = logging.getLogger(__name__)

class SeeClickAgent:

    def __init__(self, seeclick_path=None, qwen_path=None):

        if qwen_path is None:
            qwen_path = "Qwen/Qwen-VL-Chat"

        if seeclick_path is None:
            seeclick_path = "cckevinn/SeeClick-mind2web"

        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seeclick = AutoModelForCausalLM.from_pretrained(seeclick_path, device_map=device, trust_remote_code=True, bf16=True).eval()
        self.seeclick.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)

    def create_session_from_task(self, task_desc):
        return {
            "task": task_desc,
            "actions_taken": [],
        }

    def choose_action(self, session, **kwargs):
        if "screenshot" not in kwargs.keys():
            raise ValueError(f"{self.__class__}.choose_action() Must have a screenshot passed as arg")
        else:
            screenshot = kwargs['screenshot']

        if "screenshot_path" not in kwargs.keys():
            raise ValueError(f"{self.__class__}.choose_action() Must have a screenshot_path passed as arg")
        else:
            screenshot_path = kwargs['screenshot_path']

        previous_step = ""
        for i, action in enumerate(session['actions_taken'][-4:]):
            click_point = f"({action['x']/screenshot.size[0]},{action['y']/screenshot.size[1]})"
            action_step = f"{{\"action_type\": {action['type']}, \"click_point\": {click_point}, \"value\": \"{action['value']}\"}}"
            previous_step += 'Step' + str(i) + ': ' + action_step + ". "

        prompt_template = "Please generate the next move according to the ui screenshot, instruction and previous actions. Instruction: {}. Previous actions: {}"
        prompt = prompt_template.format(session['task'], previous_step)

        query = tokenizer.from_list_format(
            [{'image': screenshot_path}, {'text': prompt}, ]
        )

        with torch.no_grad():
            response, history = self.seeclick.chat(self.tokenizer, query=query, history=None)
        
        try:
            ast.literal_eval(response)
        except:
            logging.warning(f"Agent could not parse LLM output: {response}")

        return action
    
    def perform_action(self, session, action, **kwargs):
        self.session['actions_taken'].append(
            action
        )