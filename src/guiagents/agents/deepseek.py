from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import ast
import logging
import torch

logger = logging.getLogger(__name__)

class DeepSeekAgent:

    def __init__(self, seeclick_path=None, qwen_path=None):
        model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        vl_gpt = vl_gpt.to(torch.bfloat16).eval()#.cuda().eval() # NOT CUDA

    def create_session_from_task(self, task):
        pass

    def choose_action(self, session, **kwargs):
        if "screenshot" not in kwargs.keys():
            raise ValueError(f"{self.__class__}.choose_action() Must have a screenshot passed as arg")
        else:
            screenshot = kwargs['screenshot']

        if "screenshot_path" not in kwargs.keys():
            raise ValueError(f"{self.__class__}.choose_action() Must have a screenshot_path passed as arg")
        else:
            screenshot_path = kwargs['screenshot_path']

        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)

        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"{prepare_inputs['sft_format'][0]}", answer)
        return action
    
    def perform_action(self, session, action, **kwargs):
        # Adds performed action to session
        pass