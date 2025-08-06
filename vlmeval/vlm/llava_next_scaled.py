import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .base import BaseModel
from ..smp import *
import string
import pandas as pd
from llava.model import *


class LLaVA_NeXT_S(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='/root/autodl-tmp/models/llavanext-scaled-0.5b', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        print(f'Loading model from {self.model_path}')

        self.model = LlavaQwenForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        
        self.kwargs = kwargs
        torch.cuda.empty_cache()
        self.num_beams = 1 if '0.5b' in self.model_path else 3

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMBench_DEV_EN'], dataset):
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        if not self.use_custom_prompt(dataset):
            return super().build_prompt(line, dataset)

        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:'
        for key, item in options.items():
            options_prompt += f'{key}. {item}'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}'
        prompt += f'{question}'
        if len(options):
            prompt += options_prompt
            prompt = 'Study the image carefully and pick the option associated with the correct answer. ' + prompt

        message = [{'image': p, 'text': prompt} for p in tgt_path]
        return message

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)
        image = Image.open(image_path).convert('RGB')

        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            chat,
            images=[image],
            return_tensors="pt"
        ).to(self.model.device)

        max_new_tokens = 1024
        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=self.num_beams
        )
        default_kwargs.update(self.kwargs)

        output_ids = self.model.generate(**inputs, **default_kwargs)

        input_len = inputs['input_ids'].shape[1]
        newly_generated_ids = output_ids[0, input_len:]
        response = self.processor.tokenizer.decode(newly_generated_ids, skip_special_tokens=True)
        
        return response.strip()
