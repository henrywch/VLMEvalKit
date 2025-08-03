import torch
from PIL import Image
from transformers import AutoModel , AutoProcessor

from .base import BaseModel
from ..smp import *
import string
import pandas as pd


class LLaVA_Qwen_S(BaseModel):
    
    INSTALL_REQ=True
    INTERLEAVE = True

    def __init__(self, model_path='/root/autodl-tmp/models/llavanext-v1.5-0.5b', **kwargs):
        assert model_path is not None
        
        self.model_path = model_path
        print(f'Loading model from {self.model_path}')

        self.model = AutoModel .from_pretrained(
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
        
        if listinstr(['MMMU'], dataset):
            return True
        return False

    def build_prompt(self, line, dataset=None):
        if self.use_custom_prompt(dataset):
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

            full_prompt_text = ''
            if hint:
                full_prompt_text += f'Hint: {hint}'
            full_prompt_text += f'{question}'
            if len(options):
                full_prompt_text += options_prompt
                full_prompt_text = 'Study the image carefully and pick the option associated with the correct answer. \
                    Focus solely on selecting the option and avoid including any other content.' + full_prompt_text
        else:
            full_prompt_text = line['question']

        image_paths = self.dump_image(line, dataset)

        prompt_parts = full_prompt_text.split('<image>')
        
        interleaved_message = []
        for i, part in enumerate(prompt_parts):
            part = part.strip()
            if part:
                interleaved_message.append({'type': 'text', 'value': part})
            if i < len(image_paths):
                interleaved_message.append({'type': 'image', 'value': image_paths[i]})
        
        return interleaved_message

    def generate_inner(self, message, dataset=None):     
        content = ""
        images = []
        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            elif msg["type"] == "image":
                content+= "<image>"
                images.append(Image.open(msg["value"]).convert("RGB"))

        prompt = f"""<|im_start|>user
                     {content}<|im_end|>
                     <|im_start|>assistant
                  """
        image_sizes = [img.size[::-1] for img in images]

        image_inputs = self.processor.image_processor(
            images=images, 
            image_sizes=image_sizes, 
            return_tensors="pt"
        )

        text_inputs = self.processor.tokenizer(
            text=content, 
            return_tensors="pt",
            padding=True
        )

        inputs = {
            "pixel_values": image_inputs.pixel_values,
            "image_sizes": image_inputs.image_sizes,
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask
        }

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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
