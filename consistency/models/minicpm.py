import PIL
import json
from copy import deepcopy
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoProcessor
)

class MiniCPMModel:
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            "openbmb/MiniCPM-Llama3-V-2_5",
            torch_dtype=torch.float16,  # float32 for cpu
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True)
        self.model.eval()

    def generate(
        self, instructions: str, image=None, max_new_tokens=256, start_decode: str = ""
    ) -> str:
        """
        Given an instruction and an optional image, generate a response of maximum length `max_new_tokens`.

        Start decode is a string that will be appended after the instructions.
        For instance to start the assistant response.
        """
        if image is not None:
            image = image.convert('RGB')
        msgs = [{'role': 'user', 'content': instructions.replace("<image>", "") + "\n" + start_decode}]
        output = self.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            sampling=False, # if sampling=False, beam_search will be used by default
            temperature=1,
            system_prompt='You are a helpful and precise assistant for checking the quality of the answer.'
        )
        return output

    def chat(
        self, image: PIL.Image.Image, msgs: str, tokenizer: AutoTokenizer, 
        max_new_tokens=1024, sampling=True, max_inp_length=2048, system_prompt='', **kwargs
    ):
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        copy_msgs = deepcopy(msgs)

        assert len(msgs) > 0, "msgs is empty"

        if image is not None and isinstance(copy_msgs[0]["content"], str):
            # copy_msgs[0]['content'] = '(<image>./</image>)\n' + copy_msgs[0]['content']
            copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

        images = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, PIL.Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)

        if system_prompt:
            sys_msg = {'role': 'system', 'content': system_prompt}
            copy_msgs = [sys_msg] + copy_msgs        

        prompt = self.processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True)
        if len(images) > 0:
            inputs = self.processor(prompt, images, return_tensors="pt", max_length=max_inp_length).to(self.model.device)
        else:
            inputs = self.processor.tokenizer(prompt, return_tensors="pt", max_length=max_inp_length).to(self.model.device)
            inputs["pixel_values"] = None
            inputs["tgt_sizes"] = None

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )
        with torch.inference_mode():
            answers = self.model.generate(
                inputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                decode_text=True,
                **generation_config
            )
        
        return answers[0]

