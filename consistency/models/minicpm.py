import warnings

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class MiniCPMModel:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "openbmb/MiniCPM-Llama3-V-2_5",
            torch_dtype=torch.float16,  # float32 for cpu
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True
        )

    def generate(
        self, instructions: str, image=None, max_new_tokens=256, start_decode: str = ""
    ) -> str:
        prompt = f" USER: {instructions} ASSISTANT: {start_decode}"
        if image is None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            # generate
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
            )[0]
        else:
            assert (
                "<image>" in prompt
            ), "Prompt should contain '<image>' to insert the image."
            text_chunks = [
                self.tokenizer(chunk).input_ids for chunk in prompt.split("<image>")
            ]
            input_ids = (
                torch.tensor(
                    text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long
                )
                .unsqueeze(0)
                .to(self.model.device)
            )
            attention_mask = torch.ones_like(input_ids)
            image_tensor = self.model.process_images([image], self.model.config).to(
                dtype=self.model.dtype, device=self.model.device
            )
            # generate
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
            )[0]
        return self.tokenizer.decode(output, skip_special_tokens=True)
