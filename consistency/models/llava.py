import torch
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)


class LlavaModel:
    def __init__(self):
        self.processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        # set padding token to eos token
        self.model.generation_config.pad_token_id = (
            self.processor.tokenizer.eos_token_id
        )

    def generate(
        self,
        instructions: str,
        image=None,
        max_new_tokens=256,
        start_decode: str = "",
    ) -> str:
        # wrap text in [INST]...[/INST] to indicate it is an instruction
        prompt = f"[INST] {instructions} [/INST] {start_decode}"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.model.device
        )
        inputs["input_ids"][inputs["input_ids"] == 64003] = 64000
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        return self.processor.decode(output[0], skip_special_tokens=True)
