import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


class Llava15Model:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            "openbmb/RLAIF-V-7B",
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            "openbmb/RLAIF-V-7B",
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
        """
        Given an instruction and an optional image, generate a response of maximum length `max_new_tokens`.

        Start decode is a string that will be appended after the instructions.
        For instance to start the assistant response.
        """
        conversation = [
            {
            "role": "user",
            "content": [
                    {"type": "text", "text": instructions}
                ],
            },
        ]        

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        prompt = prompt + start_decode
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.model.device
        )
        # inputs["input_ids"][inputs["input_ids"] == 64003] = 64000
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        return self.processor.decode(output[0], skip_special_tokens=True)
