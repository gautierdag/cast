import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class BunnyModel:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "BAAI/Bunny-v1_1-Llama-3-8B-V",
            torch_dtype=torch.float16,  # float32 for cpu
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/Bunny-v1_1-Llama-3-8B-V", trust_remote_code=True
        )

    def generate(
        self,
        text: str,
        image=None,
        max_new_tokens=256,
    ) -> str:
        if image is None:
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(
                self.model.device
            )
            # generate
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
            )[0]
        else:
            assert (
                "<image>" in text
            ), "Prompt should contain '<image>' to insert the image."
            text_chunks = [
                self.tokenizer(chunk).input_ids for chunk in text.split("<image>")
            ]
            input_ids = (
                torch.tensor(
                    text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long
                )
                .unsqueeze(0)
                .to(self.model.device)
            )
            image_tensor = self.model.process_images([image], self.model.config).to(
                dtype=self.model.dtype, device=self.model.device
            )
            # generate
            output = self.model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
            )[0]
        return self.tokenizer.decode(output, skip_special_tokens=True)
