# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
# )


class LlavaModel:
    def __init__(self):
        pass
        # self.model = AutoModelForCausalLM.from_pretrained(
        # #
        #     torch_dtype=torch.float16,  # float32 for cpu
        #     device_map="auto",
        #     trust_remote_code=True,
        # )
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     "BAAI/Bunny-v1_1-Llama-3-8B-V", trust_remote_code=True
        # )

    def generate(
        self,
        text: str,
        image=None,
        max_new_tokens=256,
    ) -> str:
        inputs = self.tokenizer(text=text, images=image, return_tensors="pt").to(
            self.model.device
        )
        inputs["input_ids"][inputs["input_ids"] == 64003] = 64000
        output = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
