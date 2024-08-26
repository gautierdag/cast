from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)


class PhiModel:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-vision-instruct",
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Phi-3.5-vision-instruct", trust_remote_code=True, num_crops=4
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
        # wrap text in [INST]...[/INST] to indicate it is an instruction
        # prompt = f"[INST] {instructions} [/INST] {start_decode}"
        # inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
        #     self.model.device
        # )
        # inputs["input_ids"][inputs["input_ids"] == 64003] = 64000
        # output = self.model.generate(
        #     **inputs,
        #     max_new_tokens=max_new_tokens,
        #     do_sample=False,
        #     use_cache=True,
        # )
        # return self.processor.decode(output[0], skip_special_tokens=True)
        raise NotImplementedError("PhiModel.generate is not implemented yet.")
