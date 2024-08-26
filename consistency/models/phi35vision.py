from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)
from PIL import Image


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

    def get_concat_h(self, image_1, image_2) -> list:
        return [image_1, image_2]

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

        phi_instructions = instructions
        if isinstance(image, list):
            if len(image) == 2:
                phi_instructions = phi_instructions.replace(
                    "<image>", "<|image_1|>\n<|image_2|>\n"
                )
            elif len(image) == 1:
                phi_instructions = phi_instructions.replace("<image>", "<|image_1|>\n")
            else:
                raise ValueError("Only 1 or 2 images are supported")

        elif image is not None:
            raise ValueError("Should provide a list of images")
            # phi_instructions = phi_instructions.replace("<image>", "<|image_1|>\n")

        prompt = f"<|user|>\n{phi_instructions}\n<|end|>\n<|assistant|>\n{start_decode}"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.model.device
        )
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        # remove input tokens
        generate_ids = output[:, inputs["input_ids"].shape[1] :]
        return self.processor.decode(generate_ids[0], skip_special_tokens=True)
