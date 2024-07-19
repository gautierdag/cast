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
        """
        Given an instruction and an optional image, generate a response of maximum length `max_new_tokens`.

        Start decode is a string that will be appended after the instructions.
        For instance to start the assistant response.
        """
        raise NotImplementedError()
