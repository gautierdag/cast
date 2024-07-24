from .bunny import BunnyModel
from .llava import LlavaModel
from .gpt4o import GPT4OMini
from .minicpm import MiniCPMModel
from .internvl import InternVLModel


def model_factory(model_type: str):
    if model_type == "bunny":
        return BunnyModel()
    elif model_type == "llava":
        return LlavaModel()
    elif model_type == "gpt4o-mini":
        return GPT4OMini()
    elif model_type == "minicpm2":
        return MiniCPMModel()
    elif model_type == "internvl":
        return InternVLModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
