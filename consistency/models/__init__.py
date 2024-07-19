from .bunny import BunnyModel
from .llava import LlavaModel
from .gpt4o import GPT4OMini


def model_factory(model_type: str):
    if model_type == "bunny":
        return BunnyModel()
    elif model_type == "llava":
        return LlavaModel()
    elif model_type == "gpt4o-mini":
        return GPT4OMini()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
