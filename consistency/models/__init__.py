from .bunny import BunnyModel
from .llava import LlavaModel


def model_factory(model_type: str):
    if model_type == "bunny":
        return BunnyModel()
    elif model_type == "llava":
        return LlavaModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
