from .bunny import BunnyModel
from .llava import LlavaModel
from .llava_next import LlavaNextModel
from .llava_15 import Llava15Model
from .gpt4o import GPT4OMini
from .minicpm import MiniCPMModel
from .internvl import InternVLModel
# from .llava_rlaifv_v import LlavaRLAIFVModel
from .phi35vision import PhiModel


def model_factory(model_type: str):
    if model_type == "bunny":
        return BunnyModel()
    elif model_type == "llava":
        return LlavaModel()
    elif model_type == "llava15":
        return Llava15Model()
    # elif model_type == "llava15":
    #     return LlavaRLAIFVModel()
    elif model_type == "llava_next":
        return LlavaNextModel()
    elif model_type == "gpt4o-mini":
        return GPT4OMini()
    elif model_type == "minicpm2":
        return MiniCPMModel()
    elif model_type == "internvl":
        return InternVLModel()
    elif model_type == "phivision":
        return PhiModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
