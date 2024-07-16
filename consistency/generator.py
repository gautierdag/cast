import re
from consistency.utils import get_concat_h


text_only_instructions = """Given two scenes, find up to five similarities between each scene. Output each similarity in a list."""
image_only_instructions = """Given the two side-by-side images, find up to five similarities between each image.  Output each similarity in a list."""
both_instructions = """Given two scenes and their corresponding images, find up to five similarities between each scene.  Output each similarity in a list."""


def similarity_generator(
    model, example_1: dict, example_2: dict, mode="text"
) -> list[str]:
    """
    Generates a list of statements that describe the similarities between two scenes or images.
    """
    scene_1 = example_1["description"]
    scene_2 = example_2["description"]
    image_1 = example_1["image"]
    image_2 = example_2["image"]
    if mode == "text":
        image = None
        prompt = f"{text_only_instructions}\n\nScene 1:\n\n{scene_1}\n\nScene 2:\n\n{scene_2}\n\nSimilarities:\n\n"
    elif mode == "image":
        image = get_concat_h(image_1, image_2)
        prompt = f"{image_only_instructions}<image>\n\nSimilarities:\n\n"
    elif mode == "both":
        image = get_concat_h(image_1, image_2)
        prompt = f"{both_instructions}<image>\n\nScene 1:\n\n{scene_1}\n\nScene 2:\n\n{scene_2}\n\nSimilarities:\n\n"
    else:
        raise ValueError("mode should be 'text' or 'image'")
    pred = model.generate(prompt, max_new_tokens=128, image=image)
    statements = pred.split("Similarities:\n\n")[-1].split("\n")
    statements = [re.sub(r"^\d+\.\s*", "", s) for s in statements]
    return statements
