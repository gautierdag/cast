import re
from consistency.utils import get_concat_h


generate_text_only_instructions = """Given two scenes, find up to five similarities between each scene. Output each similarity in a list."""
generate_image_only_instructions = """Given the two side-by-side images, find up to five similarities between each image.  Output each similarity in a list."""
generate_both_instructions = """Given two scenes and their corresponding images, find up to five similarities between each scene.  Output each similarity in a list."""


def similarity_generator(model, example: dict, mode="text") -> list[str]:
    """
    Generates a list of statements that describe the similarities between two scenes or images.
    """
    scene_0 = example["description_0"]
    scene_1 = example["description_1"]
    image_0 = example["image_0"]
    image_1 = example["image_1"]
    if mode == "text":
        image = None
        prompt = f"{generate_text_only_instructions}\n\nScene 1:\n\n{scene_0}\n\nScene 2:\n\n{scene_1}\n\nSimilarities:\n\n"
    elif mode == "image":
        image = get_concat_h(image_0, image_1)
        prompt = f"{generate_image_only_instructions}<image>\n\nSimilarities:\n\n"
    elif mode == "both":
        image = get_concat_h(image_0, image_1)
        prompt = f"{generate_both_instructions}<image>\n\nScene 1:\n\n{scene_0}\n\nScene 2:\n\n{scene_1}\n\nSimilarities:\n\n"
    else:
        raise ValueError("mode should be 'text' or 'image'")
    pred = model.generate(prompt, max_new_tokens=128, image=image)
    statements = pred.split("Similarities:\n\n")[-1].split("\n")
    statements = [re.sub(r"^\d+\.\s*", "", s) for s in statements]
    return statements
