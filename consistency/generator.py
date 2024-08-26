import re
from consistency.utils import get_concat_h


generate_text_only_instructions = """Given two scenes, find up to five similarities between each scene. Output each similarity in a numbered list."""
generate_image_only_instructions = """Given the two side-by-side images, find up to five similarities between each image. Output each similarity in a numbered list."""
generate_both_instructions = """Given two scenes and their corresponding images, find up to five similarities between each scene. Output each similarity in a numbered list."""


def similarity_generator(model, example: dict, mode="text") -> tuple[str, list[str]]:
    """
    Generates a list of statements that describe the similarities between two scenes or images.
    """
    scene_0 = example["description_0"]
    scene_1 = example["description_1"]
    image_0 = example["image_0"]
    image_1 = example["image_1"]

    ## Check if the model has a get_concat_h method for multi-image inference
    get_concat_h_func = get_concat_h
    if hasattr(model, "get_concat_h"):
        get_concat_h_func = model.get_concat_h

    if mode == "text":
        image = None
        prompt = f"{generate_text_only_instructions}\n\nScene 1:\n\n{scene_0}\n\nScene 2:\n\n{scene_1}"
    elif mode == "image":
        image = get_concat_h_func(image_0, image_1)
        prompt = f"<image>\n{generate_image_only_instructions}"
    elif mode == "both":
        image = get_concat_h_func(image_0, image_1)
        prompt = f"<image>\n{generate_both_instructions}\n\nScene 1:\n\n{scene_0}\n\nScene 2:\n\n{scene_1}"
    else:
        raise ValueError("mode should be 'text' or 'image'")
    pred = model.generate(
        prompt, max_new_tokens=128, image=image, start_decode="Similarities:\n\n1."
    )
    response = pred.split("Similarities:\n\n")[-1]
    # process response
    statements = [
        re.sub(r"^\d+\.\s*", "", s)
        for s in response.split("\n")
        if re.match(r"^\d+\.\s*", s)
    ]
    return (response, statements)


def similarity_generator_with_ICL(
    model, example: dict, mode="text", icl_prompt=None, icl_image=None
) -> tuple[str, list[str]]:
    """
    Generates a list of statements that describe the similarities between two scenes or images.
    """
    scene_0 = example["description_0"]
    scene_1 = example["description_1"]
    image_0 = example["image_0"]
    image_1 = example["image_1"]

    ## Check if the model has a get_concat_h method for multi-image inference
    get_concat_h_func = get_concat_h
    if hasattr(model, "get_concat_h"):
        get_concat_h_func = model.get_concat_h

    if mode == "text":
        image = None
        if icl_prompt is None:
            prompt = f"{generate_text_only_instructions}\n\nScene 1:\n\n{scene_0}\n\nScene 2:\n\n{scene_1}"
        else:
            prompt = f"{generate_text_only_instructions}\n\n{icl_prompt}\n\nScene 1:\n\n{scene_0}\n\nScene 2:\n\n{scene_1}"
    elif mode == "image":
        image = get_concat_h_func(image_0, image_1)
        if icl_image is not None:
            image = [icl_image, image]
        if icl_prompt is None:
            prompt = f"{generate_image_only_instructions}\n\n<image>"
        else:
            prompt = f"{generate_image_only_instructions}\n\n{icl_prompt}\n\n<image>"
    elif mode == "both":
        image = get_concat_h_func(image_0, image_1)
        if icl_image is not None:
            image = [icl_image, image]
        if icl_prompt is None:
            prompt = f"{generate_both_instructions}\n\n<image>\n\nScene 1:\n\n{scene_0}\n\nScene 2:\n\n{scene_1}"
        else:
            prompt = f"{generate_both_instructions}\n\n{icl_prompt}\n\n<image>\n\nScene 1:\n\n{scene_0}\n\nScene 2:\n\n{scene_1}"
    else:
        raise ValueError("mode should be 'text' or 'image'")

    pred = model.generate(
        prompt, max_new_tokens=128, image=image, start_decode="Similarities:\n\n1."
    )
    response = pred.split("Similarities:\n\n")[-1]
    if "Differences" in response:
        response = response.split("Differences")[0]
    # process response
    statements = [
        re.sub(r"^\d+\.\s*", "", s)
        for s in response.split("\n")
        if re.match(r"^\d+\.\s*", s)
    ]
    return (response, statements)
