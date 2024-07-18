from consistency.utils import get_concat_h

validation_instruction_prompts = {
    "both|one": {
        "text": """Given two scenes, does the following statement apply to only one of the images or to both images? Answer with 'one' or 'both'.""",
        "image": """Given the two side-by-side images, does the following statement apply to only one of the images or to both images? Answer with 'one' or 'both'.""",
        "both": """Given two scenes and their corresponding images, does the following statement apply to only one of the images or to both images? Answer with 'one' or 'both'.""",
    },
    "true|false": {
        "text": """Given two scenes, is the following statement true for both of the scenes? Answer with 'true' or 'false' if the statement is untrue or only true for one of the scenes.""",
        "image": """Given the two side-by-side images, is the following statement true for both of the images? Answer with 'true' or 'false' if the statement is untrue or only true for one of the images.""",
        "both": """Given two scenes and their corresponding images, is the following statement true for both of the scenes? Answer with 'true' or 'false' if the statement is untrue or only true for one of the scenes.""",
    },
    "yes|no": {
        "text": """Given two descriptions, is the following statement describe both of the descriptions? Answer with 'yes' or 'no' if the statement is not applicable to one of the descriptions.""",
        "image": """Given the two side-by-side images, is the following statement describe both of the images? Answer with 'yes' or 'no' if the statement is not applicable to one of the images.""",
        "both": """Given two descriptions and their corresponding images, is the following statement describe both of the descriptions? Answer with 'yes' or 'no' if the statement is not applicable to one of the descriptions.""",
    },
}


def similarity_validator(
    model,
    example: dict,
    statement: str,
    mode="text",
    prompt_type="both|one",
) -> str:
    """
    Given a statement, asks whether the statement is true for one or both of the scenes
    """
    scene_0 = example["description_0"]
    scene_1 = example["description_1"]
    image_0 = example["image_0"]
    image_1 = example["image_1"]

    if prompt_type not in validation_instruction_prompts:
        raise ValueError(
            f"prompt_type should be one of {list(validation_instruction_prompts.keys())}"
        )

    instructions = validation_instruction_prompts[prompt_type][mode]
    if mode == "text":
        image = None
        prompt = f"{instructions}\n\nScene 1:\n\n{scene_0}\n\nScene 2:\n\n{scene_1}\n\nStatement:{statement}"
    elif mode == "image":
        image = get_concat_h(image_0, image_1)
        prompt = f"<image>\n{instructions}\n\nStatement:{statement}"
    elif mode == "both":
        image = get_concat_h(image_0, image_1)
        prompt = f"<image>\n{instructions}\n\nScene 1:\n\n{scene_0}\n\nScene 2:\n\n{scene_1}\n\nStatement:{statement}"
    else:
        raise ValueError("mode should be 'text' or 'image'")
    pred = model.generate(
        prompt, max_new_tokens=16, image=image, start_decode="Answer:"
    )
    pred = pred.split("Answer:")[-1].lower().strip(".").strip()
    return pred
