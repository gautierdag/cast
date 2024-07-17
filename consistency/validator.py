from consistency.utils import get_concat_h

validate_text_only_instructions = """Given two scenes, does the following statement apply to only one of the images or to both images? Answer with 'one' or 'both'."""
validate_image_only_instructions = """Given the two side-by-side images, does the following statement apply to only one of the images or to both images? Answer with 'one' or 'both'."""
validate_both_instructions = """Given two scenes and their corresponding images, does the following statement apply to only one of the images or to both images? Answer with 'one' or 'both'."""

def similarity_validator(
    model,
    example: dict,
    statement: str,
    mode="text",
) -> str:
    """
    Given a statement, asks whether the statement is true for one or both of the scenes
    """
    scene_0 = example["description_0"]
    scene_1 = example["description_1"]
    image_0 = example["image_0"]
    image_1 = example["image_1"]
    if mode == "text":
        image = None
        prompt = f"{validate_text_only_instructions}\n\nScene 1:\n\n{scene_0}\n\nScene 2:\n\n{scene_1}\n\nStatement:{statement}\n\nAnswer:"
    elif mode == "image":
        image = get_concat_h(image_0, image_1)
        prompt = f"{validate_image_only_instructions}<image>\n\nStatement:{statement}\n\nAnswer:"
    elif mode == "both":
        image = get_concat_h(image_0, image_1)
        prompt = f"{validate_both_instructions}<image>\n\nScene 1:\n\n{scene_0}\n\nScene 2:\n\n{scene_1}\n\nStatement:{statement}\n\nAnswer:"
    else:
        raise ValueError("mode should be 'text' or 'image'")
    pred = model.generate(prompt, max_new_tokens=16, image=image)
    pred = pred.split("Answer:")[-1].lower().strip(".").strip()
    return pred
