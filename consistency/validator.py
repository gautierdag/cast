from consistency.utils import get_concat_h

validate_text_only_instructions = """Given two scenes, does the following statement apply to one of the images or to both images? Answer with 'one' or 'both'."""
validate_image_only_instructions = """Given the two side-by-side images, does the following statement apply to one of the images or to both images? Answer with 'one' or 'both'."""
validate_both_instructions = """Given two scenes and their corresponding images, does the following statement apply to one of the images or to both images? Answer with 'one' or 'both'."""


def similarity_validator(
    model,
    example_1: dict,
    example_2: dict,
    statement: str,
    mode="text",
) -> str:
    """
    Given a statement, asks whether the statement is true for one or both of the scenes
    """
    scene_1 = example_1["description"]
    scene_2 = example_2["description"]
    image_1 = example_1["image"]
    image_2 = example_2["image"]
    if mode == "text":
        image = None
        prompt = f"{validate_text_only_instructions}\n\nScene 1:\n\n{scene_1}\n\nScene 2:\n\n{scene_2}\n\nStatement:{statement}\n\nAnswer:"
    elif mode == "image":
        image = get_concat_h(image_1, image_2)
        prompt = f"{validate_image_only_instructions}<image>\n\nStatement:{statement}\n\nAnswer:"
    elif mode == "both":
        image = get_concat_h(image_1, image_2)
        prompt = f"{validate_both_instructions}<image>\n\nScene 1:\n\n{scene_1}\n\nScene 2:\n\n{scene_2}\n\nStatement:{statement}\n\nAnswer:"
    else:
        raise ValueError("mode should be 'text' or 'image'")
    pred = model.generate(prompt, max_new_tokens=16, image=image)
    pred = pred.split("Answer:")[-1].lower().strip(".").strip()
    return pred
