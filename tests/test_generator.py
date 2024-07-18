from unittest.mock import MagicMock
from consistency.generator import similarity_generator
from consistency.validator import similarity_validator


def test_generate_text():
    mock_model = MagicMock()
    # set .generate to return a fixed string
    fake_return = "1. both of the images are photos\n2. both of the scenes have pets"
    mock_model.generate.return_value = fake_return
    example = {
        "description_0": "A photo of a cat",
        "description_1": "A photo of a dog",
        "image_0": None,
        "image_1": None,
    }
    full_response, generated_similarity_statements = similarity_generator(
        mock_model, example, mode="text"
    )
    assert generated_similarity_statements[0] == "both of the images are photos"
    assert generated_similarity_statements[1] == "both of the scenes have pets"
    assert full_response == fake_return


def test_validate_text():
    mock_model = MagicMock()
    # set .generate to return a fixed string
    mock_model.generate.return_value = "applies to both"
    example = {
        "description_0": "A photo of a cat",
        "description_1": "A photo of a dog",
        "image_0": None,
        "image_1": None,
    }
    generated_similarity_validation = similarity_validator(
        mock_model, example, statement="both of the images are photos", mode="text"
    )
    assert generated_similarity_validation == "applies to both"
