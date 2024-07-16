import re
import warnings

import pandas as pd
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from consistency.models.bunny import BunnyModel

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    print("Loading dataset")
    dataset = torch.load("dataset.pt")
    print("Loading Model")
    model = BunnyModel()

    statements = []
    # for i, row in tqdm(df.iterrows(), total=len(df)):
        # idx_1 = int(row.original_idx)
        # idx_2 = int(row.distractor_idx)
        # example_1 = dataset["train"][idx_1]
        # example_2 = dataset["train"][idx_2]
        # for modality in ["text", "image", "both"]:
        #     generated_similarity_statements = similarity_generator(
        #         model, tokenizer, example_1, example_2, mode=modality
        #     )
        #     for generated_similarity_statement in generated_similarity_statements:
        #         text_check = similarity_statement_check(
        #             model,
        #             tokenizer,
        #             example_1,
        #             example_2,
        #             statement=generated_similarity_statement,
        #             mode="text",
        #         )
        #         image_check = similarity_statement_check(
        #             model,
        #             tokenizer,
        #             example_1,
        #             example_2,
        #             statement=generated_similarity_statement,
        #             mode="image",
        #         )
        #         both_check = similarity_statement_check(
        #             model,
        #             tokenizer,
        #             example_1,
        #             example_2,
        #             statement=generated_similarity_statement,
        #             mode="both",
        #         )
        #         statements.append(
        #             {
        #                 "dataset_idx": row.dataset_idx,
        #                 "statement": generated_similarity_statement,
        #                 "generated_with": modality,
        #                 "eval_text": text_check,
        #                 "eval_image": image_check,
        #                 "eval_both": both_check,
        #             }
        #         )
    statements_df = pd.DataFrame(statements)
    # save to csv
    statements_df.to_csv("evaluated_statements.csv", index=False)
