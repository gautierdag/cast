import logging
import warnings

import hydra
import pandas as pd
import wandb
from tqdm import tqdm


from consistency.config import EvalConfig
from consistency.dataset import SimilarityPairDataset
from consistency.models import model_factory
from consistency.validator import similarity_validator
from consistency.generator import similarity_generator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg):
    logger.info(cfg)
    cfg = EvalConfig(**dict(cfg))

    print("Loading dataset")
    dataset = SimilarityPairDataset(data_dir="data")
    print("Loading Model")
    model = model_factory(cfg.consistency.model)

    # Add requirement for wandb core
    wandb.require("core")

    wandb.init(
        name=f"{cfg.consistency.model}-{cfg.consistency.special_run_name}",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config=cfg.model_dump(),
    )

    statements = []
    for example in tqdm(dataset):
        for modality in ["text", "image", "both"]:
            generated_similarity_statements = similarity_generator(
                model, example, mode=modality
            )
            print(modality)
            print(generated_similarity_statements)
            for generated_similarity_statement in generated_similarity_statements:
                text_check = similarity_validator(
                    model,
                    example,
                    statement=generated_similarity_statement,
                    mode="text",
                )
                image_check = similarity_validator(
                    model,
                    example,
                    statement=generated_similarity_statement,
                    mode="image",
                )
                both_check = similarity_validator(
                    model,
                    example,
                    statement=generated_similarity_statement,
                    mode="both",
                )
                statements.append(
                    {
                        "dataset_idx": example["id"],
                        "statement": generated_similarity_statement,
                        "generated_with": modality,
                        "eval_text": text_check,
                        "eval_image": image_check,
                        "eval_both": both_check,
                    }
                )
    statements_df = pd.DataFrame(statements)
    # save to wandb
    wandb.log({"evaluated_statements": wandb.Table(dataframe=statements_df)})
    wandb.finish()


if __name__ == "__main__":
    main()
