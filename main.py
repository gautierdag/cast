import logging
import warnings

import hydra
import pandas as pd
from tqdm import tqdm

import wandb
from consistency.config import EvalConfig
from consistency.dataset import SimilarityPairDataset
from consistency.generator import similarity_generator
from consistency.models import model_factory
from consistency.validator import similarity_validator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg):
    logger.info(cfg)
    cfg = EvalConfig(**dict(cfg))

    logger.info("Loading dataset")
    dataset = SimilarityPairDataset(data_dir="data")
    logger.info("Loading Model")
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
        for modality in cfg.consistency.generate_modalities:
            generated_text, generated_similarity_statements = similarity_generator(
                model, example, mode=modality
            )
            for generated_idx, generated_similarity_statement in enumerate(
                generated_similarity_statements
            ):
                statement = {
                    "dataset_idx": example["id"],
                    "generated_idx": generated_idx,  # can track whether position of generated statement has effect on evaluation
                    "statement": generated_similarity_statement,
                    "generated_with": modality,
                    "generated_statement_text": generated_text,
                }

                for prompt_type in cfg.consistency.validate_prompt_type:
                    for validate_modality in cfg.consistency.validate_modalities:
                        validate_response = similarity_validator(
                            model,
                            example,
                            statement=generated_similarity_statement,
                            mode=validate_modality,
                            prompt_type=prompt_type,
                        )
                        statement[f"validate_{validate_modality}_{prompt_type}"] = (
                            validate_response
                        )
                statements.append(statement)
    statements_df = pd.DataFrame(statements)
    # save to wandb
    wandb.log({"evaluated_statements": wandb.Table(dataframe=statements_df)})
    wandb.finish()


if __name__ == "__main__":
    main()
