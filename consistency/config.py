from pydantic import BaseModel


class ConsistencyConfig(BaseModel):
    model: str
    num_generations: int
    output_dir: str
    special_run_name: str = ""
    validate_modalities: list[str] = ["text", "image", "both"]
    generate_modalities: list[str] = ["text", "image", "both"]
    validate_prompt_type: list[str] = ["both|one", "true|false", "yes|no"]


class WandbConfig(BaseModel):
    project: str = "consistency"
    entity: str = "itl"
    mode: str = "online"


class LaunchConfig(BaseModel):
    command: str
    job_name: str
    gpu_limit: int
    gpu_product: str
    cpu_request: int
    ram_request: str
    interactive: bool = False
    namespace: str = "informatics"
    env_vars: dict[str, dict[str, str]]


class EvalConfig(BaseModel):
    consistency: ConsistencyConfig
    wandb: WandbConfig
    launch: LaunchConfig
