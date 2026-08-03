from dataclasses import dataclass
from pathlib import Path
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parents[2] / "configs"

@dataclass
class DataConfig:
    seed: int = 42
    max_length: int = 768
    train_samples: int = 6000
    eval_samples: int = 1000