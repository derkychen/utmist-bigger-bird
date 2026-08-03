from config_schema.imdb.data import DataConfig
from config_schema.trainer.encoder import TrainConfig

from pathlib import Path
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parents[2] / "configs"

def load_data_config() -> DataConfig:
    data_schema = OmegaConf.structured(DataConfig)
    data_cfg = OmegaConf.load(CONFIG_DIR / "benchmarks" / "imdb.yaml")
    # Overrides
    data_cfg.train_samples = 6000
    data_cfg.eval_samples = 1000
    data_cfg.max_length = 768

    return OmegaConf.merge(data_schema, data_cfg)

def load_train_config() -> TrainConfig:
    train_schema = OmegaConf.structured(TrainConfig)
    train_cfg = OmegaConf.load(CONFIG_DIR / "trainer" / "encoder.yaml")
    # Overrides
    train_cfg.epochs = 3
    train_cfg.lr = 3e-5

    return OmegaConf.merge(train_schema, train_cfg)