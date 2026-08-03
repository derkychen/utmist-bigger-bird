from config_schema.imdb.data import DataConfig
from config_schema.trainer.llama import LlamaTrainConfig

from pathlib import Path
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parents[1] / "configs"

def load_data_config() -> DataConfig:
    data_schema = OmegaConf.structured(DataConfig)
    data_cfg = OmegaConf.load(CONFIG_DIR / "benchmarks" / "imdb.yaml")
    # Overrides
    data_cfg.train_samples = 2000
    data_cfg.eval_samples = 500
    data_cfg.max_length = 512

    return OmegaConf.merge(data_schema, data_cfg)

def load_train_config() -> LlamaTrainConfig:
    train_schema = OmegaConf.structured(LlamaTrainConfig)
    train_cfg = OmegaConf.load(CONFIG_DIR / "trainer" / "llama.yaml")
    # Overrides
    train_cfg.epochs = 3
    train_cfg.per_device_train_bs = 1
    train_cfg.grad_accum_steps = 16
    train_cfg.lr = 2e-4
    train_cfg.lora_r = 16
    train_cfg.lora_alpha = 32

    return OmegaConf.merge(train_schema, train_cfg)