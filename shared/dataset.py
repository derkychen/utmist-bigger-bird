import os
from datasets import load_dataset
from dataclasses import dataclass

@dataclass
class DataConfig:
    seed: int = 42
    max_length: int = 768
    train_samples: int = 6000
    eval_samples: int = 1000

def build_imdb_dataset(tokenizer, cfg: DataConfig, fixed_length: int = None):
    ds = load_dataset("stanfordnlp/imdb")
    if cfg.train_samples:
        ds["train"] = ds["train"].shuffle(seed=cfg.seed).select(range(cfg.train_samples))
    if cfg.eval_samples:
        ds["test"] = ds["test"].shuffle(seed=cfg.seed).select(range(cfg.eval_samples))

    def tok_fn(batch):
        if fixed_length is not None:
            return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=fixed_length)
        else:
            return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return {"train": ds["train"], "validation": ds["test"]}
