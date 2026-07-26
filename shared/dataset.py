import os
import torch
from torch.utils.data import Dataset as TorchDataset
from dataclasses import dataclass

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False


@dataclass
class DataConfig:
    seed: int = 42
    max_length: int = 768
    train_samples: int = 6000
    eval_samples: int = 1000


class _SyntheticDataset(TorchDataset):
    """Fallback dataset when HuggingFace `datasets` library is unavailable.

    Generates synthetic token sequences with binary labels. Sufficient for
    testing model architecture / training pipeline; not for real benchmarks.
    """

    def __init__(self, n_samples, seq_len, vocab_size, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(4, vocab_size, (n_samples, seq_len), generator=g)
        self.attention_mask = torch.ones(n_samples, seq_len, dtype=torch.long)
        self.labels = torch.randint(0, 2, (n_samples,), generator=g)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def build_imdb_dataset(tokenizer, cfg: DataConfig, fixed_length: int = None):
    seq_len = fixed_length or cfg.max_length

    if not _HAS_DATASETS:
        print("[dataset] HuggingFace `datasets` unavailable — using synthetic data")
        vocab_size = getattr(tokenizer, "vocab_size", 128256)
        return {
            "train": _SyntheticDataset(cfg.train_samples, seq_len, vocab_size, cfg.seed),
            "validation": _SyntheticDataset(cfg.eval_samples, seq_len, vocab_size, cfg.seed + 1),
        }

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
