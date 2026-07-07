import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shared.dataset import build_imdb_dataset, DataConfig
from shared.runner import run_experiment, TrainConfig
from model import PatchedModel


def main():
    model_name = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    base_model.config.classifier_dropout = 0.1

    model = PatchedModel(
        base_model,
        shard_size=32,
        local_blocks=2,
        stride_blocks=16,
        use_sink=True,
        dense_layers={0},
    )

    data_cfg = DataConfig(train_samples=500, eval_samples=100, max_length=256)
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=None)

    train_cfg = TrainConfig(epochs=2, lr=3e-5)
    run_experiment(
        "exp_12_s2_hhst",
        model,
        tokenizer,
        ds,
        train_cfg,
        extra_meta={
            "paper": "arXiv:2407.17678",
            "shard_size": 32,
            "local_blocks": 2,
            "stride_blocks": 16,
            "use_sink": True,
            "dense_layers": [0],
        },
    )


if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
