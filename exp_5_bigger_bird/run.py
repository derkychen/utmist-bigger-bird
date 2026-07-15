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

    params = dict(window_size=64, local_k=32, num_globals=16, num_teleports=8,
                  diversity_lambda=0.3, teleport_bias=0.5)
    model = PatchedModel(base_model, **params)

    data_cfg = DataConfig(train_samples=6000, eval_samples=1000, max_length=768)
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=None)

    train_cfg = TrainConfig(epochs=3, lr=3e-5)
    run_experiment(
        "exp_5_bigger_bird",
        model, tokenizer, ds, train_cfg,
        extra_meta=params,
    )


if __name__ == "__main__":
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    main()
