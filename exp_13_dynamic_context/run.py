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

    # Dynamic context window: cap attention to 4k tokens regardless of input length
    params = dict(drop_after_layer=3, target_budget=4096, chunk_size=8192)
    model = PatchedModel(base_model, **params)

    # For long-context testing, use fixed-length padding to stress the full window
    data_cfg = DataConfig(train_samples=500, eval_samples=100, max_length=2048)
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=None)

    train_cfg = TrainConfig(epochs=2, lr=3e-5)
    run_experiment("exp_13_dynamic_context", model, tokenizer, ds, train_cfg, extra_meta=params)


if __name__ == "__main__":
    main()
