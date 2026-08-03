import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from eval.imdb.dataset import build_imdb_dataset
from patches.original_patches.runner import run_experiment
from experiments.exp_7_layer_adaptive.model import PatchedModel

from experiments.exp_data_train_configs.original_patch_configs import load_data_config, load_train_config

def main():
    model_name = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    base_model.config.classifier_dropout = 0.1

    params = dict(k_early=192, k_mid=64, k_late=32, low_rank_dim=16)
    model = PatchedModel(base_model, **params)

    data_cfg = load_data_config()
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=None)

    train_cfg = load_train_config()
    run_experiment("exp_7_layer_adaptive", model, tokenizer, ds, train_cfg, extra_meta=params)


if __name__ == "__main__":
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    main()
