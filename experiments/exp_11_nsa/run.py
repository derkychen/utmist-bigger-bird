import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from eval.imdb.dataset import build_imdb_dataset
from patches.original_patches.runner import run_experiment
from experiments.exp_11_nsa.model import PatchedModel

from experiments.exp_data_train_configs.original_patch_configs import load_data_config, load_train_config

def main():
    model_name = "facebook/bart-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    base_model.config.classifier_dropout = 0.1

    # NSA: compressed + selected + sliding-window branches (arXiv:2502.11089)
    model = PatchedModel(
        base_model,
        block_size=32,
        stride=32,
        topk_blocks=4,
        window_size=128,
    )

    data_cfg = load_data_config()
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=None)

    train_cfg = load_train_config()
    run_experiment(
        "exp_11_nsa",
        model,
        tokenizer,
        ds,
        train_cfg,
        extra_meta={
            "block_size": 32,
            "stride": 32,
            "topk_blocks": 4,
            "window_size": 128,
            "attention": "native_sparse_attention",
            "triton_inference": True,
        },
    )


if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
