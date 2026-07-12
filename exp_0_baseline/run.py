import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shared.dataset import build_imdb_dataset, DataConfig
from shared.runner import run_experiment, TrainConfig

def main():
    model_name = "facebook/bart-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Use PyTorch's fused scaled-dot-product attention (Flash / mem-efficient) for the
    # dense baseline. There is no benefit to a hand-written Triton kernel over cuDNN/Flash here.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, attn_implementation="sdpa"
    )
    model.config.classifier_dropout = 0.1

    data_cfg = DataConfig(train_samples=6000, eval_samples=1000, max_length=768)
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=None)

    train_cfg = TrainConfig(epochs=3, lr=3e-5)
    run_experiment(
        "exp_0_baseline",
        model,
        tokenizer,
        ds,
        train_cfg,
        extra_meta={"attention": "full_dense", "model": "bart-base"}
    )

if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
