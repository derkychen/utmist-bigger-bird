"""Exp 7 Layer Adaptive — exp_7_layer_adaptive_llama on R1-Distill-Llama-8B.

Run on a GPU node:
    source /scratch/$USER/r1-venv/bin/activate
    python exp_7_layer_adaptive/run_llama.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from eval.imdb.dataset import build_imdb_dataset
from patches.llama.llama_runner import run_llama_experiment
from experiments.exp_7_layer_adaptive.model_llama import build_model

from experiments.exp_data_train_configs.llama_configs import load_data_config, load_train_config

MODEL_PATH = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B")


def main():
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(model_path=MODEL_PATH)
    model = model.to("cuda")

    data_cfg = load_data_config()
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=None)

    train_cfg = load_train_config()
    run_llama_experiment(
        "exp_7_layer_adaptive_llama",
        model,
        tokenizer,
        ds,
        train_cfg,
        extra_meta={"k_early": 192, "k_mid": 64, "k_late": 32, "low_rank_dim": 16, "base_model": "r1-distill-llama-8b"},
    )


if __name__ == "__main__":
    main()
