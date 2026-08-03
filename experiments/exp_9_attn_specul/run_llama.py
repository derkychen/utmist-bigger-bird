"""Exp 9 Attn Specul — exp_9_attn_specul_llama on R1-Distill-Llama-8B.

Run on a GPU node:
    source /scratch/$USER/r1-venv/bin/activate
    python exp_9_attn_specul/run_llama.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from eval.imdb.dataset import build_imdb_dataset
from patches.llama.llama_runner import run_llama_experiment
from experiments.exp_9_attn_specul.model_llama import build_model

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
        "exp_9_attn_specul_llama",
        model,
        tokenizer,
        ds,
        train_cfg,
        extra_meta={"window_size": 64, "num_anchors": 4, "verify_every": 4, "verify_kl_weight": 0.1, "base_model": "r1-distill-llama-8b"},
    )


if __name__ == "__main__":
    main()
