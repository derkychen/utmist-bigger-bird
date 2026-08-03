"""Exp 1 — DeepSeek Top-K on R1-Distill-Llama-8B.

Run on a GPU node:
    source /scratch/$USER/r1-venv/bin/activate
    python exp_1_deepseek_topk/run_llama.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from eval.imdb.dataset import build_imdb_dataset
from patches.llama.llama_runner import run_llama_experiment
from experiments.exp_1_deepseek_topk.model_llama import build_model

from experiments.exp_data_train_configs.llama_configs import load_data_config, load_train_config

MODEL_PATH = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B")

def main():
    torch.set_float32_matmul_precision("high")

    # 1. Tokenizer (Llama-3)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Build patched model with DeepSeek top-k attention + LoRA
    model = build_model(
        model_path=MODEL_PATH,
        top_k=64,
        low_rank_dim=16,
    )
    model = model.to("cuda")

    # 3. Dataset — smaller than BART runs; 8B + LoRA is slower per step
    data_cfg = load_data_config()
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=None)

    # 4. Train + evaluate
    train_cfg = load_train_config()
    run_llama_experiment(
        "exp_1_deepseek_topk_llama",
        model,
        tokenizer,
        ds,
        train_cfg,
        extra_meta={"top_k": 64, "low_rank_dim": 16, "base_model": "r1-distill-llama-8b"},
    )


if __name__ == "__main__":
    main()
