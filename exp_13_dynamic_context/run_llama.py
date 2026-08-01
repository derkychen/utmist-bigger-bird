"""Exp 13 Dynamic Context — exp_13_dynamic_context_llama on R1-Distill-Llama-8B.

Run on a GPU node:
    source /scratch/$USER/r1-venv/bin/activate
    python exp_13_dynamic_context/run_llama.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from shared.dataset import build_imdb_dataset, DataConfig
from shared.llama_runner import run_llama_experiment, LlamaTrainConfig
from exp_13_dynamic_context.model_llama import build_model

MODEL_PATH = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B")


def main():
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(model_path=MODEL_PATH)
    model = model.to("cuda")

    data_cfg = DataConfig(train_samples=2000, eval_samples=500, max_length=512)
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=None)

    train_cfg = LlamaTrainConfig(
        epochs=3,
        per_device_train_bs=1,
        grad_accum_steps=16,
        lr=2e-4,
        lora_r=16,
        lora_alpha=32,
    )
    run_llama_experiment(
        "exp_13_dynamic_context_llama",
        model,
        tokenizer,
        ds,
        train_cfg,
        extra_meta={"drop_after_layer": 3, "target_budget": 4096, "chunk_size": 8192, "base_model": "r1-distill-llama-8b"},
    )


if __name__ == "__main__":
    main()
