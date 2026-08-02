"""Unified test runner for Llama-3 IMDb experiments on multi-GPU.

Usage (single GPU):
    python run_llama_tests.py --exp 0 --size small

Usage (4x H100 DDP via torchrun):
    torchrun --nproc_per_node=4 run_llama_tests.py --exp 0 --size small

Size presets:
    tiny:   100 train / 50 eval / 128 max_len / 1 epoch   (~1 min)
    small:  500 train / 100 eval / 256 max_len / 1 epoch   (~3 min)
    medium: 2000 train / 500 eval / 512 max_len / 2 epochs  (~15 min)
    large:  6000 train / 1000 eval / 768 max_len / 3 epochs (~45 min)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
from transformers import AutoTokenizer
from shared.dataset import build_imdb_dataset
from shared.llama_runner import run_llama_experiment

from config_schema.imdb.data import DataConfig
from config_schema.trainer.llama import LlamaTrainConfig

from pathlib import Path
from omegaconf import OmegaConf

# --- Experiment registry ---
EXPERIMENTS = {
    0:  ("exp_0_baseline",            "exp_0_baseline.model_llama"),
    1:  ("exp_1_deepseek_topk",       "exp_1_deepseek_topk.model_llama"),
    8:  ("exp_8_token_drop",          "exp_8_token_drop.model_llama"),
    13: ("exp_13_dynamic_context",    "exp_13_dynamic_context.model_llama"),
    14: ("exp_14_token_drop_deepseek", "exp_14_token_drop_deepseek.model_llama"),
    15: ("exp_15_bigger_bird",        "exp_15_bigger_bird.model_llama"),
}

# --- Size presets ---
SIZE_PRESETS = {
    "tiny":         {"train": 100,  "eval": 50,  "max_len": 128,  "epochs": 1, "bs": 2, "accum": 4},
    "small":        {"train": 500,  "eval": 100, "max_len": 256,  "epochs": 2, "bs": 4, "accum": 4},
    "small-long":   {"train": 500,  "eval": 100, "max_len": 4096, "epochs": 2, "bs": 1, "accum": 8},
    "medium":       {"train": 2000, "eval": 500, "max_len": 512,  "epochs": 3, "bs": 2, "accum": 8},
    "medium-long":  {"train": 2000, "eval": 500, "max_len": 4096, "epochs": 3, "bs": 1, "accum": 16},
    "large":        {"train": 6000, "eval": 1000,"max_len": 1024, "epochs": 3, "bs": 1, "accum": 16},
}

MODEL_PATH = os.path.join(
    os.environ.get("SCRATCH", "/scratch/$USER"),
    "models", "DeepSeek-R1-Distill-Llama-8B"
)

CONFIG_DIR = Path(__file__).parent / "configs"

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", type=int, required=True, help="Experiment number (0,1,8,13,14,15)")
    parser.add_argument("--size", choices=list(SIZE_PRESETS.keys()), default="small", help="Dataset/training size preset")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    args = parser.parse_args()

    if args.exp not in EXPERIMENTS:
        print(f"ERROR: exp {args.exp} not in registry. Available: {list(EXPERIMENTS.keys())}")
        return 1

    exp_name, module_name = EXPERIMENTS[args.exp]
    preset = SIZE_PRESETS[args.size]

    torch.set_float32_matmul_precision("high")

    # --- DDP setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    if is_main:
        print(f"=== Experiment {args.exp}: {exp_name} ===")
        print(f"Size: {args.size} | DDP: world_size={world_size} | local_rank={local_rank}")
        print(f"Data: train={preset['train']} eval={preset['eval']} max_len={preset['max_len']}")
        print(f"Train: epochs={preset['epochs']} bs={preset['bs']} accum={preset['accum']} lr={args.lr}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Build model ---
    import importlib
    mod = importlib.import_module(module_name)
    model = mod.build_model(model_path=MODEL_PATH)

    # With DDP, Trainer handles device placement. Without DDP, move to cuda.
    if world_size == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        if is_main:
            print(f"Single-GPU: moved model to {device}")
    else:
        if is_main:
            print(f"DDP: world_size={world_size}, Trainer will handle device placement")

    # --- Dataset ---
    data_schema = OmegaConf.structured(DataConfig)
    data_cfg = OmegaConf.load(CONFIG_DIR / "imdb" / "data.yaml")
    # Overrides
    data_cfg.train_samples = preset["train"]
    data_cfg.eval_samples = preset["eval"]
    data_cfg.max_length = preset["max_len"]
    data_cfg = OmegaConf.merge(data_schema, data_cfg)
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=None)

    # --- Train config ---
    train_schema = OmegaConf.structured(LlamaTrainConfig)
    train_cfg = OmegaConf.load(CONFIG_DIR / "trainer" / "llama.yaml")
    # Overrides
    train_cfg.epochs = preset["epochs"]
    train_cfg.per_device_train_bs = preset["bs"]
    train_cfg.per_device_eval_bs = preset["bs"]
    train_cfg.grad_accum_steps = preset["accum"]
    train_cfg.lr = args.lr
    train_cfg.lora_r = args.lora_r
    train_cfg.lora_alpha = args.lora_r * 2
    train_cfg.gradient_checkpointing = True

    train_cfg = OmegaConf.merge(train_schema, train_cfg)

    # --- Run ---
    run_name = f"{exp_name}_llama_{args.size}"
    run_llama_experiment(
        run_name,
        model,
        tokenizer,
        ds,
        train_cfg,
        extra_meta={
            "base_model": "r1-distill-llama-8b",
            "size_preset": args.size,
            "world_size": world_size,
        },
    )

    if is_main:
        print(f"\n=== DONE: {run_name} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
