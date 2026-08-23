#!/usr/bin/env python3
"""Run a single LRA/RULER experiment on R1-Distill-Llama-8B with LoRA.

Usage:
  python -m eval.lra_llama.run --task listops --exp 0 --seq 4096 --size lra-smoke
  python -m eval.ruler_llama.run --task niah --exp 1 --seq 4096 --depth 0.5 --size ruler-smoke

This replaces the from-scratch BART encoder with R1-Llama-8B + LoRA, so the
long-context eval tracks use the same pretrained backbone as the IMDb experiments.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import AutoTokenizer

from eval.lra.lra_dataset import build_lra_dataset
from eval.ruler.ruler_dataset import build_ruler_dataset
from eval.lra_llama.lra_llama_model import build_lra_llama_model
from eval.lra_llama.lra_llama_dataset import build_llama_dataset, get_num_labels
from patches.llama.llama_runner import run_llama_experiment

from config_schema.trainer.llama import LlamaTrainConfig

CONFIG_DIR = Path(__file__).parents[2] / "configs"
LRA_CFG = OmegaConf.load(CONFIG_DIR / "benchmarks" / "lra.yaml")
LRA_COMPUTE = LRA_CFG.compute
LRA_DEFAULT_SEQ = LRA_CFG.default_seq
LRA_TASK_INFO = LRA_CFG.task_info

RULER_CFG = OmegaConf.load(CONFIG_DIR / "benchmarks" / "ruler.yaml")
RULER_COMPUTE = RULER_CFG.compute
RULER_DEFAULT_SEQ = RULER_CFG.default.seq_len
DEFAULT_DEPTH = RULER_CFG.default.depth
RULER_TASK_INFO = RULER_CFG.task_info


MODEL_PATH = os.path.join(
    os.environ.get("SCRATCH", "/scratch/$USER"),
    "models", "DeepSeek-R1-Distill-Llama-8B"
)

ALL_TASKS = {
    "listops": ("lra", LRA_COMPUTE, LRA_DEFAULT_SEQ),
    "text": ("lra", LRA_COMPUTE, LRA_DEFAULT_SEQ),
    "niah": ("ruler", RULER_COMPUTE, RULER_DEFAULT_SEQ),
    "mq_niah": ("ruler", RULER_COMPUTE, RULER_DEFAULT_SEQ),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run a single LRA/RULER experiment on R1-Llama-8B with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", choices=list(ALL_TASKS.keys()), help="Task name")
    parser.add_argument("--exp", type=int, choices=[0, 1, 8, 13, 14, 15, 18], help="Experiment number")
    parser.add_argument("--size", default="lra-smoke", help="Compute preset (lra-smoke, ruler-smoke, etc.)")
    parser.add_argument("--seq", type=int, help="Context window (overrides task default)")
    parser.add_argument("--depth", type=float, help="Needle depth 0..1 (RULER only)")
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--accum", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.task is None or args.exp is None:
        parser.error("--task and --exp are required")

    track, compute_presets, default_seqs = ALL_TASKS[args.task]
    preset_name = args.size if args.size in compute_presets else list(compute_presets.keys())[0]
    compute = dict(compute_presets[preset_name])

    if args.train_samples:
        compute["train_samples"] = args.train_samples
    if args.eval_samples:
        compute["eval_samples"] = args.eval_samples
    if args.batch:
        compute["batch_size"] = args.batch
    if args.accum:
        compute["grad_accum"] = args.accum
    if args.epochs:
        compute["epochs"] = args.epochs

    seq_len = args.seq or default_seqs[args.task]
    num_labels = get_num_labels(args.task)

    print(f"\n{'='*70}")
    print(f"{track.upper()} Llama: {args.task} | exp {args.exp} | seq {seq_len} | {preset_name}")
    print(f"{'='*70}\n")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding so the last token is always a real token (for last-token pooling)
    tokenizer.padding_side = "left"

    # --- Build original dataset (byte-level) ---
    if track == "lra":
        data = build_lra_dataset(
            task=args.task,
            seq_len=seq_len,
            train_samples=compute["train_samples"],
            eval_samples=compute["eval_samples"],
            seed=args.seed,
        )
        pair = LRA_TASK_INFO[args.task]["pair"]
    else:
        depth = args.depth if args.depth is not None else DEFAULT_DEPTH
        data = build_ruler_dataset(
            task=args.task,
            seq_len=seq_len,
            needle_depth=depth,
            train_samples=compute["train_samples"],
            eval_samples=compute["eval_samples"],
            seed=args.seed,
        )
        pair = RULER_TASK_INFO[args.task]["pair"]

    original_ds = {"train": data["train"], "validation": data["validation"]}

    # --- Convert to Llama-tokenized format ---
    print("Converting dataset to Llama tokenization...")
    ds = build_llama_dataset(args.task, tokenizer, original_ds, seq_len, pair=pair)
    print(f"  Train: {len(ds['train'])} samples, Eval: {len(ds['validation'])} samples")

    # --- Build model ---
    model, exp_name, meta = build_lra_llama_model(
        exp_num=args.exp,
        num_labels=num_labels,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model moved to {device}")

    # --- Train config ---
    train_schema = OmegaConf.structured(LlamaTrainConfig)
    train_cfg = OmegaConf.load(CONFIG_DIR / "trainer" / "llama.yaml")
    
    train_cfg.epochs = compute["epochs"]
    train_cfg.per_device_train_bs = compute["batch_size"]
    train_cfg.per_device_eval_bs = compute["batch_size"]
    train_cfg.grad_accum_steps = compute["grad_accum"]
    train_cfg.lr = args.lr
    train_cfg.lora_r = args.lora_r
    train_cfg.lora_alpha = args.lora_r * 2
    train_cfg.gradient_checkpointing=  True

    train_cfg = OmegaConf.merge(train_schema, train_cfg)

    # --- Run ---
    run_name = f"{exp_name}_llama_{track}_{args.task}_seq{seq_len}_{preset_name}"
    run_llama_experiment(
        run_name,
        model,
        tokenizer,
        ds,
        train_cfg,
        extra_meta={
            **meta,
            "track": track,
            "task": args.task,
            "seq_length": seq_len,
            "compute_preset": preset_name,
            "base_model": "r1-distill-llama-8b",
        },
    )

    print(f"\n=== DONE: {run_name} ===")
    return 0


if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    raise SystemExit(main())
