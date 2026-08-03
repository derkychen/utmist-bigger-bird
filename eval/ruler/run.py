#!/usr/bin/env python3
"""Run a single RULER long-context experiment (full 13-task suite).

Usage:
  python -m eval.ruler.run --task niah_single_1 --exp 1 --seq 4096 --depth 0.5
  python -m eval.ruler.run --task niah_multikey_1 --exp 7 --seq 8192 --depth 0.9
  python -m eval.ruler.run --task vt --exp 0 --seq 4096 --size ruler-report
  python -m eval.ruler.run --task cwe --exp 0 --seq 4096
  python -m eval.ruler.run --task qa_2 --exp 1 --seq 4096 --depth 0.5
  python -m eval.ruler.run --list
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from eval.lra.lra_model import build_lra_model
from eval.ruler.ruler_dataset import TASK_INFO, build_ruler_dataset
from patches.original_patches.runner import run_lra
from eval.ruler.presets import DEFAULT_DEPTH, DEFAULT_SEQ, RULER_COMPUTE

from config_schema.trainer.encoder import TrainConfig

CONFIG_DIR = Path(__file__).parents[2] / "configs"
EXPERIMENT_CONFIGS = OmegaConf.load(CONFIG_DIR / "experiments.yaml")
RULER_CFG = OmegaConf.load(CONFIG_DIR / "benchmarks" / "ruler.yaml")
DEFAULT_DEPTH = RULER_CFG.default_depth
DEFAULT_SEQ = RULER_CFG.default_seq
RULER_COMPUTE = RULER_CFG.compute

def main():
    parser = argparse.ArgumentParser(
        description="Run a single RULER-style long-context experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", choices=list(TASK_INFO.keys()), help="RULER task")
    parser.add_argument("--exp", type=int, choices=sorted(EXPERIMENT_CONFIGS.keys()),
                        help="Experiment number (0=dense baseline)")
    parser.add_argument("--size", choices=list(RULER_COMPUTE.keys()), default="ruler-smoke",
                        help="Compute preset (default: ruler-smoke)")
    parser.add_argument("--seq", type=int, help="Context window")
    parser.add_argument("--depth", type=float, help="Needle depth fraction 0..1 (default 0.5)")
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--accum", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--list", action="store_true", help="List presets and tasks")
    args = parser.parse_args()

    if args.list:
        print("\nRULER compute presets:")
        for name, c in RULER_COMPUTE.items():
            print(f"  {name}: {c['desc']} "
                  f"({c['train_samples']} train, {c['epochs']} epochs, batch {c['batch_size']}x{c['grad_accum']})")
        print("\nOfficial tasks (NVIDIA RULER synthetic.yaml):")
        for t in OFFICIAL_TASKS:
            info = TASK_INFO[t]
            print(f"  {t}: num_labels={info['num_labels']}, uses_depth={info['uses_depth']}, "
                  f"default_seq={DEFAULT_SEQ.get(t)}")
        print("\nAliases: niah -> niah_single_1; mq_niah = dedicated 2-key selective retrieval")
        print("\nNote: exp_15 uses a from-scratch BigBird backbone (BART for 0–14).")
        print("\nExperiments:")
        for num, (name, _, params) in EXPERIMENT_CONFIGS.items():
            print(f"  {num}: {name} ({params})")
        return

    if args.task is None or args.exp is None:
        parser.error("--task and --exp are required (unless --list)")

    compute = dict(RULER_COMPUTE[args.size])
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

    seq_len = args.seq or DEFAULT_SEQ[args.task]
    depth = args.depth if args.depth is not None else DEFAULT_DEPTH

    print(f"\n{'='*70}")
    print(f"RULER task: {args.task} | exp {args.exp} | seq {seq_len} | depth {depth} | {args.size}")
    print(f"{'='*70}\n")

    data = build_ruler_dataset(
        task=args.task,
        seq_len=seq_len,
        needle_depth=depth,
        train_samples=compute["train_samples"],
        eval_samples=compute["eval_samples"],
        seed=args.seed,
    )
    ds = {"train": data["train"], "validation": data["validation"]}

    model, exp_name, meta = build_lra_model(
        task=args.task,
        exp_num=args.exp,
        vocab_size=data["vocab_size"],
        seq_len=seq_len,
        num_labels=data["num_labels"],
        pair=data["pair"],
    )

    if args.grad_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    train_schema = OmegaConf.structured(TrainConfig)
    train_cfg = OmegaConf.load(CONFIG_DIR / "trainer" / "encoder.yaml")
    train_cfg.epochs = compute["epochs"]
    train_cfg.per_device_train_bs = compute["batch_size"]
    train_cfg.per_device_eval_bs = compute["batch_size"]
    train_cfg.grad_accum_steps = compute["grad_accum"]
    train_cfg.lr = args.lr
    train_cfg.use_cpu = args.cpu 
    train_cfg.torch_compile = args.compile

    train_cfg = OmegaConf.merge(train_schema, train_cfg)

    run_lra(
        task=args.task,
        exp_name=exp_name,
        model=model,
        ds=ds,
        cfg=train_cfg,
        num_labels=data["num_labels"],
        seq_len=seq_len,
        vocab_size=data["vocab_size"],
        pair=data["pair"],
        extra_meta={
            **meta,
            "compute_preset": args.size,
            "needle_depth": depth,
            "canonical_task": data.get("canonical_task", args.task),
        },
        save_weights=args.save_weights,
        track="ruler",
    )


if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
