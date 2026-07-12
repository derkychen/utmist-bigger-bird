#!/usr/bin/env python3
"""Run a single Long Range Arena (LRA) experiment.

Usage:
  python -m eval.lra.run --task listops --exp 0 --seq 2048
  python -m eval.lra.run --task text --exp 1 --seq 4096 --size lra-full
  python -m eval.lra.run --task retrieval --exp 7 --seq 4096 --data-dir lra_data/retrieval
  python -m eval.lra.run --list
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from eval.lra.presets import DEFAULT_SEQ, LRA_COMPUTE
from run_experiment import EXPERIMENT_CONFIGS
from shared.lra_dataset import TASK_INFO, build_lra_dataset
from shared.lra_model import build_lra_model
from shared.runner import TrainConfig, run_lra


def main():
    parser = argparse.ArgumentParser(
        description="Run a single LRA long-context experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", choices=list(TASK_INFO.keys()), help="LRA task")
    parser.add_argument("--exp", type=int, choices=sorted(EXPERIMENT_CONFIGS.keys()),
                        help="Experiment number (0=dense baseline)")
    parser.add_argument("--size", choices=list(LRA_COMPUTE.keys()), default="lra-smoke",
                        help="Compute preset (default: lra-smoke)")
    parser.add_argument("--seq", type=int, help="Context window (overrides task default)")
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--accum", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=None, help="Data dir for retrieval (AAN)")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--list", action="store_true", help="List presets and experiments")
    args = parser.parse_args()

    if args.list:
        print("\nLRA compute presets:")
        for name, c in LRA_COMPUTE.items():
            print(f"  {name}: {c['desc']} "
                  f"({c['train_samples']} train, {c['epochs']} epochs, batch {c['batch_size']}x{c['grad_accum']})")
        print("\nTasks (default seq):")
        for t, info in TASK_INFO.items():
            print(f"  {t}: num_labels={info['num_labels']}, pair={info['pair']}, default_seq={DEFAULT_SEQ[t]}")
        print("\nExperiments:")
        for num, (name, _, params) in EXPERIMENT_CONFIGS.items():
            print(f"  {num}: {name} ({params})")
        return

    if args.task is None or args.exp is None:
        parser.error("--task and --exp are required (unless --list)")

    compute = dict(LRA_COMPUTE[args.size])
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

    print(f"\n{'='*70}")
    print(f"LRA task: {args.task} | exp {args.exp} | seq {seq_len} | preset {args.size}")
    print(f"{'='*70}\n")

    data = build_lra_dataset(
        task=args.task,
        seq_len=seq_len,
        train_samples=compute["train_samples"],
        eval_samples=compute["eval_samples"],
        seed=args.seed,
        data_dir=args.data_dir,
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

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    train_cfg = TrainConfig(
        epochs=compute["epochs"],
        per_device_train_bs=compute["batch_size"],
        per_device_eval_bs=compute["batch_size"],
        grad_accum_steps=compute["grad_accum"],
        lr=args.lr,
        use_cpu=args.cpu,
        torch_compile=args.compile,
    )

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
        extra_meta={**meta, "compute_preset": args.size},
        save_weights=args.save_weights,
    )


if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
