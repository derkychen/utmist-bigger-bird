#!/usr/bin/env python3
"""Run a single NoLiMa long-context experiment (encoder-only classification).

Uses the official Adobe/HF needlesets + haystacks, reduced to 10-way character
classification (same host path as LRA / RULER).

Usage:
  bash scripts/get_nolima_data.sh
  python -m eval.nolima.run --task onehop --exp 0 --seq 2048 --size nolima-smoke
  python -m eval.nolima.run --task twohop --exp 1 --seq 4096 --depth 0.5
  python -m eval.nolima.run --task hard --exp 0 --seq 4096 --size nolima-report
  python -m eval.nolima.run --list
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from eval.nolima.presets import DEFAULT_DEPTH, DEFAULT_SEQ, NOLIMA_COMPUTE
from run_experiment import EXPERIMENT_CONFIGS
from eval.lra.lra_model import build_lra_model
from eval.nolima.nolima_dataset import TASK_INFO, build_nolima_dataset
from patches.original_patches.runner import TrainConfig, run_lra


def main():
    parser = argparse.ArgumentParser(
        description="Run a single NoLiMa long-context experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", choices=list(TASK_INFO.keys()), help="NoLiMa task")
    parser.add_argument("--exp", type=int, choices=sorted(EXPERIMENT_CONFIGS.keys()),
                        help="Experiment number (0=dense baseline)")
    parser.add_argument("--size", choices=list(NOLIMA_COMPUTE.keys()), default="nolima-smoke",
                        help="Compute preset (default: nolima-smoke)")
    parser.add_argument("--seq", type=int, help="Context window")
    parser.add_argument("--depth", type=float, help="Needle depth fraction 0..1 (default 0.5)")
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--accum", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="NoLiMa data root (default: ./nolima_data or $NOLIMA_DATA_DIR)")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--list", action="store_true", help="List presets and tasks")
    args = parser.parse_args()

    if args.list:
        print("\nNoLiMa compute presets:")
        for name, c in NOLIMA_COMPUTE.items():
            print(f"  {name}: {c['desc']} "
                  f"({c['train_samples']} train, {c['epochs']} epochs, "
                  f"batch {c['batch_size']}x{c['grad_accum']})")
        print("\nTasks (Adobe needle sets → 10-way character classification):")
        for t, info in TASK_INFO.items():
            hops = info.get("hops")
            hop_s = "all hops" if hops is None else ",".join(hops)
            print(f"  {t}: file={info['needle_file']}, hops={hop_s}, "
                  f"default_seq={DEFAULT_SEQ.get(t)}")
        print("\nExperiments:")
        for num, (name, _, params) in EXPERIMENT_CONFIGS.items():
            print(f"  {num}: {name} ({params})")
        print("\nData setup:")
        print("  bash scripts/get_nolima_data.sh")
        return

    if args.task is None or args.exp is None:
        parser.error("--task and --exp are required (unless --list)")

    compute = dict(NOLIMA_COMPUTE[args.size])
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
    print(f"NoLiMa task: {args.task} | exp {args.exp} | seq {seq_len} | "
          f"depth {depth} | {args.size}")
    print(f"{'='*70}\n")

    data = build_nolima_dataset(
        task=args.task,
        seq_len=seq_len,
        needle_depth=depth,
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

    if args.grad_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
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
        extra_meta={
            **meta,
            "compute_preset": args.size,
            "needle_depth": depth,
            "canonical_task": data.get("canonical_task", args.task),
            "num_pairs": data.get("num_pairs"),
            "nolima_data_dir": data.get("data_dir"),
            "benchmark": "NoLiMa",
        },
        save_weights=args.save_weights,
        track="nolima",
    )


if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
