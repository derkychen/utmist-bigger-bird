#!/usr/bin/env python3
"""
Long-context efficiency evaluation.
Tests each experiment at increasing sequence lengths with a FIXED compute budget
(500 samples, 2 epochs) to answer:
  "With the same resources, how long a context can each method handle?"

Usage:
  python benchmarks/efficiency_eval.py --exp 1,2,3,4 --seq 256,512,1024,2048
  python benchmarks/efficiency_eval.py --exp 0 --seq 256,512,1024,2048,4096
"""

import sys
import os
import argparse
import json
import gc
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shared.dataset import build_imdb_dataset, DataConfig
from shared.runner import run_experiment, TrainConfig
from run_experiment import extend_position_embeddings

from exp_1_deepseek_topk.model import PatchedModel as DeepSeekModel
from exp_2_lightning_hybrid.model import PatchedModel as LightningModel
from exp_3_dynamic_globals.model import PatchedModel as DynamicGlobalsModel
from exp_4_pbs_attn.model import PatchedModel as PBSModel

EXPERIMENT_CONFIGS = {
    0: ("exp_0_baseline", None, {"attention": "full_dense"}),
    1: ("exp_1_deepseek_topk", DeepSeekModel, {"top_k": 64, "low_rank_dim": 16, "use_triton": True}),
    2: ("exp_2_lightning_hybrid", LightningModel, {"block_size": 128, "use_triton": True}),
    3: ("exp_3_dynamic_globals", DynamicGlobalsModel, {"window_size": 64, "num_globals": 16, "use_triton": True}),
    4: ("exp_4_pbs_attn", PBSModel, {"block_size": 64, "num_blocks": 2, "use_triton": True}),
}


def run_single(
    exp_num,
    seq_len,
    train_samples=500,
    eval_samples=100,
    epochs=2,
    batch_size=1,
    grad_accum=1,
    grad_checkpoint=False,
):
    """Run one experiment at one sequence length. Returns result dict or None on OOM."""
    exp_name, ModelClass, model_params = EXPERIMENT_CONFIGS[exp_num]
    run_label = f"{exp_name}_seq{seq_len}"

    print(f"\n{'='*70}")
    print(f"[efficiency_eval] {run_label} | samples={train_samples}, epochs={epochs}, seq={seq_len}")
    print(f"{'='*70}")

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    try:
        model_name = "facebook/bart-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        base_model.config.classifier_dropout = 0.1

        if seq_len > base_model.config.max_position_embeddings:
            extend_position_embeddings(base_model, seq_len)
        if grad_checkpoint:
            base_model.gradient_checkpointing_enable()
        if ModelClass is None:
            model = base_model
        else:
            model = ModelClass(base_model, **model_params)

        data_cfg = DataConfig(
            train_samples=train_samples,
            eval_samples=eval_samples,
            max_length=seq_len
        )
        ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=seq_len)

        train_cfg = TrainConfig(
            epochs=epochs,
            per_device_train_bs=batch_size,
            per_device_eval_bs=batch_size,
            grad_accum_steps=grad_accum,
            lr=3e-5
        )

        eval_res = run_experiment(
            run_label,
            model,
            tokenizer,
            ds,
            train_cfg,
            extra_meta={
                **model_params,
                "efficiency_eval": True,
                "seq_length": seq_len,
                "train_samples": train_samples,
            },
        )

        # Extract key metrics from the freshly saved JSON
        out_dir = os.path.join(os.path.dirname(__file__), run_label)
        json_files = [f for f in os.listdir(out_dir) if f.startswith("eval_") and f.endswith(".json")]
        latest_json = max(json_files, key=lambda f: os.path.getmtime(os.path.join(out_dir, f)))
        with open(os.path.join(out_dir, latest_json)) as f:
            data = json.load(f)

        perf = data["performance_metrics"]
        return {
            "exp_name": exp_name,
            "exp_num": exp_num,
            "seq_length": seq_len,
            "train_samples": train_samples,
            "epochs": epochs,
            "f1": perf["eval"].get("eval_f1", 0),
            "accuracy": perf["eval"].get("eval_accuracy", 0),
            "train_time_s": perf["training_time_seconds"],
            "peak_memory_mb": perf.get("peak_memory_mb", 0),
            "inference_latency_ms": perf.get("inference_latency_ms"),
            "softmax_comparisons": perf.get("softmax_comparisons"),
            "train_samples_per_sec": perf["train"].get("train_samples_per_second", 0),
            "eval_samples_per_sec": perf["eval"].get("eval_samples_per_second", 0),
            "oom": False,
        }

    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM] {run_label}: {e}")
        return {
            "exp_name": exp_name,
            "exp_num": exp_num,
            "seq_length": seq_len,
            "train_samples": train_samples,
            "epochs": epochs,
            "f1": None,
            "accuracy": None,
            "train_time_s": None,
            "peak_memory_mb": None,
            "inference_latency_ms": None,
            "softmax_comparisons": None,
            "train_samples_per_sec": None,
            "eval_samples_per_sec": None,
            "oom": True,
        }
    except Exception as e:
        print(f"[ERROR] {run_label}: {e}")
        traceback.print_exc()
        return {
            "exp_name": exp_name,
            "exp_num": exp_num,
            "seq_length": seq_len,
            "train_samples": train_samples,
            "epochs": epochs,
            "f1": None,
            "accuracy": None,
            "train_time_s": None,
            "peak_memory_mb": None,
            "inference_latency_ms": None,
            "softmax_comparisons": None,
            "train_samples_per_sec": None,
            "eval_samples_per_sec": None,
            "oom": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Long-context efficiency evaluation")
    parser.add_argument("--exp", type=str, default="0,1,2,3,4",
                        help="Comma-separated experiment numbers")
    parser.add_argument("--seq", type=str, default="256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--samples", type=int, default=500,
                        help="Fixed number of training samples")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Fixed number of epochs")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size (auto-lower on OOM not yet supported)")
    parser.add_argument("--accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing (recommended for seq>=2048)")
    args = parser.parse_args()

    exp_nums = [int(x.strip()) for x in args.exp.split(",")]
    seq_lengths = [int(x.strip()) for x in args.seq.split(",")]

    results = []
    for exp_num in exp_nums:
        for seq_len in seq_lengths:
            res = run_single(
                exp_num, seq_len,
                train_samples=args.samples,
                eval_samples=args.samples // 5,
                epochs=args.epochs,
                batch_size=args.batch,
                grad_accum=args.accum,
                grad_checkpoint=args.grad_checkpoint,
            )
            results.append(res)

    # Save aggregate results
    out_path = os.path.join(os.path.dirname(__file__), "efficiency_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "train_samples": args.samples,
                "epochs": args.epochs,
            },
            "results": results,
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"EFFICIENCY EVAL COMPLETE — {len(results)} runs")
    print(f"Saved aggregate results to: {out_path}")
    print(f"{'='*70}")

    # Print summary table
    print("\nSUMMARY TABLE")
    print("-" * 100)
    header = f"{'Exp':<20} {'Seq':>6} {'F1':>7} {'Time(s)':>10} {'Peak(MB)':>10} {'Lat(ms)':>10} {'OOM':>5}"
    print(header)
    print("-" * 100)
    for r in results:
        oom_flag = "YES" if r["oom"] else "NO"
        f1_str = f"{r['f1']:.3f}" if r["f1"] is not None else "N/A"
        time_str = f"{r['train_time_s']:.1f}" if r["train_time_s"] is not None else "N/A"
        mem_str = f"{r['peak_memory_mb']:.0f}" if r["peak_memory_mb"] is not None else "N/A"
        lat_str = f"{r['inference_latency_ms']:.1f}" if r["inference_latency_ms"] is not None else "N/A"
        print(f"{r['exp_name']:<20} {r['seq_length']:>6} {f1_str:>7} {time_str:>10} {mem_str:>10} {lat_str:>10} {oom_flag:>5}")
    print("-" * 100)


if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
