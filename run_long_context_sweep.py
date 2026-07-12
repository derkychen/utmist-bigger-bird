#!/usr/bin/env python3
"""
Long-context sweep: baseline vs sparse methods at increasing sequence lengths.
Shows where sparse attention wins (OOM avoidance + wall-clock speedup).

Usage:
  python run_long_context_sweep.py --seqs 1024,2048,4096 --exps 0,3,5,8
  python run_long_context_sweep.py --seqs 2048,4096 --exps 0,5 --grad-checkpoint
"""

import sys
import os
import subprocess
import json
import argparse
from datetime import datetime

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

EXP_BENCH_DIRS = {
    0: "exp_0_baseline",
    1: "exp_1_deepseek_topk",
    2: "exp_2_lightning_hybrid",
    3: "exp_3_dynamic_globals",
    4: "exp_4_pbs_attn",
    5: "exp_5_bigger_bird",
    6: "exp_6_deepseek_pbs",
    7: "exp_7_layer_adaptive",
    8: "exp_8_token_drop",
    9: "exp_9_attn_specul",
    10: "exp_10_gqa_sparse",
}


def exp_bench_dir(exp: int) -> str:
    return EXP_BENCH_DIRS.get(exp, f"exp_{exp}")


def needs_cpu_for_seq(seq):
    """Apple MPS cannot allocate >12GB buffers; seq>=2048 training exceeds that."""
    mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    return mps and not torch.cuda.is_available() and seq >= 2048


def run_single(exp, seq, train_samples=500, eval_samples=100, grad_checkpoint=False, force_cpu=False, save_weights=False):
    """Run one experiment and return results JSON path or None on failure."""
    cmd = [
        sys.executable, "run_experiment.py",
        "--exp", str(exp),
        "--size", "long",
        "--seq", str(seq),
        "--train-samples", str(train_samples),
        "--eval-samples", str(eval_samples),
        "--fixed-length",
        "--epochs", "2",
        "--batch", "1",
        "--accum", "1",
    ]
    if grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if save_weights:
        cmd.append("--save-weights")
    if force_cpu or needs_cpu_for_seq(seq):
        cmd.append("--cpu")

    exp_name = exp_bench_dir(exp)
    device_note = " [CPU]" if (force_cpu or needs_cpu_for_seq(seq)) else ""
    print(f"\n{'='*70}")
    print(f"Running: {exp_name} | seq={seq}{device_note}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"[FAIL] {exp_name} @ seq={seq}")
        return None
    
    bench_dir = os.path.join(os.path.dirname(__file__), "benchmarks", exp_name)
    if not os.path.isdir(bench_dir):
        return None
    
    json_files = [f for f in os.listdir(bench_dir) if f.startswith("eval_") and f.endswith(".json")]
    if not json_files:
        return None
    json_files.sort()
    latest = os.path.join(bench_dir, json_files[-1])
    
    with open(latest) as f:
        data = json.load(f)
    
    perf = data.get("performance_metrics", {})
    meta = data.get("experiment_metadata", {})
    
    return {
        "exp": exp,
        "exp_name": exp_name,
        "seq": seq,
        "accuracy": perf.get("eval", {}).get("eval_accuracy"),
        "f1": perf.get("eval", {}).get("eval_f1"),
        "train_time_s": perf.get("training_time_seconds"),
        "peak_mem_mb": perf.get("peak_memory_mb"),
        "inference_ms": perf.get("inference_latency_ms"),
        "softmax_comparisons": perf.get("softmax_comparisons"),
        "oom": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Long-context sweep across sequence lengths")
    parser.add_argument("--seqs", type=str, default="1024,2048,4096",
                       help="Comma-separated sequence lengths to test")
    parser.add_argument("--exps", type=str, default="0,3,5,8",
                       help="Comma-separated experiment numbers")
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--grad-checkpoint", action="store_true",
                       help="Enable gradient checkpointing for baseline")
    parser.add_argument("--save-weights", action="store_true",
                       help="Save model weights to benchmarks/<exp>/weights_<timestamp>/ for later eval")
    parser.add_argument("--skip-baseline-oom", action="store_true",
                       help="If baseline OOMs, skip it and continue sparse runs")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU for all runs (auto-enabled for seq>=2048 on Apple MPS)")
    args = parser.parse_args()

    seqs = [int(x) for x in args.seqs.split(",")]
    exps = [int(x) for x in args.exps.split(",")]

    all_results = []
    baseline_oom_seqs = set()

    for seq in seqs:
        for exp in exps:
            # Skip baseline if it already OOMed at a shorter seq length
            if exp == 0 and seq in baseline_oom_seqs:
                print(f"\n[SKIP] Baseline known to OOM at seq={seq}, skipping.")
                all_results.append({"exp": 0, "exp_name": exp_bench_dir(0), "seq": seq, "oom": True})
                continue

            res = run_single(
                exp, seq,
                train_samples=args.train_samples,
                eval_samples=args.eval_samples,
                grad_checkpoint=args.grad_checkpoint,
                force_cpu=args.cpu,
                save_weights=args.save_weights,
            )
            
            if res is None:
                all_results.append({"exp": exp, "exp_name": exp_bench_dir(exp), "seq": seq, "oom": True})
                if exp == 0:
                    baseline_oom_seqs.add(seq)
            else:
                all_results.append(res)

    # Print summary table
    print("\n" + "="*100)
    print("LONG-CONTEXT SWEEP SUMMARY")
    print("="*100)
    print(f"{'Exp':<20} {'Seq':>6} {'Acc':>8} {'F1':>8} {'Time(s)':>10} {'PeakMem(MB)':>12} {'Inf(ms)':>10} {'SoftmaxCmp':>14} {'Status':>8}")
    print("-"*100)
    
    for r in all_results:
        status = "OOM" if r.get("oom") else "OK"
        acc = f"{r['accuracy']:.3f}" if r.get("accuracy") is not None else "N/A"
        f1 = f"{r['f1']:.3f}" if r.get("f1") is not None else "N/A"
        t = f"{r['train_time_s']:.1f}" if r.get("train_time_s") else "N/A"
        mem = f"{r['peak_mem_mb']:.0f}" if r.get("peak_mem_mb") else "N/A"
        inf = f"{r['inference_ms']:.1f}" if r.get("inference_ms") else "N/A"
        cmp_ = f"{r['softmax_comparisons']:,}" if r.get("softmax_comparisons") else "N/A"
        print(f"{r['exp_name']:<20} {r['seq']:>6} {acc:>8} {f1:>8} {t:>10} {mem:>12} {inf:>10} {cmp_:>14} {status:>8}")
    
    print("="*100)

    bench_root = os.path.join(os.path.dirname(__file__), "benchmarks")
    payload = {
        "config": {
            "seqs": seqs,
            "exps": exps,
            "train_samples": args.train_samples,
            "eval_samples": args.eval_samples,
            "grad_checkpoint": args.grad_checkpoint,
            "fixed_length": True,
            "timestamp": datetime.now().isoformat(),
        },
        "results": all_results,
    }
    ts_path = os.path.join(bench_root, f"long_context_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    latest_path = os.path.join(bench_root, "long_context_sweep_results.json")
    for out_path in (ts_path, latest_path):
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
    print(f"\nSaved sweep results to: {ts_path}")
    print(f"Also updated: {latest_path}")


if __name__ == "__main__":
    main()
