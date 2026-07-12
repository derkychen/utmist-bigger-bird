#!/usr/bin/env python3
"""RULER-style context × depth × experiment sweep.

Usage:
  python -m eval.ruler.sweep --tasks niah --exps 0,1,7 --seqs 2048,4096 --depths 0.1,0.5,0.9 --size ruler-smoke
  python -m eval.ruler.sweep --size ruler-report --skip-existing
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.ruler.presets import DEFAULT_DEPTHS, DEFAULT_SEQS, RULER_COMPUTE, TRACK
from eval.sweep_utils import (
    is_oom,
    load_matching_eval,
    parse_csv_floats,
    parse_csv_ints,
    rebuild_dashboard,
    run_module,
    save_sweep,
)
from run_experiment import EXPERIMENT_CONFIGS

ALL_EXPS = sorted(EXPERIMENT_CONFIGS.keys())
DEFAULT_EXPS = ",".join(str(x) for x in ALL_EXPS)


def _match_ruler(data: dict, seq: int, depth: float, train_samples: int) -> bool:
    meta = data.get("experiment_metadata", {})
    mc = meta.get("model_config", {})
    return (
        meta.get("seq_length") == seq
        and abs(mc.get("needle_depth", -1) - depth) < 1e-6
        and meta.get("dataset_info", {}).get("train_size") == train_samples
    )


def _result_from_eval(task, exp, exp_name, seq, depth, data, status):
    perf = data.get("performance_metrics", {})
    ev = perf.get("eval", {})
    return {
        "task": task,
        "exp": exp,
        "exp_name": exp_name,
        "seq": seq,
        "depth": depth,
        "accuracy": ev.get("eval_accuracy"),
        "f1": ev.get("eval_f1"),
        "train_time_s": perf.get("training_time_seconds"),
        "peak_mem_mb": perf.get("peak_memory_mb"),
        "inference_ms": perf.get("inference_latency_ms"),
        "softmax_comparisons": perf.get("softmax_comparisons"),
        "oom": False,
        "status": status,
    }


def run_single(task, exp, seq, depth, size, cpu, skip_existing=False, batch=None, grad_checkpoint=False):
    exp_name = EXPERIMENT_CONFIGS[exp][0]
    train_samples = RULER_COMPUTE[size]["train_samples"]
    matcher = lambda d: _match_ruler(d, seq, depth, train_samples)

    if skip_existing and load_matching_eval(TRACK, task, exp_name, matcher):
        print(f"\n{'='*70}\nRULER sweep: {task} | {exp_name} | seq={seq} depth={depth} — SKIP\n{'='*70}")
        data = load_matching_eval(TRACK, task, exp_name, matcher)
        if data is None:
            return {"task": task, "exp": exp, "exp_name": exp_name, "seq": seq,
                    "depth": depth, "status": "skipped"}
        return _result_from_eval(task, exp, exp_name, seq, depth, data, "skipped")

    cmd = ["--task", task, "--exp", str(exp), "--seq", str(seq), "--depth", str(depth), "--size", size]
    if cpu:
        cmd.append("--cpu")
    if batch is not None:
        cmd += ["--batch", str(batch)]
    if grad_checkpoint:
        cmd.append("--grad-checkpoint")

    print(f"\n{'='*70}\nRULER sweep: {task} | {exp_name} | seq={seq} depth={depth}\n{'='*70}")
    res = run_module("eval.ruler.run", cmd)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    if res.returncode != 0:
        blob = (res.stderr or "") + (res.stdout or "")
        return {
            "task": task, "exp": exp, "exp_name": exp_name, "seq": seq, "depth": depth,
            "oom": is_oom(blob), "status": "oom" if is_oom(blob) else "fail",
        }

    data = load_matching_eval(TRACK, task, exp_name, matcher)
    if data is None:
        return {"task": task, "exp": exp, "exp_name": exp_name, "seq": seq, "depth": depth,
                "status": "no_output"}

    acc = data.get("performance_metrics", {}).get("eval", {}).get("eval_accuracy")
    collapsed = acc is not None and acc < 0.15
    return _result_from_eval(task, exp, exp_name, seq, depth, data, "collapse" if collapsed else "ok")


def main():
    parser = argparse.ArgumentParser(description="RULER depth × context × experiment sweep")
    parser.add_argument("--tasks", default="niah,mq_niah", help="Comma-separated tasks")
    parser.add_argument("--exps", default=DEFAULT_EXPS, help="Experiment numbers (default: all 13)")
    parser.add_argument("--seqs", default="", help="Override seq lengths for all tasks")
    parser.add_argument("--depths", default="", help="Override needle depths (e.g. 0.1,0.5,0.9)")
    parser.add_argument("--size", default="ruler-report", help="Compute preset")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    exps = parse_csv_ints(args.exps)
    seq_override = parse_csv_ints(args.seqs) if args.seqs else None
    depths = parse_csv_floats(args.depths) if args.depths else DEFAULT_DEPTHS

    all_results = []
    for task in tasks:
        seqs = seq_override or DEFAULT_SEQS.get(task, [4096])
        for seq in seqs:
            for depth in depths:
                for exp in exps:
                    all_results.append(
                        run_single(task, exp, seq, depth, args.size, args.cpu, args.skip_existing, args.batch, args.grad_checkpoint)
                    )

    print("\n" + "=" * 120)
    print("RULER DEPTH × CONTEXT SWEEP SUMMARY")
    print("=" * 120)
    hdr = (f"{'Task':<10} {'Exp':<22} {'Seq':>6} {'Depth':>6} {'Acc':>7} {'F1':>7} "
           f"{'Time(s)':>9} {'Mem(MB)':>9} {'Status':>9}")
    print(hdr)
    print("-" * 120)
    for r in all_results:
        acc = f"{r['accuracy']:.3f}" if r.get("accuracy") is not None else "N/A"
        f1 = f"{r['f1']:.3f}" if r.get("f1") is not None else "N/A"
        t = f"{r['train_time_s']:.1f}" if r.get("train_time_s") else "N/A"
        mem = f"{r['peak_mem_mb']:.0f}" if r.get("peak_mem_mb") else "N/A"
        print(f"{r['task']:<10} {r['exp_name']:<22} {r['seq']:>6} {r['depth']:>6.2f} "
              f"{acc:>7} {f1:>7} {t:>9} {mem:>9} {r.get('status','?'):>9}")
    print("=" * 120)

    payload = {
        "config": {
            "tasks": tasks, "exps": exps, "seqs": seq_override, "depths": depths,
            "size": args.size, "timestamp": datetime.now().isoformat(),
        },
        "results": all_results,
    }
    ts_path, latest_path = save_sweep(payload, "ruler_sweep")
    print(f"\nSaved sweep results to: {ts_path}\nAlso updated: {latest_path}")
    rebuild_dashboard()


if __name__ == "__main__":
    main()
