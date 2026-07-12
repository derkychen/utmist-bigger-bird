#!/usr/bin/env python3
"""LRA context-window sweep: tasks × sequence lengths × experiments.

Usage:
  python -m eval.lra.sweep --tasks listops,text --exps 0,1,7 --size lra-smoke
  python -m eval.lra.sweep --tasks listops --seqs 512,1024,2048 --size lra-report --skip-existing
  python -m eval.lra.sweep --tasks retrieval --exps 0,1,7 --data-dir lra_data/retrieval
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.lra.presets import DEFAULT_SEQS, LRA_COMPUTE, TRACK
from eval.sweep_utils import (
    is_oom,
    load_matching_eval,
    parse_csv_ints,
    rebuild_dashboard,
    run_module,
    save_sweep,
)
from run_experiment import EXPERIMENT_CONFIGS

ALL_EXPS = sorted(EXPERIMENT_CONFIGS.keys())
DEFAULT_EXPS = ",".join(str(x) for x in ALL_EXPS)


def _match_lra(data: dict, seq: int, train_samples: int) -> bool:
    meta = data.get("experiment_metadata", {})
    return (
        meta.get("seq_length") == seq
        and meta.get("dataset_info", {}).get("train_size") == train_samples
    )


def _result_from_eval(task, exp, exp_name, seq, data, status):
    perf = data.get("performance_metrics", {})
    ev = perf.get("eval", {})
    f1 = ev.get("eval_f1")
    return {
        "task": task,
        "exp": exp,
        "exp_name": exp_name,
        "seq": seq,
        "accuracy": ev.get("eval_accuracy"),
        "f1": f1,
        "train_time_s": perf.get("training_time_seconds"),
        "peak_mem_mb": perf.get("peak_memory_mb"),
        "inference_ms": perf.get("inference_latency_ms"),
        "softmax_comparisons": perf.get("softmax_comparisons"),
        "oom": False,
        "status": status,
    }


def run_single(task, exp, seq, size, data_dir, cpu, skip_existing=False, batch=None, grad_checkpoint=False):
    exp_name = EXPERIMENT_CONFIGS[exp][0]
    train_samples = LRA_COMPUTE[size]["train_samples"]

    if skip_existing and load_matching_eval(
        TRACK, task, exp_name, lambda d: _match_lra(d, seq, train_samples)
    ):
        print(f"\n{'='*70}\nLRA sweep: {task} | {exp_name} | seq={seq} — SKIP (existing result)\n{'='*70}")
        data = load_matching_eval(
            TRACK, task, exp_name, lambda d: _match_lra(d, seq, train_samples)
        )
        if data is None:
            return {"task": task, "exp": exp, "exp_name": exp_name, "seq": seq, "status": "skipped"}
        return _result_from_eval(task, exp, exp_name, seq, data, "skipped")

    cmd = ["--task", task, "--exp", str(exp), "--seq", str(seq), "--size", size]
    if data_dir:
        cmd += ["--data-dir", data_dir]
    if cpu:
        cmd.append("--cpu")
    if batch is not None:
        cmd += ["--batch", str(batch)]
    if grad_checkpoint:
        cmd.append("--grad-checkpoint")

    print(f"\n{'='*70}\nLRA sweep: {task} | {exp_name} | seq={seq}\n{'='*70}")
    res = run_module("eval.lra.run", cmd)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    if res.returncode != 0:
        blob = (res.stderr or "") + (res.stdout or "")
        return {
            "task": task, "exp": exp, "exp_name": exp_name, "seq": seq,
            "oom": is_oom(blob), "status": "oom" if is_oom(blob) else "fail",
        }

    data = load_matching_eval(
        TRACK, task, exp_name, lambda d: _match_lra(d, seq, train_samples)
    )
    if data is None:
        return {"task": task, "exp": exp, "exp_name": exp_name, "seq": seq, "status": "no_output"}

    f1 = data.get("performance_metrics", {}).get("eval", {}).get("eval_f1")
    collapsed = f1 is not None and f1 < 0.05
    return _result_from_eval(task, exp, exp_name, seq, data, "collapse" if collapsed else "ok")


def main():
    parser = argparse.ArgumentParser(description="LRA context-window sweep")
    parser.add_argument("--tasks", default="listops,text", help="Comma-separated LRA tasks")
    parser.add_argument("--exps", default=DEFAULT_EXPS,
                        help=f"Comma-separated experiment numbers (default: all {len(ALL_EXPS)})")
    parser.add_argument("--seqs", default="", help="Override seq lengths (applies to all tasks)")
    parser.add_argument("--size", default="lra-report", help="Compute preset (default: lra-report)")
    parser.add_argument("--data-dir", default=None, help="Data dir for retrieval (AAN)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse existing eval artifacts for the same task/exp/seq/budget")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    exps = parse_csv_ints(args.exps)
    seq_override = parse_csv_ints(args.seqs) if args.seqs else None

    all_results = []
    for task in tasks:
        seqs = seq_override or DEFAULT_SEQS.get(task, [1024, 2048])
        for seq in seqs:
            for exp in exps:
                all_results.append(
                    run_single(task, exp, seq, args.size, args.data_dir, args.cpu, args.skip_existing, args.batch, args.grad_checkpoint)
                )

    print("\n" + "=" * 110)
    print("LRA CONTEXT-WINDOW SWEEP SUMMARY")
    print("=" * 110)
    hdr = f"{'Task':<10} {'Exp':<22} {'Seq':>6} {'Acc':>7} {'F1':>7} {'Time(s)':>9} {'Mem(MB)':>9} {'Inf(ms)':>8} {'Status':>9}"
    print(hdr)
    print("-" * 110)
    for r in all_results:
        acc = f"{r['accuracy']:.3f}" if r.get("accuracy") is not None else "N/A"
        f1 = f"{r['f1']:.3f}" if r.get("f1") is not None else "N/A"
        t = f"{r['train_time_s']:.1f}" if r.get("train_time_s") else "N/A"
        mem = f"{r['peak_mem_mb']:.0f}" if r.get("peak_mem_mb") else "N/A"
        inf = f"{r['inference_ms']:.1f}" if r.get("inference_ms") else "N/A"
        print(f"{r['task']:<10} {r['exp_name']:<22} {r['seq']:>6} {acc:>7} {f1:>7} {t:>9} {mem:>9} {inf:>8} {r.get('status','?'):>9}")
    print("=" * 110)

    payload = {
        "config": {
            "tasks": tasks,
            "exps": exps,
            "seqs": seq_override,
            "size": args.size,
            "timestamp": datetime.now().isoformat(),
        },
        "results": all_results,
    }
    ts_path, latest_path = save_sweep(payload, "lra_sweep")
    print(f"\nSaved sweep results to: {ts_path}\nAlso updated: {latest_path}")
    rebuild_dashboard()


if __name__ == "__main__":
    main()
