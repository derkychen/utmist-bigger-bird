#!/usr/bin/env python3
"""Visualize LRA long-context results: accuracy/F1 retention vs context window.

Reads the per-experiment artifacts written by run_lra (benchmarks/lra_<task>_<exp>/eval_*.json),
emits one accuracy-vs-seq plot per task plus a retention table, and exports
benchmarks/lra_comparison.csv for dashboard ingestion (alongside the IMDb comparison.csv).
"""

import csv
import glob
import json
import os
import sys

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmarks")


# Each (task, exp) dir accumulates one eval_*.json per seq length; keep the newest run
# per (task, exp, seq). MIN_SEQ drops the tiny seq-256 smoke points and MIN_TRAIN drops
# the tiny OOM-probe runs (different, non-comparable budgets).
MIN_SEQ = 512
MIN_TRAIN = 100


def _exp_sort_key(name):
    """Sort exp_0_baseline, exp_1_..., exp_10_... numerically."""
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return name


def _sorted_exps(names):
    return sorted(set(names), key=_exp_sort_key)


def load_lra_results(min_seq=MIN_SEQ, min_train=MIN_TRAIN):
    """Return deduped rows: newest run per (task, exp_name, seq) across all lra_* dirs."""
    best = {}  # (task, exp, seq) -> (timestamp, row)
    for d in sorted(glob.glob(os.path.join(BENCH, "lra_*"))):
        if not os.path.isdir(d):
            continue
        for path in glob.glob(os.path.join(d, "eval_*.json")):
            with open(path) as f:
                data = json.load(f)
            meta = data.get("experiment_metadata", {})
            perf = data.get("performance_metrics", {})
            ev = perf.get("eval", {})
            seq = meta.get("seq_length")
            if seq is None or (min_seq and seq < min_seq):
                continue
            if min_train and meta.get("dataset_info", {}).get("train_size", 0) < min_train:
                continue
            key = (meta.get("task", "lra_unknown"), meta.get("name", os.path.basename(d)), seq)
            ts = meta.get("timestamp", "")
            row = {
                "task": key[0],
                "exp_name": key[1],
                "seq": seq,
                "f1": ev.get("eval_f1"),
                "accuracy": ev.get("eval_accuracy"),
                "train_time_s": perf.get("training_time_seconds"),
                "peak_mem_mb": perf.get("peak_memory_mb"),
                "inference_ms": perf.get("inference_latency_ms"),
                "softmax_comparisons": perf.get("softmax_comparisons"),
            }
            if key not in best or ts > best[key][0]:
                best[key] = (ts, row)
    return [v[1] for v in best.values()]


def export_csv(rows, path=None):
    path = path or os.path.join(BENCH, "lra_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Task", "Experiment", "Seq", "F1", "Accuracy",
            "Train_Time_s", "Peak_Memory_MB", "Inference_Latency_ms", "Softmax_Comparisons",
        ])
        for r in sorted(rows, key=lambda x: (x["task"], x["exp_name"], x["seq"] or 0)):
            w.writerow([
                r["task"], r["exp_name"], r["seq"], r["f1"], r["accuracy"],
                r["train_time_s"], r["peak_mem_mb"], r["inference_ms"], r["softmax_comparisons"],
            ])
    print(f"Saved: {path}")
    return path


def print_retention_table(rows):
    tasks = sorted({r["task"] for r in rows})
    for task in tasks:
        sub = [r for r in rows if r["task"] == task]
        seqs = sorted({r["seq"] for r in sub if r["seq"] is not None})
        exps = _sorted_exps(r["exp_name"] for r in sub)
        print("\n" + "=" * 90)
        print(f"RETENTION (F1) — {task}")
        print("=" * 90)
        header = f"{'Experiment':<26}" + "".join(f"{s:>10}" for s in seqs)
        print(header)
        print("-" * 90)
        for e in exps:
            cells = []
            for s in seqs:
                match = next((r for r in sub if r["exp_name"] == e and r["seq"] == s), None)
                cells.append(f"{match['f1']:.3f}" if match and match["f1"] is not None else "  -  ")
            print(f"{e:<26}" + "".join(f"{c:>10}" for c in cells))


def plot_retention(rows, out_dir=None):
    out_dir = out_dir or BENCH
    saved = []
    tasks = sorted({r["task"] for r in rows})
    for task in tasks:
        sub = [r for r in rows if r["task"] == task and r["seq"] is not None]
        if not sub:
            continue
        exps = _sorted_exps(r["exp_name"] for r in sub)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"LRA {task}: long-range retention vs context window", fontweight="bold")
        for e in exps:
            pts = sorted((r for r in sub if r["exp_name"] == e), key=lambda x: x["seq"])
            xs = [p["seq"] for p in pts]
            f1s = [p["f1"] for p in pts]
            mems = [p["peak_mem_mb"] for p in pts]
            label = e.replace("exp_", "")
            axes[0].plot(xs, f1s, marker="o", label=label)
            axes[1].plot(xs, mems, marker="o", label=label)
        axes[0].set_xlabel("Context window (tokens)")
        axes[0].set_ylabel("F1")
        axes[0].set_title("F1 vs context window")
        axes[0].set_ylim(0, 1)
        axes[0].set_xscale("log", base=2)
        axes[0].legend(fontsize=6, ncol=2, loc="best")
        axes[1].set_xlabel("Context window (tokens)")
        axes[1].set_ylabel("Peak memory (MB)")
        axes[1].set_title("Peak memory vs context window")
        axes[1].set_xscale("log", base=2)
        axes[1].legend(fontsize=6, ncol=2, loc="best")
        plt.tight_layout()
        path = os.path.join(out_dir, f"lra_retention_{task}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")
        saved.append(path)
    return saved


def plot_efficiency(rows, out_dir=None):
    """Per-task efficiency vs context window: train time, latency, softmax comparisons."""
    out_dir = out_dir or BENCH
    saved = []
    tasks = sorted({r["task"] for r in rows})
    for task in tasks:
        sub = [r for r in rows if r["task"] == task and r["seq"] is not None]
        if not sub:
            continue
        exps = _sorted_exps(r["exp_name"] for r in sub)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"LRA {task}: efficiency vs context window", fontweight="bold")
        for e in exps:
            pts = sorted((r for r in sub if r["exp_name"] == e), key=lambda x: x["seq"])
            xs = [p["seq"] for p in pts]
            label = e.replace("exp_", "")
            axes[0].plot(xs, [p["train_time_s"] for p in pts], marker="o", label=label)
            axes[1].plot(xs, [p["inference_ms"] for p in pts], marker="o", label=label)
            axes[2].plot(xs, [p["softmax_comparisons"] for p in pts], marker="o", label=label)
        for ax, title, ylab in zip(
            axes,
            ["Train time vs context", "Inference latency vs context", "Softmax comparisons vs context"],
            ["Train time (s)", "Latency (ms/seq)", "Softmax comparisons"],
        ):
            ax.set_xlabel("Context window (tokens)")
            ax.set_ylabel(ylab)
            ax.set_title(title)
            ax.set_xscale("log", base=2)
            ax.legend(fontsize=6, ncol=2, loc="best")
        axes[2].set_yscale("log")
        plt.tight_layout()
        path = os.path.join(out_dir, f"lra_efficiency_{task}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")
        saved.append(path)
    return saved


def plot_f1_comparison(rows, out_dir=None):
    """Grouped bar chart of F1 per experiment, one panel per task, bars per seq."""
    out_dir = out_dir or BENCH
    import numpy as np
    saved = []
    tasks = sorted({r["task"] for r in rows})
    for task in tasks:
        sub = [r for r in rows if r["task"] == task]
        if not sub:
            continue
        exps = _sorted_exps(r["exp_name"] for r in sub)
        seqs = sorted({r["seq"] for r in sub if r["seq"] is not None})
        x = np.arange(len(exps))
        width = 0.8 / max(1, len(seqs))
        fig, ax = plt.subplots(figsize=(max(14, len(exps) * 0.9), 6))
        for i, s in enumerate(seqs):
            vals = []
            for e in exps:
                m = next((r for r in sub if r["exp_name"] == e and r["seq"] == s), None)
                vals.append(m["f1"] if m and m["f1"] is not None else 0.0)
            ax.bar(x + i * width, vals, width, label=f"seq {s}")
        ax.set_xticks(x + width * (len(seqs) - 1) / 2)
        ax.set_xticklabels([e.replace("exp_", "") for e in exps], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        ax.set_title(f"LRA {task}: F1 by experiment and context window")
        ax.legend(fontsize=8)
        plt.tight_layout()
        path = os.path.join(out_dir, f"lra_f1_comparison_{task}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")
        saved.append(path)
    return saved


# Survival (OOM) probe uses a tiny budget, so accuracy is meaningless there: the only
# signal is whether a run SURVIVED (ran to completion) or hit OOM / otherwise FAILED.
_SURVIVAL_CODE = {"ok": 0, "collapse": 0, "no_output": 2, "oom": 1, "fail": 2}


def plot_survival_matrix(sweep_path=None, out_dir=None):
    """Experiment x sequence-length survival grid from a sweep JSON (SURVIVED / OOM / FAIL)."""
    out_dir = out_dir or BENCH
    sweep_path = sweep_path or os.path.join(BENCH, "lra_oom_results.json")
    if not os.path.isfile(sweep_path):
        print(f"(survival matrix skipped: {sweep_path} not found)")
        return []
    import numpy as np
    from matplotlib.colors import ListedColormap

    with open(sweep_path) as f:
        data = json.load(f)
    results = data.get("results", [])
    tasks = sorted({r["task"] for r in results})
    saved = []
    cmap = ListedColormap(["#2e7d32", "#c62828", "#616161"])  # survived, oom, fail
    labels = {0: "SURVIVED", 1: "OOM", 2: "FAIL"}
    for task in tasks:
        sub = [r for r in results if r["task"] == task]
        exps = _sorted_exps(r["exp_name"] for r in sub)
        seqs = sorted({r["seq"] for r in sub})
        grid = np.full((len(exps), len(seqs)), 2)
        text = [["" for _ in seqs] for _ in exps]
        for i, e in enumerate(exps):
            for j, s in enumerate(seqs):
                m = next((r for r in sub if r["exp_name"] == e and r["seq"] == s), None)
                if m is None:
                    grid[i, j] = 2
                    text[i][j] = "-"
                    continue
                code = _SURVIVAL_CODE.get(m.get("status", "fail"), 2)
                grid[i, j] = code
                text[i][j] = labels[code]
        fig, ax = plt.subplots(figsize=(max(6, len(seqs) * 1.4), max(4, len(exps) * 0.5)))
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, aspect="auto")
        ax.set_xticks(range(len(seqs)))
        ax.set_xticklabels(seqs)
        ax.set_yticks(range(len(exps)))
        ax.set_yticklabels([e.replace("exp_", "") for e in exps], fontsize=8)
        ax.set_xlabel("Context window (tokens)")
        ax.set_title(f"LRA {task}: survival vs context window (SURVIVED / OOM / FAIL)")
        for i in range(len(exps)):
            for j in range(len(seqs)):
                ax.text(j, i, text[i][j], ha="center", va="center", color="white", fontsize=8)
        plt.tight_layout()
        path = os.path.join(out_dir, f"lra_survival_{task}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")
        saved.append(path)
    return saved


def plot_sweep_efficiency(sweep_path=None, out_dir=None):
    """All-experiments efficiency vs context from a single sweep JSON (same budget => comparable).

    Uses the OOM-probe sweep so every one of the 13 experiments is represented. Peak memory,
    softmax comparisons, latency and train time are all measured under one fixed budget, so the
    only differences are architectural. OOM/failed runs leave gaps (the line stops).
    """
    out_dir = out_dir or BENCH
    sweep_path = sweep_path or os.path.join(BENCH, "lra_oom_results.json")
    if not os.path.isfile(sweep_path):
        print(f"(all-exp efficiency skipped: {sweep_path} not found)")
        return []
    with open(sweep_path) as f:
        data = json.load(f)
    # "collapse" survived (it produced metrics) -- only its F1 is meaningless at probe budget;
    # peak memory / softmax / latency are still valid. Exclude only OOM/fail (no metrics).
    results = [r for r in data.get("results", []) if r.get("status") in ("ok", "collapse")]
    if not results:
        print("(all-exp efficiency skipped: no successful runs in sweep)")
        return []
    tasks = sorted({r["task"] for r in results})
    saved = []
    for task in tasks:
        sub = [r for r in results if r["task"] == task]
        exps = _sorted_exps(r["exp_name"] for r in sub)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"LRA {task}: all-experiment efficiency vs context window "
            f"(fixed OOM-probe budget; gaps = OOM/fail)", fontweight="bold")
        for e in exps:
            pts = sorted((r for r in sub if r["exp_name"] == e), key=lambda x: x["seq"])
            xs = [p["seq"] for p in pts]
            label = e.replace("exp_", "")
            axes[0].plot(xs, [p.get("peak_mem_mb") for p in pts], marker="o", label=label)
            axes[1].plot(xs, [p.get("softmax_comparisons") for p in pts], marker="o", label=label)
            axes[2].plot(xs, [p.get("inference_ms") for p in pts], marker="o", label=label)
        for ax, title, ylab in zip(
            axes,
            ["Peak memory vs context", "Softmax comparisons vs context", "Inference latency vs context"],
            ["Peak memory (MB)", "Softmax comparisons", "Latency (ms/seq)"],
        ):
            ax.set_xlabel("Context window (tokens)")
            ax.set_ylabel(ylab)
            ax.set_title(title)
            ax.set_xscale("log", base=2)
            ax.legend(fontsize=7, ncol=2)
        axes[1].set_yscale("log")
        plt.tight_layout()
        path = os.path.join(out_dir, f"lra_all_efficiency_{task}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")
        saved.append(path)
    return saved


def main():
    rows = load_lra_results()
    if not rows:
        print("No LRA results found. Run python -m eval.lra.run or eval.lra.sweep first.")
        sys.exit(1)
    print(f"Loaded {len(rows)} LRA result(s).")
    print_retention_table(rows)
    export_csv(rows)
    if HAS_MPL:
        plot_retention(rows)
        plot_efficiency(rows)
        plot_f1_comparison(rows)
        plot_survival_matrix()
        plot_sweep_efficiency()
    else:
        print("Install matplotlib to generate plots.")


if __name__ == "__main__":
    main()
