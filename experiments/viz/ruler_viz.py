#!/usr/bin/env python3
"""Visualize RULER-style long-context results: accuracy vs depth and context.

Reads ``benchmarks/ruler_<task>_<exp>/eval_*.json``, produces depth retention heatmaps,
accuracy-vs-depth curves, and ``benchmarks/ruler_comparison.csv``.
"""

import csv
import glob
import json
import os
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
MIN_TRAIN = 100


def _exp_sort_key(name):
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return name


def _sorted_exps(names):
    return sorted(set(names), key=_exp_sort_key)


def load_ruler_results(min_train=MIN_TRAIN):
    best = {}
    for d in sorted(glob.glob(os.path.join(BENCH, "ruler_*"))):
        if not os.path.isdir(d):
            continue
        for path in glob.glob(os.path.join(d, "eval_*.json")):
            with open(path) as f:
                data = json.load(f)
            meta = data.get("experiment_metadata", {})
            perf = data.get("performance_metrics", {})
            ev = perf.get("eval", {})
            mc = meta.get("model_config", {})
            seq = meta.get("seq_length")
            depth = mc.get("needle_depth")
            if seq is None or depth is None:
                continue
            if min_train and meta.get("dataset_info", {}).get("train_size", 0) < min_train:
                continue
            key = (meta.get("task", "ruler_unknown"), meta.get("name", os.path.basename(d)), seq, depth)
            ts = meta.get("timestamp", "")
            row = {
                "task": key[0],
                "exp_name": key[1],
                "seq": seq,
                "depth": depth,
                "accuracy": ev.get("eval_accuracy"),
                "f1": ev.get("eval_f1"),
                "train_time_s": perf.get("training_time_seconds"),
                "peak_mem_mb": perf.get("peak_memory_mb"),
                "inference_ms": perf.get("inference_latency_ms"),
                "softmax_comparisons": perf.get("softmax_comparisons"),
            }
            if key not in best or ts > best[key][0]:
                best[key] = (ts, row)
    return [v[1] for v in best.values()]


def export_csv(rows, path=None):
    path = path or os.path.join(BENCH, "ruler_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Task", "Experiment", "Seq", "Needle_Depth", "Accuracy", "F1",
            "Train_Time_s", "Peak_Memory_MB", "Inference_Latency_ms", "Softmax_Comparisons",
        ])
        for r in sorted(rows, key=lambda x: (x["task"], _exp_sort_key(x["exp_name"]), x["seq"], x["depth"])):
            w.writerow([
                r["task"], r["exp_name"], r["seq"], r["depth"], r["accuracy"], r["f1"],
                r["train_time_s"], r["peak_mem_mb"], r["inference_ms"], r["softmax_comparisons"],
            ])
    print(f"Saved: {path}")
    return path


def print_depth_table(rows):
    tasks = sorted({r["task"] for r in rows})
    for task in tasks:
        sub = [r for r in rows if r["task"] == task]
        seqs = sorted({r["seq"] for r in sub})
        depths = sorted({r["depth"] for r in sub})
        exps = _sorted_exps(r["exp_name"] for r in sub)
        print("\n" + "=" * 100)
        print(f"RULER accuracy — {task} (rows=experiment, cols=seq@depth)")
        print("=" * 100)
        for s in seqs:
            for d in depths:
                print(f"\n  seq={s} depth={d:.2f}")
                print(f"  {'Experiment':<26} {'Acc':>8} {'Mem(MB)':>10}")
                print("  " + "-" * 48)
                for e in exps:
                    m = next((r for r in sub if r["exp_name"] == e and r["seq"] == s and r["depth"] == d), None)
                    if m:
                        acc = f"{m['accuracy']:.3f}" if m["accuracy"] is not None else "N/A"
                        mem = f"{m['peak_mem_mb']:.0f}" if m["peak_mem_mb"] else "N/A"
                        print(f"  {e:<26} {acc:>8} {mem:>10}")


def plot_depth_curves(rows, out_dir=None):
    """Accuracy vs needle depth, one panel per (task, seq)."""
    out_dir = out_dir or BENCH
    saved = []
    tasks = sorted({r["task"] for r in rows})
    for task in tasks:
        sub = [r for r in rows if r["task"] == task]
        seqs = sorted({r["seq"] for r in sub})
        exps = _sorted_exps(r["exp_name"] for r in sub)
        n = len(seqs)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
        fig.suptitle(f"RULER {task}: retrieval accuracy vs needle depth", fontweight="bold")
        for i, s in enumerate(seqs):
            ax = axes[0, i]
            for e in exps:
                pts = sorted(
                    (r for r in sub if r["exp_name"] == e and r["seq"] == s),
                    key=lambda x: x["depth"],
                )
                if not pts:
                    continue
                ax.plot([p["depth"] for p in pts], [p["accuracy"] for p in pts],
                        marker="o", label=e.replace("exp_", ""))
            ax.axhline(0.10, color="gray", linestyle="--", linewidth=0.8, label="chance (10-way)")
            ax.set_xlabel("Needle depth (0=start, 1=end)")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"seq={s}")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=6, ncol=2, loc="best")
        plt.tight_layout()
        path = os.path.join(out_dir, f"ruler_depth_curves_{task.replace('ruler_', '')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")
        saved.append(path)
    return saved


def plot_depth_heatmap(rows, out_dir=None, baseline_exp="exp_0_baseline"):
    """Heatmap: accuracy vs (seq × depth) for baseline vs best sparse (or per-exp grid)."""
    out_dir = out_dir or BENCH
    saved = []
    tasks = sorted({r["task"] for r in rows})
    for task in tasks:
        sub = [r for r in rows if r["task"] == task]
        exps = _sorted_exps(r["exp_name"] for r in sub)
        seqs = sorted({r["seq"] for r in sub})
        depths = sorted({r["depth"] for r in sub})
        n_exp = len(exps)
        fig, axes = plt.subplots(1, min(n_exp, 4), figsize=(4 * min(n_exp, 4), 4), squeeze=False)
        # Show up to 4 representative exps: baseline + 3 efficient ones if present
        pick = [e for e in exps if e == baseline_exp]
        for e in exps:
            if e != baseline_exp and len(pick) < 4:
                pick.append(e)
        for i, e in enumerate(pick[:4]):
            ax = axes[0, i]
            grid = np.full((len(depths), len(seqs)), np.nan)
            for di, d in enumerate(depths):
                for si, s in enumerate(seqs):
                    m = next((r for r in sub if r["exp_name"] == e and r["seq"] == s and r["depth"] == d), None)
                    if m and m["accuracy"] is not None:
                        grid[di, si] = m["accuracy"]
            im = ax.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
            ax.set_xticks(range(len(seqs)))
            ax.set_xticklabels(seqs, fontsize=8)
            ax.set_yticks(range(len(depths)))
            ax.set_yticklabels([f"{d:.1f}" for d in depths], fontsize=8)
            ax.set_xlabel("Context (tokens)")
            ax.set_ylabel("Needle depth")
            ax.set_title(e.replace("exp_", ""))
            for di in range(len(depths)):
                for si in range(len(seqs)):
                    if not np.isnan(grid[di, si]):
                        ax.text(si, di, f"{grid[di, si]:.2f}", ha="center", va="center", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"RULER {task}: accuracy heatmap (seq × depth)", fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, f"ruler_depth_heatmap_{task.replace('ruler_', '')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")
        saved.append(path)
    return saved


def main():
    rows = load_ruler_results()
    if not rows:
        print("No RULER results found. Run python -m eval.ruler.run or eval.ruler.sweep first.")
        sys.exit(1)
    print(f"Loaded {len(rows)} RULER result(s).")
    print_depth_table(rows)
    export_csv(rows)
    if HAS_MPL:
        plot_depth_curves(rows)
        plot_depth_heatmap(rows)
    else:
        print("Install matplotlib to generate plots.")


if __name__ == "__main__":
    main()
