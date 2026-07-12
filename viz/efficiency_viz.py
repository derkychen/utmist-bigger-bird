#!/usr/bin/env python3
"""
Visualize long-context efficiency results.
Reads benchmarks/efficiency_results.json and generates plots.

Usage:
  python viz/efficiency_viz.py
"""

import json
import os
import sys
from typing import List, Dict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
RESULTS_PATH = os.path.join(BENCHMARK_DIR, "efficiency_results.json")


def load_efficiency_results() -> List[Dict]:
    if not os.path.exists(RESULTS_PATH):
        print(f"No efficiency results found at {RESULTS_PATH}")
        print("Run: python benchmarks/efficiency_eval.py")
        return []
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return data.get("results", [])


def plot_metric(results: List[Dict], metric: str, ylabel: str, title: str, out_path: str):
    """Plot a single metric vs sequence length, one line per experiment."""
    if not HAS_MATPLOTLIB:
        print("Install matplotlib: pip install matplotlib")
        return

    # Group by experiment
    by_exp = {}
    for r in results:
        if r.get("oom") or r.get(metric) is None:
            continue
        name = r["exp_name"]
        if name not in by_exp:
            by_exp[name] = []
        by_exp[name].append((r["seq_length"], r[metric]))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, (name, pts) in enumerate(sorted(by_exp.items())):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = name.replace("exp_", "").replace("_", " ")
        ax.plot(xs, ys, marker='o', label=label, color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_oom_matrix(results: List[Dict], out_path: str):
    """Heatmap showing which (exp, seq_len) combinations OOM'd."""
    if not HAS_MATPLOTLIB:
        return

    exps = sorted(set(r["exp_name"] for r in results))
    seqs = sorted(set(r["seq_length"] for r in results))

    matrix = []
    for exp in exps:
        row = []
        for seq in seqs:
            matches = [r for r in results if r["exp_name"] == exp and r["seq_length"] == seq]
            if matches and matches[0].get("oom"):
                row.append(1)  # OOM
            elif matches and matches[0].get("f1") is not None:
                row.append(0)  # OK
            else:
                row.append(0.5)  # Error/other
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(seqs)))
    ax.set_xticklabels(seqs)
    ax.set_yticks(range(len(exps)))
    ax.set_yticklabels([e.replace("exp_", "") for e in exps])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Experiment")
    ax.set_title("OOM Heatmap (red=OOM, green=OK)", fontweight='bold')
    plt.colorbar(im, ax=ax, label="Status")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    results = load_efficiency_results()
    if not results:
        return

    print(f"Loaded {len(results)} efficiency results")

    out_dir = BENCHMARK_DIR
    plot_metric(results, "f1", "F1 Score", "F1 vs Sequence Length", os.path.join(out_dir, "efficiency_f1.png"))
    plot_metric(results, "peak_memory_mb", "Peak Memory (MB)", "Memory vs Sequence Length", os.path.join(out_dir, "efficiency_memory.png"))
    plot_metric(results, "train_samples_per_sec", "Throughput (samples/sec)", "Throughput vs Sequence Length", os.path.join(out_dir, "efficiency_throughput.png"))
    plot_metric(results, "inference_latency_ms", "Inference Latency (ms)", "Latency vs Sequence Length", os.path.join(out_dir, "efficiency_latency.png"))
    plot_oom_matrix(results, os.path.join(out_dir, "efficiency_oom_matrix.png"))

    print(f"\nAll efficiency plots saved to {out_dir}")


if __name__ == "__main__":
    main()
