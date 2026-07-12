#!/usr/bin/env python3
"""Visualize long-context sweep results (F1 and train time vs sequence length)."""

import json
import os
import sys

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
SWEEP_JSON = os.path.join(BENCH, "long_context_sweep_results.json")

EXP_LABELS = {
    "exp_0_baseline": "0 baseline",
    "exp_1_deepseek_topk": "1 top-k",
    "exp_2_lightning_hybrid": "2 hybrid",
    "exp_3_dynamic_globals": "3 globals",
    "exp_4_pbs_attn": "4 pbs",
    "exp_5_bigger_bird": "5 bigger bird",
    "exp_6_deepseek_pbs": "6 deepseek pbs",
    "exp_7_layer_adaptive": "7 layer adaptive",
    "exp_8_token_drop": "8 token drop",
    "exp_9_attn_specul": "9 attn specul",
    "exp_10_gqa_sparse": "10 gqa sparse",
}


def load_sweep(path=None):
    path = path or SWEEP_JSON
    if not os.path.isfile(path):
        print(f"Missing {path}. Run run_long_context_sweep.py first.")
        return None
    with open(path) as f:
        return json.load(f)


def plot_sweep(data, out_dir=None):
    out_dir = out_dir or BENCH
    results = [r for r in data["results"] if not r.get("oom")]
    if not results:
        print("No successful runs to plot.")
        return []

    exps = sorted({r["exp_name"] for r in results}, key=lambda x: x)
    seqs = sorted({r["seq"] for r in results})

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Long-Context Sweep (fixed-length IMDb, "
        f"{data['config'].get('train_samples', '?')} train samples)",
        fontsize=12,
        fontweight="bold",
    )

    for exp_name in exps:
        label = EXP_LABELS.get(exp_name, exp_name)
        sub = [r for r in results if r["exp_name"] == exp_name]
        xs = [r["seq"] for r in sorted(sub, key=lambda x: x["seq"])]
        f1s = [r.get("f1") for r in sorted(sub, key=lambda x: x["seq"])]
        times = [r.get("train_time_s") for r in sorted(sub, key=lambda x: x["seq"])]
        mems = [r.get("peak_mem_mb") for r in sorted(sub, key=lambda x: x["seq"])]
        axes[0].plot(xs, f1s, marker="o", label=label)
        axes[1].plot(xs, times, marker="o", label=label)
        axes[2].plot(xs, mems, marker="o", label=label)

    axes[0].set_xlabel("Sequence length")
    axes[0].set_ylabel("F1")
    axes[0].set_title("F1 vs seq length")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    axes[0].set_xscale("log", base=2)

    axes[1].set_xlabel("Sequence length")
    axes[1].set_ylabel("Train time (s)")
    axes[1].set_title("Training time vs seq length")
    axes[1].legend(fontsize=8)
    axes[1].set_xscale("log", base=2)

    axes[2].set_xlabel("Sequence length")
    axes[2].set_ylabel("Peak memory (MB)")
    axes[2].set_title("Peak memory vs seq length")
    axes[2].legend(fontsize=8)
    axes[2].set_xscale("log", base=2)

    plt.tight_layout()
    path = os.path.join(out_dir, "long_context_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return [path]


def export_sweep_csv(data, out_dir=None):
    import csv
    out_dir = out_dir or BENCH
    path = os.path.join(out_dir, "long_context_sweep_table.csv")
    rows = [r for r in data["results"] if not r.get("oom")]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exp_name", "seq", "f1", "accuracy", "train_time_s", "peak_mem_mb", "inference_ms", "softmax_comparisons"])
        for r in sorted(rows, key=lambda x: (x["exp_name"], x["seq"])):
            w.writerow([
                r["exp_name"], r["seq"],
                r.get("f1"), r.get("accuracy"),
                r.get("train_time_s"), r.get("peak_mem_mb"),
                r.get("inference_ms"), r.get("softmax_comparisons"),
            ])
    print(f"Saved: {path}")
    return path


def main():
    data = load_sweep()
    if data is None:
        sys.exit(1)
    export_sweep_csv(data)
    if not HAS_MPL:
        print("Install matplotlib to generate plots.")
        sys.exit(1)
    plot_sweep(data)


if __name__ == "__main__":
    main()
