#!/usr/bin/env python3
"""Compare experiment results across all benchmark runs."""

import json
import os
import glob
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")

@dataclass
class ExperimentResult:
    name: str
    timestamp: str
    f1: float
    accuracy: float
    train_time: float
    eval_time: float
    train_samples: int
    eval_samples: int
    epochs: int
    train_loss: float
    eval_loss: float
    peak_memory_mb: float = 0.0
    inference_latency_ms: float = None
    softmax_comparisons: int = None
    raw: Dict[str, Any] = None

def _is_fixed_length(meta: Dict[str, Any]) -> bool:
    if meta.get("fixed_length"):
        return True
    di = meta.get("dataset_info", {})
    mc = meta.get("model_config", {})
    return bool(di.get("fixed_length") or mc.get("fixed_length"))


def _passes_filters(meta: Dict[str, Any], fixed_length_only: bool, min_timestamp: str) -> bool:
    if fixed_length_only and not _is_fixed_length(meta):
        return False
    if min_timestamp and meta.get("timestamp", "") < min_timestamp:
        return False
    return True


def load_all_results(
    fixed_length_only: bool = False,
    min_timestamp: str = "",
) -> List[ExperimentResult]:
    """Load all experiment results from benchmark JSON files."""
    results = []
    
    exp_dirs = [d for d in os.listdir(BENCHMARK_DIR) 
                if os.path.isdir(os.path.join(BENCHMARK_DIR, d)) and not d.startswith('.')]
    
    for exp_dir in exp_dirs:
        json_files = glob.glob(os.path.join(BENCHMARK_DIR, exp_dir, "eval_*.json"))
        for json_path in json_files:
            try:
                with open(json_path) as f:
                    data = json.load(f)
                
                meta = data["experiment_metadata"]
                if not _passes_filters(meta, fixed_length_only, min_timestamp):
                    continue
                perf = data["performance_metrics"]
                
                results.append(ExperimentResult(
                    name=meta["name"],
                    timestamp=meta["timestamp"],
                    f1=perf["eval"].get("eval_f1", 0),
                    accuracy=perf["eval"].get("eval_accuracy", 0),
                    train_time=perf["training_time_seconds"],
                    eval_time=perf["eval"].get("eval_runtime", 0),
                    train_samples=meta["dataset_info"]["train_size"],
                    eval_samples=meta["dataset_info"]["eval_size"],
                    epochs=meta["training_config"]["epochs"],
                    train_loss=perf["train"].get("train_loss", 0),
                    eval_loss=perf["eval"].get("eval_loss", 0),
                    peak_memory_mb=perf.get("peak_memory_mb", 0) or meta.get("environment", {}).get("peak_memory_mb", 0),
                    inference_latency_ms=perf.get("inference_latency_ms"),
                    softmax_comparisons=perf.get("softmax_comparisons"),
                    raw=data
                ))
            except Exception as e:
                print(f"Warning: Could not load {json_path}: {e}")
    
    return sorted(results, key=lambda x: (x.name, x.timestamp))

def group_by_sample_size(results: List[ExperimentResult]) -> Dict[int, Dict[str, ExperimentResult]]:
    """Group results by train_samples. Within each group, keep latest run per experiment."""
    groups: Dict[int, Dict[str, ExperimentResult]] = {}
    for r in results:
        n = r.train_samples
        if n not in groups:
            groups[n] = {}
        if r.name not in groups[n] or r.timestamp > groups[n][r.name].timestamp:
            groups[n][r.name] = r
    return dict(sorted(groups.items()))


def print_comparison_table(groups: Dict[int, Dict[str, ExperimentResult]]):
    """Print one comparison table per sample-size group."""
    if not groups:
        print("No results found. Run experiments first.")
        return

    for n_samples, exps in groups.items():
        print("\n" + "="*100)
        print(f"COMPARISON — {n_samples} training samples (latest run per experiment)")
        print("="*100)
        header = f"{'Experiment':<25} {'F1':>8} {'Acc':>7} {'Train(s)':>10} {'Peak(MB)':>10} {'Lat(ms)':>10}"
        print(header)
        print("-"*100)
        for name, r in sorted(exps.items()):
            lat = f"{r.inference_latency_ms:.1f}" if r.inference_latency_ms is not None else "N/A"
            print(f"{name:<25} {r.f1:>8.3f} {r.accuracy:>7.3f} {r.train_time:>10.1f} {r.peak_memory_mb:>10.0f} {lat:>10}")
        print("="*100)
        best_f1 = max(exps.values(), key=lambda x: x.f1)
        fastest = min(exps.values(), key=lambda x: x.train_time)
        print(f"  Best F1:          {best_f1.name} ({best_f1.f1:.3f})")
        print(f"  Fastest Training: {fastest.name} ({fastest.train_time:.1f}s)")


def plot_comparison(groups: Dict[int, Dict[str, ExperimentResult]], out_dir: str):
    """Generate one comparison PNG per sample-size group."""
    if not HAS_MATPLOTLIB:
        print("\nNote: Install matplotlib for visualization: pip install matplotlib")
        return

    saved = []
    for n_samples, exps in groups.items():
        exp_names = sorted(exps.keys())
        f1s   = [exps[e].f1         for e in exp_names]
        accs  = [exps[e].accuracy   for e in exp_names]
        times = [exps[e].train_time for e in exp_names]

        short_names = [e.replace("exp_", "").replace("_", "\n") for e in exp_names]

        mems  = [exps[e].peak_memory_mb for e in exp_names]
        lats  = [exps[e].inference_latency_ms or 0 for e in exp_names]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Experiment Comparison — {n_samples} training samples", fontsize=13, fontweight='bold')

        axes[0, 0].bar(short_names, f1s, color='steelblue')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_title('F1 Score')
        axes[0, 0].set_ylim(0, 1)

        axes[0, 1].bar(short_names, accs, color='forestgreen')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_ylim(0, 1)

        axes[0, 2].bar(short_names, times, color='coral')
        axes[0, 2].set_ylabel('Training Time (s)')
        axes[0, 2].set_title('Training Time')

        axes[1, 0].bar(short_names, mems, color='mediumpurple')
        axes[1, 0].set_ylabel('Peak Memory (MB)')
        axes[1, 0].set_title('Peak Memory')

        axes[1, 1].bar(short_names, lats, color='goldenrod')
        axes[1, 1].set_ylabel('Inference Latency (ms)')
        axes[1, 1].set_title('Inference Latency')

        # Softmax comparison reduction
        comps = [exps[e].softmax_comparisons for e in exp_names]
        valid_comps = [c for c in comps if c is not None]
        baseline_comp = max(valid_comps) if valid_comps else 1
        reductions = []
        for c in comps:
            if c is not None and baseline_comp and baseline_comp != c:
                reductions.append((1 - c / baseline_comp) * 100)
            else:
                reductions.append(0)
        axes[1, 2].bar(short_names, reductions, color='teal')
        axes[1, 2].set_ylabel('Reduction vs Baseline (%)')
        axes[1, 2].set_title('Softmax Comparison Reduction')

        plt.tight_layout()
        path = os.path.join(out_dir, f"comparison_{n_samples}samples.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        saved.append(path)
        print(f"Saved: {path}")

    return saved


def export_csv(results: List[ExperimentResult], csv_path: str):
    """Export all results to CSV."""
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Timestamp', 'Train_Samples', 'F1', 'Accuracy',
                         'Train_Time_s', 'Eval_Time_s', 'Epochs', 'Train_Loss', 'Eval_Loss',
                         'Peak_Memory_MB', 'Inference_Latency_ms', 'Softmax_Comparisons'])
        for r in results:
            writer.writerow([r.name, r.timestamp, r.train_samples, r.f1, r.accuracy,
                             r.train_time, r.eval_time, r.epochs, r.train_loss, r.eval_loss,
                             r.peak_memory_mb, r.inference_latency_ms, r.softmax_comparisons])
    print(f"CSV exported: {csv_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare experiment benchmark JSONs")
    parser.add_argument(
        "--fixed-length-only",
        action="store_true",
        help="Only include runs with experiment_metadata.fixed_length=true",
    )
    parser.add_argument(
        "--min-timestamp",
        type=str,
        default="",
        help="Only include runs at or after this timestamp (e.g. 20260531)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="CSV output path (default: benchmarks/comparison.csv or comparison_fixed_length.csv)",
    )
    args = parser.parse_args()

    results = load_all_results(
        fixed_length_only=args.fixed_length_only,
        min_timestamp=args.min_timestamp,
    )

    if not results:
        print("No experiment results found. Run some experiments first!")
        return

    groups = group_by_sample_size(results)

    label = "fixed-length " if args.fixed_length_only else ""
    print(f"\nFound {len(results)} {label}runs across {len(groups)} sample-size group(s): {sorted(groups.keys())}")
    if args.fixed_length_only or args.min_timestamp:
        print(
            "Note: Pre-May-2026 6000-sample runs used dynamic padding (~166 tokens). "
            "See context/long-context-leaderboard-may2026.md."
        )

    print_comparison_table(groups)
    plot_comparison(groups, out_dir=BENCHMARK_DIR)

    csv_path = args.csv or os.path.join(
        BENCHMARK_DIR,
        "comparison_fixed_length.csv" if args.fixed_length_only else "comparison.csv",
    )
    export_csv(results, csv_path)

if __name__ == "__main__":
    main()
