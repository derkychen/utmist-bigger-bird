#!/usr/bin/env python3
"""Scaling law trajectory visualization - how metrics evolve with data/compute scale."""

import json
import os
import glob
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Install matplotlib and numpy: pip install matplotlib numpy")
    exit(1)

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")

@dataclass
class ScalingRun:
    exp_name: str
    timestamp: str
    train_samples: int
    eval_samples: int
    max_seq_len: int
    epochs: int
    f1: float
    accuracy: float
    train_time: float
    train_loss: float
    eval_loss: float
    throughput: float  # samples/sec
    raw: Dict[str, Any]

def load_all_runs() -> List[ScalingRun]:
    """Load all historical runs for scaling analysis."""
    runs = []
    
    exp_dirs = [d for d in os.listdir(BENCHMARK_DIR) 
                if os.path.isdir(os.path.join(BENCHMARK_DIR, d)) and not d.startswith('.')]
    
    for exp_dir in exp_dirs:
        json_files = glob.glob(os.path.join(BENCHMARK_DIR, exp_dir, "eval_*.json"))
        for json_path in json_files:
            try:
                with open(json_path) as f:
                    data = json.load(f)
                
                meta = data["experiment_metadata"]
                perf = data["performance_metrics"]
                
                runs.append(ScalingRun(
                    exp_name=meta["name"],
                    timestamp=meta["timestamp"],
                    train_samples=meta["dataset_info"]["train_size"],
                    eval_samples=meta["dataset_info"]["eval_size"],
                    max_seq_len=meta["dataset_info"].get("max_seq_len", 256),
                    epochs=meta["training_config"]["epochs"],
                    f1=perf["eval"].get("eval_f1", 0),
                    accuracy=perf["eval"].get("eval_accuracy", 0),
                    train_time=perf["training_time_seconds"],
                    train_loss=perf["train"].get("train_loss", 0),
                    eval_loss=perf["eval"].get("eval_loss", 0),
                    throughput=perf["train"].get("train_samples_per_second", 0),
                    raw=data
                ))
            except Exception as e:
                print(f"Warning: Could not load {json_path}: {e}")
    
    return runs

def group_by_experiment(runs: List[ScalingRun]) -> Dict[str, List[ScalingRun]]:
    """Group runs by experiment name."""
    groups = defaultdict(list)
    for r in runs:
        groups[r.exp_name].append(r)
    # Sort each group by training samples (scale)
    for exp_name in groups:
        groups[exp_name].sort(key=lambda x: (x.train_samples, x.epochs, x.timestamp))
    return dict(groups)

def plot_scaling_trajectories(grouped_runs: Dict[str, List[ScalingRun]], save_dir: str):
    """Create scaling law visualizations."""
    
    # Define color palette
    colors = {
        'exp_1_deepseek_topk': '#1f77b4',
        'exp_2_lightning_hybrid': '#ff7f0e', 
        'exp_3_dynamic_globals': '#2ca02c',
        'exp_4_pbs_attn': '#d62728',
        'exp_5_bigger_bird': '#9467bd',
        'exp_6_deepseek_pbs': '#8c564b',
        'exp_7_layer_adaptive': '#e377c2',
        'exp_8_token_drop': '#7f7f7f',
        'exp_9_attn_specul': '#bcbd22',
        'exp_10_gqa_sparse': '#17becf',
        'exp_11_nsa': '#1a9850',
        'exp_12_s2_hhst': '#d73027',
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Scaling Law Trajectories', fontsize=16, fontweight='bold')
    
    # 1. F1 vs Training Samples (Data Scaling)
    ax = axes[0, 0]
    for exp_name, runs in sorted(grouped_runs.items()):
        if len(runs) > 1:  # Only plot if multiple runs
            samples = [r.train_samples for r in runs]
            f1s = [r.f1 for r in runs]
            ax.plot(samples, f1s, 'o-', label=exp_name.replace('exp_', ''), 
                   color=colors.get(exp_name, 'gray'), linewidth=2, markersize=8)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 vs Data Scale')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 2. Accuracy vs Training Samples
    ax = axes[0, 1]
    for exp_name, runs in sorted(grouped_runs.items()):
        if len(runs) > 1:
            samples = [r.train_samples for r in runs]
            accs = [r.accuracy for r in runs]
            ax.plot(samples, accs, 'o-', label=exp_name.replace('exp_', ''),
                   color=colors.get(exp_name, 'gray'), linewidth=2, markersize=8)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Data Scale')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 3. F1 vs Training Time (Compute Scaling)
    ax = axes[0, 2]
    for exp_name, runs in sorted(grouped_runs.items()):
        times = [r.train_time for r in runs]
        f1s = [r.f1 for r in runs]
        marker = 'o' if len(runs) > 1 else 's'
        ax.scatter(times, f1s, label=exp_name.replace('exp_', ''),
                  color=colors.get(exp_name, 'gray'), s=100, marker=marker, alpha=0.7)
        if len(runs) > 1:
            ax.plot(times, f1s, '-', color=colors.get(exp_name, 'gray'), alpha=0.5)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 vs Compute Budget')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Loss Curves (Train vs Eval)
    ax = axes[1, 0]
    for exp_name, runs in sorted(grouped_runs.items()):
        if len(runs) > 1:
            samples = [r.train_samples for r in runs]
            train_losses = [r.train_loss for r in runs]
            eval_losses = [r.eval_loss for r in runs]
            ax.plot(samples, train_losses, 'o--', label=f"{exp_name.replace('exp_', '')} (train)",
                   color=colors.get(exp_name, 'gray'), alpha=0.6)
            ax.plot(samples, eval_losses, 's-', label=f"{exp_name.replace('exp_', '')} (eval)",
                   color=colors.get(exp_name, 'gray'), linewidth=2)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Trajectories')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 5. Throughput vs Sequence Length
    ax = axes[1, 1]
    for exp_name, runs in sorted(grouped_runs.items()):
        lens = [r.max_seq_len for r in runs]
        throughputs = [r.throughput for r in runs]
        ax.scatter(lens, throughputs, label=exp_name.replace('exp_', ''),
                  color=colors.get(exp_name, 'gray'), s=100, alpha=0.7)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Throughput (samples/sec)')
    ax.set_title('Speed vs Sequence Length')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 6. Pareto Frontier: F1 vs Training Time
    ax = axes[1, 2]
    all_points = []
    for exp_name, runs in grouped_runs.items():
        for r in runs:
            all_points.append((r.train_time, r.f1, exp_name, r.train_samples))
    
    # Plot all points
    for exp_name, runs in sorted(grouped_runs.items()):
        points = [(r.train_time, r.f1, r.train_samples) for r in runs]
        times = [p[0] for p in points]
        f1s = [p[1] for p in points]
        ax.scatter(times, f1s, label=exp_name.replace('exp_', ''),
                  color=colors.get(exp_name, 'gray'), s=150, alpha=0.7, edgecolors='black')
        
        # Annotate with sample size
        for t, f, s in points:
            ax.annotate(f'{s}', (t, f), fontsize=7, alpha=0.7)
    
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Pareto Frontier: Quality vs Speed')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "scaling_laws.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Scaling law visualization saved to: {save_path}")
    
    # Also create a summary text
    summary_path = os.path.join(save_dir, "scaling_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("SCALING LAW ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        for exp_name, runs in sorted(grouped_runs.items()):
            f.write(f"\n{exp_name}:\n")
            f.write("-"*40 + "\n")
            for r in runs:
                f.write(f"  Samples={r.train_samples:>5}, Epochs={r.epochs}, "
                       f"F1={r.f1:.3f}, Time={r.train_time:.1f}s\n")
            
            if len(runs) > 1:
                # Calculate scaling trend
                f1_start = runs[0].f1
                f1_end = runs[-1].f1
                samples_ratio = runs[-1].train_samples / max(runs[0].train_samples, 1)
                f1_improvement = f1_end - f1_start
                f.write(f"  -> Scaling: {samples_ratio:.1f}x data → F1 Δ={f1_improvement:+.3f}\n")
    
    print(f"Summary saved to: {summary_path}")

def main():
    runs = load_all_runs()
    
    if not runs:
        print("No runs found. Train experiments with different sample sizes!")
        return
    
    grouped = group_by_experiment(runs)
    
    print(f"Loaded {len(runs)} runs across {len(grouped)} experiments:")
    for exp_name, exp_runs in sorted(grouped.items()):
        print(f"  {exp_name}: {len(exp_runs)} runs")
    
    print()
    plot_scaling_trajectories(grouped, BENCHMARK_DIR)

if __name__ == "__main__":
    main()
