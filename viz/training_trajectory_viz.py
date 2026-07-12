#!/usr/bin/env python3
"""Visualize training trajectories (per-epoch metrics) for each experiment."""

import json
import os
import glob
from typing import List, Dict, Any
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Install matplotlib: pip install matplotlib")
    exit(1)

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")

@dataclass
class TrainingRun:
    exp_name: str
    timestamp: str
    train_samples: int
    epochs: int
    trajectory: List[Dict]  # per-epoch metrics
    final_f1: float
    final_acc: float
    raw: Dict[str, Any]

def load_all_runs() -> List[TrainingRun]:
    """Load all runs with trajectory data."""
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
                traj = perf.get("trajectory", [])

                if not traj:
                    continue  # Skip old runs without trajectory

                runs.append(TrainingRun(
                    exp_name=meta["name"],
                    timestamp=meta["timestamp"],
                    train_samples=meta["dataset_info"]["train_size"],
                    epochs=meta["training_config"]["epochs"],
                    trajectory=traj,
                    final_f1=perf["eval"].get("eval_f1", 0),
                    final_acc=perf["eval"].get("eval_accuracy", 0),
                    raw=data
                ))
            except Exception as e:
                print(f"Warning: Could not load {json_path}: {e}")

    return sorted(runs, key=lambda x: (x.exp_name, x.timestamp))


def _is_main_exp(name: str) -> bool:
    """Keep only core experiment names (drop _seqNNNN suffix variants)."""
    return name in {
        'exp_0_baseline', 'exp_1_deepseek_topk', 'exp_2_lightning_hybrid',
        'exp_3_dynamic_globals', 'exp_4_pbs_attn', 'exp_5_bigger_bird',
        'exp_6_deepseek_pbs', 'exp_7_layer_adaptive', 'exp_8_token_drop',
        'exp_9_attn_specul', 'exp_10_gqa_sparse', 'exp_11_nsa', 'exp_12_s2_hhst',
    }


def plot_training_trajectories(runs: List[TrainingRun], save_dir: str):
    """Plot per-epoch training curves for core experiments only."""

    colors = {
        'exp_0_baseline': '#7f7f7f',
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

    # Filter to main experiments and keep latest run per (exp_name, train_samples)
    by_key = {}
    for r in runs:
        if not _is_main_exp(r.exp_name):
            continue
        key = (r.exp_name, r.train_samples)
        if key not in by_key or r.timestamp > by_key[key].timestamp:
            by_key[key] = r

    # Group by sample size so we can plot one figure per sample size
    by_samples = {}
    for r in by_key.values():
        by_samples.setdefault(r.train_samples, []).append(r)

    for n_samples, sample_runs in sorted(by_samples.items()):
        sample_runs.sort(key=lambda r: r.exp_name)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Trajectories - {n_samples} samples (latest run per experiment)',
                     fontsize=14, fontweight='bold')

        # 1. F1 over epochs
        ax = axes[0, 0]
        for run in sample_runs:
            epochs = [p["epoch"] for p in run.trajectory if p["epoch"] is not None]
            f1s = [p["eval_f1"] for p in run.trajectory if p["eval_f1"] is not None]
            if epochs and f1s:
                short = run.exp_name.replace('exp_', '').replace('_', ' ')
                ax.plot(epochs, f1s, 'o-', label=short,
                        color=colors.get(run.exp_name, 'gray'), linewidth=2, markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score over Training')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='lower right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # 2. Accuracy over epochs
        ax = axes[0, 1]
        for run in sample_runs:
            epochs = [p["epoch"] for p in run.trajectory if p["epoch"] is not None]
            accs = [p["eval_accuracy"] for p in run.trajectory if p["eval_accuracy"] is not None]
            if epochs and accs:
                short = run.exp_name.replace('exp_', '').replace('_', ' ')
                ax.plot(epochs, accs, 's-', label=short,
                        color=colors.get(run.exp_name, 'gray'), linewidth=2, markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy over Training')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='lower right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # 3. Eval Loss
        ax = axes[1, 0]
        for run in sample_runs:
            epochs = [p["epoch"] for p in run.trajectory if p["epoch"] is not None]
            eval_losses = [p["eval_loss"] for p in run.trajectory if p["eval_loss"] is not None]
            if epochs and eval_losses:
                short = run.exp_name.replace('exp_', '').replace('_', ' ')
                ax.plot(epochs, eval_losses, 'o-', label=short,
                        color=colors.get(run.exp_name, 'gray'), linewidth=2, markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Eval Loss')
        ax.set_title('Eval Loss over Training')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # 4. F1 vs normalized training progress
        ax = axes[1, 1]
        for run in sample_runs:
            steps = [p["step"] for p in run.trajectory if p["step"] is not None]
            f1s = [p["eval_f1"] for p in run.trajectory if p["eval_f1"] is not None]
            if steps and f1s:
                total_steps = max(steps) if steps else 1
                norm_steps = [s / total_steps for s in steps]
                short = run.exp_name.replace('exp_', '').replace('_', ' ')
                ax.plot(norm_steps, f1s, 'o-', label=short,
                        color=colors.get(run.exp_name, 'gray'), linewidth=2, markersize=6)
        ax.set_xlabel('Training Progress (normalized steps)')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 vs Training Progress')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='lower right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"training_trajectories_{n_samples}samples.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path} ({len(sample_runs)} experiments)")

    # Also print a clean summary table (largest sample size only)
    max_samples = max(by_samples.keys())
    print("\n" + "="*90)
    print(f"TRAINING TRAJECTORY SUMMARY - {max_samples} samples (latest main experiment run)")
    print("="*90)
    for run in sorted(by_samples[max_samples], key=lambda x: x.exp_name):
        print(f"\n{run.exp_name} ({run.epochs} epochs):")
        print(f"{'Epoch':>6} {'Step':>8} {'Train Loss':>12} {'Eval Loss':>12} {'F1':>8} {'Acc':>8}")
        print("-"*70)
        for p in run.trajectory:
            epoch = p.get("epoch", "-")
            step = p.get("step", "-")
            tl = f"{p['train_loss']:.4f}" if p.get("train_loss") else "-"
            el = f"{p['eval_loss']:.4f}" if p.get("eval_loss") else "-"
            f1 = f"{p['eval_f1']:.3f}" if p.get("eval_f1") else "-"
            acc = f"{p['eval_accuracy']:.3f}" if p.get("eval_accuracy") else "-"
            print(f"{epoch:>6} {step:>8} {tl:>12} {el:>12} {f1:>8} {acc:>8}")
        print(f"Final F1: {run.final_f1:.3f}, Final Acc: {run.final_acc:.3f}")
    print("="*90)


def main():
    runs = load_all_runs()

    if not runs:
        print("No runs with trajectory data found.")
        print("Run experiments first with the updated runner (captures per-epoch metrics).")
        return

    print(f"Loaded {len(runs)} runs with trajectory data")
    plot_training_trajectories(runs, BENCHMARK_DIR)

if __name__ == "__main__":
    main()
