#!/usr/bin/env python3
"""Regenerate complexity plots from merged results JSON."""
import json
import os
from math import log

import matplotlib.pyplot as plt
import numpy as np

with open(os.path.join(os.path.dirname(__file__), "complexity_results.json")) as f:
    data = json.load(f)

results = data["results"]

# Determine all unique experiments and sequence lengths
exp_nums = sorted(set(r["exp_num"] for r in results))
seq_lengths = sorted(set(r["seq_length"] for r in results))

# Compute log-log slopes
print("\nEMPIRICAL COMPLEXITY (log-log slope ≈ 1 means O(n), ≈ 2 means O(n²))")
print("-" * 60)
slope_map = {}
for exp_num in exp_nums:
    exp_name = f"exp_{exp_num}_baseline" if exp_num == 0 else f"exp_{exp_num}"
    pts = [(r["seq_length"], r["time_ms"]) for r in results
           if r["exp_num"] == exp_num and r["time_ms"] is not None and not r.get("oom", False)]
    pts = sorted(set(pts))
    if len(pts) >= 2:
        slopes = []
        for i in range(1, len(pts)):
            s1, t1 = pts[i-1]
            s2, t2 = pts[i]
            slope = log(t2 / t1) / log(s2 / s1)
            slopes.append(slope)
        avg_slope = sum(slopes) / len(slopes)
        slope_map[exp_name] = avg_slope
        print(f"  {exp_name:<20} avg slope = {avg_slope:.3f}")
    else:
        print(f"  {exp_name:<20} insufficient data")

# ---- EMPIRICAL PLOT ----
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.tab10.colors

for i, exp_num in enumerate(exp_nums):
    exp_name = f"exp_{exp_num}_baseline" if exp_num == 0 else f"exp_{exp_num}"
    pts = [(r["seq_length"], r["time_ms"]) for r in results
           if r["exp_num"] == exp_num and r["time_ms"] is not None and not r.get("oom", False)]
    pts = sorted(set(pts))
    if len(pts) < 2:
        continue
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    slope_label = slope_map.get(exp_name, 0)
    label = f"{exp_name.replace('_', ' ')} (slope={slope_label:.2f})"
    ax.loglog(xs, ys, marker='o', label=label, color=colors[i % len(colors)], linewidth=2)

# OOM markers
for i, exp_num in enumerate(exp_nums):
    exp_name = f"exp_{exp_num}_baseline" if exp_num == 0 else f"exp_{exp_num}"
    oom_pts = [(r["seq_length"], 1.0) for r in results
               if r["exp_num"] == exp_num and r.get("oom", False)]
    if oom_pts:
        ax.scatter([p[0] for p in oom_pts], [ax.get_ylim()[0] * 5] * len(oom_pts),
                   marker='x', s=100, color=colors[i % len(colors)], linewidths=3,
                   label=f"{exp_name.replace('_', ' ')} OOM")

x_ref = np.array([min(seq_lengths), max(seq_lengths)])
ax.loglog(x_ref, x_ref * (1 / x_ref[0]), '--', color='gray', alpha=0.5, label='O(n) reference')
ax.loglog(x_ref, (x_ref ** 2) / (x_ref[0] ** 2), '--', color='black', alpha=0.5, label='O(n²) reference')

ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
ax.set_ylabel("Time per Forward Pass (ms)", fontsize=12)
ax.set_title("Empirical Time Complexity (Log-Log Plot)\nslope ≈ 1 → O(n),  slope ≈ 2 → O(n²)", fontsize=14, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, which="both", ls="-", alpha=0.2)

plot_path = os.path.join(os.path.dirname(__file__), "complexity_plot.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved complexity plot to: {plot_path}")

# ---- THEORETICAL PLOT ----
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

cfg_map = {
    0: ("baseline", None, "n"),
    1: ("deepseek_topk", 64, "k"),
    2: ("lightning_hybrid", 128, "block_size"),
    3: ("dynamic_globals", 80, "window+globals"),
    4: ("pbs_attn", 128, "blocks*bsize"),
    5: ("bigger_bird", 56, "k+g+t"),
    6: ("deepseek_pbs", 64, "topk_in_blocks"),
    7: ("layer_adaptive", 96, "avg_k"),
    9: ("attn_specul", 68, "window+anchors"),
    10: ("gqa_sparse", 64, "topk"),
}

n_layers = 12
n_heads = 12
batch = 4  # theoretical baseline
baseline = None

for i, exp_num in enumerate(exp_nums):
    if exp_num not in cfg_map:
        continue
    name, k, label = cfg_map[exp_num]
    xs_th = np.array(seq_lengths)
    if k is None:
        ys_th = n_layers * n_heads * batch * xs_th * xs_th
        baseline = ys_th
    else:
        ys_th = n_layers * n_heads * batch * xs_th * k
    ax2a.plot(xs_th, ys_th / 1e6, marker='o', label=f"{name} ({label}={k or 'n'})",
             color=colors[i % len(colors)], linewidth=2)

ax2a.set_xlabel("Sequence Length (tokens)", fontsize=12)
ax2a.set_ylabel("Total Softmax Comparisons (millions)", fontsize=12)
ax2a.set_title("Theoretical Softmax Comparisons\nper Forward Pass", fontsize=13, fontweight='bold')
ax2a.legend(fontsize=8)
ax2a.grid(True, alpha=0.2)

if baseline is not None:
    for i, exp_num in enumerate(exp_nums):
        if exp_num == 0 or exp_num not in cfg_map:
            continue
        name, k, label = cfg_map[exp_num]
        xs_th = np.array(seq_lengths)
        ys_th = n_layers * n_heads * batch * xs_th * k
        reduction = (1 - ys_th / baseline) * 100
        ax2b.plot(xs_th, reduction, marker='o', label=f"{name}",
                 color=colors[i % len(colors)], linewidth=2)

ax2b.set_xlabel("Sequence Length (tokens)", fontsize=12)
ax2b.set_ylabel("Reduction vs Baseline (%)", fontsize=12)
ax2b.set_title("Theoretical % Reduction\nin Softmax Comparisons", fontsize=13, fontweight='bold')
ax2b.legend(fontsize=8)
ax2b.grid(True, alpha=0.2)
ax2b.set_ylim(0, 100)

plot_path2 = os.path.join(os.path.dirname(__file__), "complexity_theoretical.png")
plt.tight_layout()
plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved theoretical plot to: {plot_path2}")
