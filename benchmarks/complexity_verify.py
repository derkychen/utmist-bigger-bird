#!/usr/bin/env python3
"""
Empirical complexity verification.
Micro-benchmarks the attention forward pass directly to verify O(n) vs O(n²).

Usage:
  python benchmarks/complexity_verify.py --exp 1,2,3,4
  python benchmarks/complexity_verify.py --exp 0 --seq 128,256,512,1024,2048,4096
"""

import sys
import os
import argparse
import json
import time
import gc
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.bart.modeling_bart import BartAttention

from exp_1_deepseek_topk.model import DeepSeekTopKAttention
from exp_2_lightning_hybrid.model import LightningHybridAttention
from exp_3_dynamic_globals.model import DynamicGlobalAttention
from exp_4_pbs_attn.model import PBSAttention
from exp_5_bigger_bird.model import BiggerBirdAttention
from exp_6_deepseek_pbs.model import DeepSeekPBSAttention
from exp_7_layer_adaptive.model import LayerAdaptiveAttention
from exp_9_attn_specul.model import AttnSpeculAttention
from exp_10_gqa_sparse.model import GQASparseAttention
# Note: exp_8 (token_drop) doesn't have a standalone attention class — it modifies
# the encoder loop, so it's excluded from the per-layer micro-benchmark.


class DummyBartAttention(BartAttention):
    """Minimal BART attention for timing baseline."""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__(embed_dim, num_heads, dropout, is_decoder=False, bias=True)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, **kwargs):
        bsz, tgt_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states) * (self.head_dim ** -0.5)
        key = self._shape(self.k_proj(hidden_states), -1, bsz)
        value = self._shape(self.v_proj(hidden_states), -1, bsz)
        Q = self._shape(query, tgt_len, bsz).reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        K = key.reshape(bsz * self.num_heads, -1, self.head_dim)
        V = value.reshape(bsz * self.num_heads, -1, self.head_dim)
        scores = torch.bmm(Q, K.transpose(1, 2))
        attn = nn.functional.softmax(scores, dim=-1)
        out = torch.bmm(attn, V)
        out = out.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        return self.out_proj(out), None


def time_attention(module, batch_size, seq_len, device, n_trials=20, warmup=5):
    """Time a single attention forward pass. Returns avg ms."""
    hidden = torch.randn(batch_size, seq_len, 768, device=device)
    module = module.to(device).eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(hidden)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(n_trials):
                _ = module(hidden)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_trials  # ms per forward
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_trials):
                _ = module(hidden)
        return (time.perf_counter() - t0) * 1000 / n_trials


def create_module(exp_num, device):
    """Create an attention module for the given experiment."""
    base = DummyBartAttention(embed_dim=768, num_heads=12, dropout=0.1).to(device)
    if exp_num == 0:
        return base
    elif exp_num == 1:
        return DeepSeekTopKAttention(base, top_k=64, low_rank_dim=16)
    elif exp_num == 2:
        return LightningHybridAttention(base, block_size=128)
    elif exp_num == 3:
        return DynamicGlobalAttention(base, window_size=64, num_globals=16)
    elif exp_num == 4:
        return PBSAttention(base, block_size=64, num_blocks=2)
    elif exp_num == 5:
        return BiggerBirdAttention(base, window_size=64, local_k=32,
                                   num_globals=16, num_teleports=8,
                                   diversity_lambda=0.3, teleport_bias=0.5)
    elif exp_num == 6:
        return DeepSeekPBSAttention(base, top_k=64, low_rank_dim=16,
                                    block_size=32, num_blocks=4)
    elif exp_num == 7:
        return LayerAdaptiveAttention(base, top_k=64, low_rank_dim=16, layer_idx=6)
    elif exp_num == 8:
        # Token drop is encoder-level — skip micro-benchmark
        return base
    elif exp_num == 9:
        return AttnSpeculAttention(base, window_size=64, num_anchors=4,
                                   verify=False, verify_kl_weight=0.0)
    elif exp_num == 10:
        return GQASparseAttention(base, kv_groups=4, top_k=64, low_rank_dim=16)
    else:
        raise ValueError(f"Unknown exp_num: {exp_num}")


def main():
    parser = argparse.ArgumentParser(description="Empirical complexity verification")
    parser.add_argument("--exp", type=str, default="0,1,2,3,4",
                        help="Comma-separated experiment numbers")
    parser.add_argument("--seq", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size for timing")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of timing trials")
    args = parser.parse_args()

    exp_nums = [int(x.strip()) for x in args.exp.split(",")]
    seq_lengths = [int(x.strip()) for x in args.seq.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Experiments: {exp_nums}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Batch size: {args.batch}, Trials: {args.trials}")

    results = []
    for exp_num in exp_nums:
        exp_name = f"exp_{exp_num}_baseline" if exp_num == 0 else f"exp_{exp_num}"
        print(f"\n--- Timing {exp_name} ---")
        for seq_len in seq_lengths:
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            try:
                module = create_module(exp_num, device)
                ms = time_attention(module, args.batch, seq_len, device, n_trials=args.trials)
                print(f"  seq={seq_len:>5} | {ms:>8.3f} ms")
                results.append({
                    "exp_num": exp_num,
                    "exp_name": exp_name,
                    "seq_length": seq_len,
                    "batch_size": args.batch,
                    "time_ms": ms,
                    "oom": False,
                })
            except torch.cuda.OutOfMemoryError:
                print(f"  seq={seq_len:>5} | OOM")
                results.append({
                    "exp_num": exp_num,
                    "exp_name": exp_name,
                    "seq_length": seq_len,
                    "batch_size": args.batch,
                    "time_ms": None,
                    "oom": True,
                })
            except Exception as e:
                print(f"  seq={seq_len:>5} | ERROR: {e}")
                results.append({
                    "exp_num": exp_num,
                    "exp_name": exp_name,
                    "seq_length": seq_len,
                    "batch_size": args.batch,
                    "time_ms": None,
                    "oom": False,
                    "error": str(e),
                })

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "complexity_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": str(device),
                "batch_size": args.batch,
                "trials": args.trials,
            },
            "results": results,
        }, f, indent=2)

    print(f"\nSaved results to: {out_path}")

    # Compute log-log slopes (empirical complexity)
    print("\nEMPIRICAL COMPLEXITY (log-log slope ≈ 1 means O(n), ≈ 2 means O(n²))")
    print("-" * 60)
    from math import log
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

    # Generate log-log plot
    try:
        import matplotlib.pyplot as plt
        import numpy as np

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

        # Reference lines
        if len(seq_lengths) >= 2:
            x_ref = np.array([min(seq_lengths), max(seq_lengths)])
            # O(n) reference
            ax.loglog(x_ref, x_ref * (1 / x_ref[0]), '--', color='gray', alpha=0.5, label='O(n) reference')
            # O(n²) reference
            ax.loglog(x_ref, (x_ref ** 2) / (x_ref[0] ** 2), '--', color='black', alpha=0.5, label='O(n²) reference')

        ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
        ax.set_ylabel("Time per Forward Pass (ms)", fontsize=12)
        ax.set_title("Empirical Time Complexity (Log-Log Plot)\nslope ≈ 1 → O(n),  slope ≈ 2 → O(n²)", fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, which="both", ls="-", alpha=0.2)

        plot_path = os.path.join(os.path.dirname(__file__), "complexity_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved complexity plot to: {plot_path}")

        # ---- THEORETICAL SOFTMAX COMPARISONS PLOT (the metric that actually matters) ----
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

        # Config params per experiment
        cfg_map = {
            0: ("baseline", None, "n"),
            1: ("deepseek_topk", 64, "k"),
            2: ("lightning_hybrid", 128, "block_size"),
            3: ("dynamic_globals", 80, "window+globals"),
            4: ("pbs_attn", 128, "blocks*bsize"),
            5: ("bigger_bird", 56, "k+g+t"),   # 32+16+8
            6: ("deepseek_pbs", 64, "topk_in_blocks"),
            7: ("layer_adaptive", 96, "avg_k"), # (192+64+32)/3 ≈ 96
            9: ("attn_specul", 68, "window+anchors"),  # 64+4
            10: ("gqa_sparse", 64, "topk"),
        }

        n_layers = 12
        n_heads = 12
        baseline = None

        for i, exp_num in enumerate(exp_nums):
            if exp_num not in cfg_map:
                continue
            name, k, label = cfg_map[exp_num]
            xs_th = np.array(seq_lengths)
            # Total softmax key positions evaluated across all layers/heads/batch
            # Baseline: n_layers * n_heads * batch * seq_len * seq_len
            # Sparse:   n_layers * n_heads * batch * seq_len * k
            if k is None:
                ys_th = n_layers * n_heads * args.batch * xs_th * xs_th
                baseline = ys_th
            else:
                ys_th = n_layers * n_heads * args.batch * xs_th * k

            ax2a.plot(xs_th, ys_th / 1e6, marker='o', label=f"{name} ({label}={k or 'n'})",
                     color=colors[i % len(colors)], linewidth=2)

        ax2a.set_xlabel("Sequence Length (tokens)", fontsize=12)
        ax2a.set_ylabel("Total Softmax Comparisons (millions)", fontsize=12)
        ax2a.set_title("Theoretical Softmax Comparisons\nper Forward Pass", fontsize=13, fontweight='bold')
        ax2a.legend(fontsize=8)
        ax2a.grid(True, alpha=0.2)

        # Normalized reduction vs baseline
        if baseline is not None:
            for i, exp_num in enumerate(exp_nums):
                if exp_num == 0 or exp_num not in cfg_map:
                    continue
                name, k, label = cfg_map[exp_num]
                xs_th = np.array(seq_lengths)
                ys_th = n_layers * n_heads * args.batch * xs_th * k
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
    except ImportError:
        print("\nInstall matplotlib to generate the complexity plot: pip install matplotlib")
    except Exception as e:
        print(f"\nPlot generation failed: {e}")


if __name__ == "__main__":
    main()
