"""Compare rewiring modes: edge overlap, diversity, and routing overhead.

Runs on CPU with synthetic hidden states — no model or GPU needed.

    python experiments/graph_sparse/compare_rewiring.py
"""
import sys
import time

import torch

sys.path.insert(0, ".")

from experiments.graph_sparse.config import GraphConfig, SemanticConfig, RewireConfig
from experiments.graph_sparse.graph_state import GraphCoordinator


def run_mode(mode, num_layers=32, n=2048, rew_kw=None):
    torch.manual_seed(0)
    hidden = torch.randn(n, 256)
    rew_kw = rew_kw or {}
    cfg = GraphConfig(
        query_chunk_size=256, key_block_size=64, routed_blocks=4,
        routing_dim=64, seed=42,
        semantic=SemanticConfig(near_blocks=2, medium_blocks=1,
                                far_blocks=0, random_blocks=1),
        rewiring=RewireConfig(mode=mode, replace_probability=0.5, seed=42, **rew_kw),
    )
    coord = GraphCoordinator(cfg, num_heads=8, num_kv_heads=2,
                             head_dim=64, num_layers=num_layers)
    coord.begin_prefill()

    all_blocks = set()
    t0 = time.perf_counter()
    for layer in range(num_layers):
        idx = coord.build_layer_graph(layer, hidden)
        for g in range(idx.shape[1]):
            valid = idx[0, g][idx[0, g] >= 0]
            # Convert token indices back to block indices
            for v in valid.tolist():
                all_blocks.add(v // cfg.key_block_size)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    overlaps = coord.state.layer_edge_overlap
    return {
        "mode": mode,
        "mean_overlap": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "min_overlap": min(overlaps) if overlaps else 0.0,
        "max_overlap": max(overlaps) if overlaps else 0.0,
        "unique_blocks": len(all_blocks),
        "elapsed_ms": elapsed_ms,
    }


def main():
    modes = [
        ("none", {}),
        ("static", {}),
        ("random", {}),
        ("partial_random", {"replace_probability": 0.5}),
        ("semantic", {"replace_probability": 0.5}),
        ("semantic", {"replace_probability": 0.3}),
    ]

    print("=" * 80)
    print("Graph Rewiring Mode Comparison")
    print(f"  Sequence: 2048 tokens, 32 layers, chunk=256, block=64, routed_blocks=4")
    print("=" * 80)
    print()
    print(f"{'Mode':<25s} {'Mean Overlap':>12s} {'Min':>6s} {'Max':>6s} "
          f"{'Unique Blk':>10s} {'Time (ms)':>10s}")
    print("-" * 80)

    for mode, kw in modes:
        label = mode
        if mode in ("semantic", "partial_random") and "replace_probability" in kw:
            label = f"{mode} (rp={kw['replace_probability']})"
        r = run_mode(mode, rew_kw=kw)
        print(f"{label:<25s} {r['mean_overlap']:>12.3f} {r['min_overlap']:>6.3f} "
              f"{r['max_overlap']:>6.3f} {r['unique_blocks']:>10d} "
              f"{r['elapsed_ms']:>10.1f}")

    print()
    print("Overlap = fraction of block edges at layer L also present at layer L-1.")
    print("Unique blocks = distinct key blocks visited across all layers (head 0).")


if __name__ == "__main__":
    main()
