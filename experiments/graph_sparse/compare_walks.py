"""Compare walk modes: path coverage, diversity, and routing overhead.

Runs on CPU with synthetic hidden states — no model or GPU needed.

    python experiments/graph_sparse/compare_walks.py
"""
import sys
import time

import torch

sys.path.insert(0, ".")

from experiments.graph_sparse.config import (
    GraphConfig, SemanticConfig, RewireConfig, WalkConfig,
)
from experiments.graph_sparse.graph_state import GraphCoordinator


def run_config(label, num_layers=32, n=2048, walk_kw=None, rew_kw=None):
    torch.manual_seed(0)
    hidden = torch.randn(n, 256)
    walk_kw = walk_kw or {}
    rew_kw = rew_kw or {}
    cfg = GraphConfig(
        query_chunk_size=256, key_block_size=64, routed_blocks=4,
        routing_dim=64, seed=42,
        semantic=SemanticConfig(near_blocks=2, medium_blocks=1,
                                far_blocks=0, random_blocks=1),
        rewiring=RewireConfig(mode="partial_semantic", replace_probability=0.5,
                              seed=42, **rew_kw),
        walks=WalkConfig(**walk_kw),
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
            for v in valid.tolist():
                all_blocks.add(v // cfg.key_block_size)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    overlaps = coord.state.layer_edge_overlap
    diags = coord.state.walk_diagnostics_per_layer
    last_diag = diags[-1] if diags else {}
    return {
        "label": label,
        "mean_overlap": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "unique_blocks": len(all_blocks),
        "active_walks": last_diag.get("active_count", 0),
        "completed_walks": last_diag.get("completed_count", 0),
        "mean_path_len": last_diag.get("mean_path_length", 0.0),
        "walk_coverage": last_diag.get("unique_blocks_visited", 0),
        "elapsed_ms": elapsed_ms,
    }


def main():
    configs = [
        ("No walks (baseline)", {}),
        ("Random walks", {"enabled": True, "walk_type": "random",
         "walk_length": 5, "walks_per_start": 1, "reserved_edge_budget": 1}),
        ("Random walks ×2", {"enabled": True, "walk_type": "random",
         "walk_length": 5, "walks_per_start": 2, "reserved_edge_budget": 2}),
        ("Semantic walks", {"enabled": True, "walk_type": "semantic",
         "walk_length": 5, "walks_per_start": 1, "reserved_edge_budget": 1}),
        ("Novelty walks", {"enabled": True, "walk_type": "novelty",
         "walk_length": 5, "walks_per_start": 1, "reserved_edge_budget": 1}),
        ("Long random (len=10)", {"enabled": True, "walk_type": "random",
         "walk_length": 10, "walks_per_start": 1, "reserved_edge_budget": 1,
         "continuation_probability": 0.9}),
        ("Random + restart", {"enabled": True, "walk_type": "random",
         "walk_length": 5, "walks_per_start": 1, "reserved_edge_budget": 1,
         "restart_probability": 0.3}),
    ]

    print("=" * 100)
    print("Walk Mode Comparison")
    print(f"  Sequence: 2048 tokens, 32 layers, chunk=256, block=64, routed_blocks=4")
    print("=" * 100)
    print()
    print(f"{'Config':<25s} {'Overlap':>8s} {'Uniq Blk':>8s} {'Active':>7s} "
          f"{'Compl':>6s} {'Path Len':>9s} {'Walk Cov':>9s} {'Time ms':>8s}")
    print("-" * 100)

    for label, walk_kw in configs:
        r = run_config(label, walk_kw=walk_kw)
        print(f"{r['label']:<25s} {r['mean_overlap']:>8.3f} {r['unique_blocks']:>8d} "
              f"{r['active_walks']:>7d} {r['completed_walks']:>6d} "
              f"{r['mean_path_len']:>9.2f} {r['walk_coverage']:>9d} "
              f"{r['elapsed_ms']:>8.1f}")

    print()
    print("Overlap = fraction of block edges at layer L also present at L-1.")
    print("Walk Cov = unique blocks visited by any walk across all layers.")
    print("Active/Compl = walk counts at the final layer.")


if __name__ == "__main__":
    main()
