"""Benchmark memory-layout strategies for graph-sparse attention.

Compares clustering cost, packing time, bytes moved, peak memory,
and estimated neighborhood locality across all four strategies.

Runs on CPU with synthetic data — no model or GPU needed.

    python experiments/graph_sparse/bench_memory_layout.py
"""
import sys
import time

import torch

sys.path.insert(0, ".")

from experiments.graph_sparse.config import (
    GraphConfig, SemanticConfig, RewireConfig, MemoryLayoutConfig,
)
from experiments.graph_sparse.graph_state import GraphCoordinator
from experiments.graph_sparse.memory.clusters import (
    cluster_tokens, semantic_cluster, graph_cluster,
)
from experiments.graph_sparse.memory.permutation import (
    build_permutation, identity_permutation,
)
from experiments.graph_sparse.memory.packing import pack_kv, unpack_kv, translate_indices
from experiments.graph_sparse.memory.cost_model import (
    measure_packing_cost, estimate_locality, build_co_access_matrix,
)


def run_benchmark(strategy, n=4096, d=128, head_dim=64, kv_heads=8,
                  num_heads=32, num_layers=4, cluster_size=64, seed=42):
    """Run a single strategy benchmark and return metrics."""
    torch.manual_seed(seed)
    hidden = torch.randn(n, d)
    k = torch.randn(kv_heads, n, head_dim)
    v = torch.randn(kv_heads, n, head_dim)

    cfg = GraphConfig(
        query_chunk_size=512, key_block_size=64, routed_blocks=4,
        routing_dim=64, seed=seed,
        semantic=SemanticConfig(near_blocks=2, medium_blocks=1,
                                far_blocks=0, random_blocks=1),
        rewiring=RewireConfig(mode="partial_semantic", seed=seed),
    )
    coord = GraphCoordinator(cfg, num_heads=num_heads, num_kv_heads=kv_heads,
                             head_dim=head_dim, num_layers=num_layers)
    coord.begin_prefill()

    route_history = []
    for layer in range(num_layers):
        idx = coord.build_layer_graph(layer, hidden)
        route_history.append(idx)

    indices = route_history[-1]
    locality_original = estimate_locality(indices, n)

    # --- Strategy-specific clustering ---
    t_cluster_start = time.perf_counter()

    if strategy == "original":
        perm = identity_permutation(n)
    elif strategy == "sorted_indices":
        assignments = semantic_cluster(hidden, cluster_size, seed=seed)
        perm = build_permutation(assignments, cluster_size)
    elif strategy == "semantic_clustered":
        assignments = semantic_cluster(hidden, cluster_size, seed=seed)
        perm = build_permutation(assignments, cluster_size)
    elif strategy == "graph_clustered":
        co_access = build_co_access_matrix(route_history, n, decay=0.9)
        assignments = graph_cluster(co_access, cluster_size, seed=seed)
        perm = build_permutation(assignments, cluster_size)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    t_cluster = (time.perf_counter() - t_cluster_start) * 1000

    # --- Pack/Unpack timing ---
    stats = measure_packing_cost(k, v, perm, logical_indices=indices,
                                  warmup=1, repeats=3)
    stats.cluster_time_ms = t_cluster
    stats.strategy = strategy
    stats.locality_before = locality_original

    # --- Verify correctness ---
    if strategy != "original":
        kp, vp = pack_kv(k, v, perm)
        kr, vr = unpack_kv(kp, vp, perm)
        assert torch.equal(kr, k), f"{strategy}: K not restored"
        assert torch.equal(vr, v), f"{strategy}: V not restored"

    return stats


def main():
    configs = [
        ("original", 64),
        ("sorted_indices", 32),
        ("sorted_indices", 64),
        ("sorted_indices", 128),
        ("semantic_clustered", 32),
        ("semantic_clustered", 64),
        ("semantic_clustered", 128),
        ("graph_clustered", 64),
    ]

    n = 4096
    print("=" * 110)
    print("Memory-Layout Strategy Benchmark")
    print(f"  Seq: {n}, KV heads: 8, head_dim: 64, 4 routing layers, 4 routed blocks")
    print("=" * 110)
    print()
    print(f"{'Strategy':<25s} {'Clust sz':>8s} {'Cluster ms':>10s} "
          f"{'Pack ms':>8s} {'Unpack ms':>9s} {'Bytes moved':>12s} "
          f"{'Loc before':>10s} {'Loc after':>10s} {'Improvement':>11s}")
    print("-" * 110)

    for strategy, csz in configs:
        stats = run_benchmark(strategy, n=n, cluster_size=csz)
        improvement = stats.locality_after - stats.locality_before
        bytes_str = f"{stats.bytes_moved / 1024:.0f} KB"
        print(f"{strategy:<25s} {csz:>8d} {stats.cluster_time_ms:>10.2f} "
              f"{stats.pack_time_ms:>8.2f} {stats.unpack_time_ms:>9.2f} "
              f"{bytes_str:>12s} {stats.locality_before:>10.3f} "
              f"{stats.locality_after:>10.3f} {improvement:>+11.3f}")

    print()
    print("Locality = fraction of consecutive routed-position pairs within neighborhood=64.")
    print("'sorted_indices' only translates indices (no K/V data movement).")
    print("'semantic/graph_clustered' physically reorder K/V cache.")

    # --- Scaling test ---
    print()
    print("=" * 60)
    print("Scaling: cluster time vs sequence length")
    print("=" * 60)
    print(f"{'N':>8s} {'Semantic ms':>12s} {'Graph ms':>10s}")
    print("-" * 40)
    for n_scale in [512, 1024, 2048, 4096]:
        torch.manual_seed(42)
        hidden = torch.randn(n_scale, 128)
        t0 = time.perf_counter()
        semantic_cluster(hidden, 64, seed=42)
        t_sem = (time.perf_counter() - t0) * 1000

        co = torch.rand(n_scale, n_scale)
        t0 = time.perf_counter()
        graph_cluster(co, 64, seed=42)
        t_graph = (time.perf_counter() - t0) * 1000
        print(f"{n_scale:>8d} {t_sem:>12.2f} {t_graph:>10.2f}")


if __name__ == "__main__":
    main()
