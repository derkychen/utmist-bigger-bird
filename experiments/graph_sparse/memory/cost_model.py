"""Configuration and cost model for memory-layout strategies.

MemoryLayoutConfig: all tunable parameters for the memory layout manager.
LayoutStats: measured overhead of a pack/unpack cycle.
measure_packing_cost: time and measure a pack+unpack round trip.
estimate_locality: estimate how well routing indices hit contiguous memory.
"""
import time
import torch
from dataclasses import dataclass, field
from typing import Optional

from .permutation import Permutation, identity_permutation
from .packing import pack_kv, unpack_kv, translate_indices


STRATEGIES = frozenset({"original", "sorted_indices", "semantic_clustered", "graph_clustered"})


@dataclass
class MemoryLayoutConfig:
    """Configuration for memory-layout optimization.

    Attributes:
        strategy: one of "original", "sorted_indices", "semantic_clustered",
            "graph_clustered".
        cluster_size: target tokens per cluster for clustering strategies.
        clustering_method: "semantic" or "graph" (used by semantic/graph_clustered).
        cluster_refresh_layers: recompute clusters every N layers (0 = once at start).
        pack_each_layer: if True, physically repack K/V each layer; if False,
            only recompute cluster assignments (cheaper, no data movement).
        graph_history_decay: exponential decay for co-access counts in graph clustering.
        reorder_threshold: only repack if estimated locality gain exceeds this fraction.
        seed: for deterministic clustering.
    """
    strategy: str = "original"
    cluster_size: int = 64
    clustering_method: str = "semantic"
    cluster_refresh_layers: int = 0
    pack_each_layer: bool = False
    graph_history_decay: float = 0.9
    reorder_threshold: float = 0.0
    seed: int = 42

    def validate(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy {self.strategy!r}, "
                f"expected one of {sorted(STRATEGIES)}")
        if self.cluster_size < 1:
            raise ValueError("cluster_size must be >= 1")
        if self.clustering_method not in ("semantic", "graph"):
            raise ValueError(
                f"Unknown clustering_method {self.clustering_method!r}")
        if self.cluster_refresh_layers < 0:
            raise ValueError("cluster_refresh_layers must be >= 0")
        if not (0.0 < self.graph_history_decay <= 1.0):
            raise ValueError("graph_history_decay must be in (0, 1]")
        if self.reorder_threshold < 0.0:
            raise ValueError("reorder_threshold must be >= 0")


@dataclass
class LayoutStats:
    """Measured overhead of a memory-layout operation."""
    strategy: str = "original"
    cluster_time_ms: float = 0.0
    pack_time_ms: float = 0.0
    unpack_time_ms: float = 0.0
    bytes_moved: int = 0
    peak_memory_bytes: int = 0
    locality_before: float = 0.0
    locality_after: float = 0.0
    num_clusters: int = 0


def measure_packing_cost(k: torch.Tensor, v: torch.Tensor,
                         perm: Permutation,
                         logical_indices: Optional[torch.Tensor] = None,
                         warmup: int = 1, repeats: int = 3) -> LayoutStats:
    """Time a full pack → translate → unpack cycle and measure data movement.

    Args:
        k: [num_kv_heads, N, head_dim] key cache.
        v: [num_kv_heads, N, head_dim] value cache.
        perm: the permutation to apply.
        logical_indices: optional [H, G, K] routing indices for locality measurement.
        warmup: warmup iterations (not timed).
        repeats: timed iterations (minimum time reported).

    Returns:
        LayoutStats with timing and data-movement metrics.
    """
    stats = LayoutStats()
    n = perm.n
    bytes_per_tensor = k.numel() * k.element_size()
    stats.bytes_moved = bytes_per_tensor * 2 * 2

    if logical_indices is not None:
        stats.locality_before = estimate_locality(logical_indices, n)

    for _ in range(warmup):
        kp, vp = pack_kv(k, v, perm)
        _, _ = unpack_kv(kp, vp, perm)

    pack_times, unpack_times = [], []
    for _ in range(repeats):
        if k.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        kp, vp = pack_kv(k, v, perm)
        if k.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        _, _ = unpack_kv(kp, vp, perm)
        if k.is_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        pack_times.append((t1 - t0) * 1000)
        unpack_times.append((t2 - t1) * 1000)

    stats.pack_time_ms = min(pack_times)
    stats.unpack_time_ms = min(unpack_times)

    if logical_indices is not None:
        phys_indices = translate_indices(logical_indices, perm)
        stats.locality_after = estimate_locality(phys_indices, n)

    return stats


def estimate_locality(indices: torch.Tensor, seq_len: int,
                      neighborhood: int = 64) -> float:
    """Estimate what fraction of routed token accesses hit contiguous neighborhoods.

    For each query group's routed indices, measures how many consecutive pairs
    of sorted physical positions are within ``neighborhood`` of each other.
    Higher values mean the routing pattern's memory accesses are more contiguous.

    Args:
        indices: [H, G, K] int32 token positions (-1 = padding).
        seq_len: total sequence length.
        neighborhood: two positions are "local" if |p1 - p2| <= neighborhood.

    Returns:
        Fraction of consecutive routed-position pairs that are within neighborhood.
    """
    H, G, K = indices.shape
    total_pairs = 0
    local_pairs = 0
    flat = indices.reshape(-1, K)
    for row_idx in range(flat.shape[0]):
        row = flat[row_idx]
        valid = row[row >= 0].sort().values
        if valid.numel() < 2:
            continue
        diffs = (valid[1:] - valid[:-1]).abs()
        total_pairs += diffs.numel()
        local_pairs += (diffs <= neighborhood).sum().item()
    return local_pairs / total_pairs if total_pairs > 0 else 1.0


def build_co_access_matrix(route_history: list, seq_len: int,
                           decay: float = 0.9,
                           block_size: int = 64) -> torch.Tensor:
    """Build a block-level co-access count matrix from routing history.

    Uses block-level granularity (O(B^2) where B = seq_len/block_size) instead
    of token-level (O(N^2)) to scale to 128K+ sequences.  At block_size=64 and
    seq_len=131072, B=2048 so the matrix is 2048x2048 (~16 MB) instead of
    131072x131072 (~64 GB).

    Args:
        route_history: list of [H, G, K] int32 routing index tensors,
            one per layer that has been routed.
        seq_len: total sequence length N.
        decay: older layers' contributions are decayed by decay^(age).
        block_size: granularity for co-access counting.

    Returns:
        [B, B] float32 co-access matrix where B = ceil(seq_len / block_size).
        co_access[i, j] counts how often blocks i and j had tokens routed to
        the same query group.
    """
    num_blocks = -(-seq_len // block_size)
    co_access = torch.zeros(num_blocks, num_blocks)
    for age, indices in enumerate(reversed(route_history)):
        weight = decay ** age
        H, G, K = indices.shape
        for g in range(G):
            for h in range(H):
                valid = indices[h, g][indices[h, g] >= 0].long()
                if valid.numel() < 2:
                    continue
                blocks = valid // block_size
                unique_blocks = blocks.unique()
                if unique_blocks.numel() < 2:
                    continue
                bi = unique_blocks.unsqueeze(1).expand(-1, unique_blocks.numel()).reshape(-1)
                bj = unique_blocks.unsqueeze(0).expand(unique_blocks.numel(), -1).reshape(-1)
                co_access[bi, bj] += weight
    co_access.fill_diagonal_(0)
    return co_access
