"""Memory-layout strategies for graph-sparse attention.

Provides configurable K/V reordering to improve GPU memory locality for
sparse attention patterns without modifying the Triton kernel.
"""
from .clusters import (
    cluster_tokens,
    semantic_cluster,
    graph_cluster,
)
from .permutation import (
    Permutation,
    build_permutation,
    identity_permutation,
)
from .packing import (
    pack_kv,
    unpack_kv,
    translate_indices,
)
from .cost_model import (
    MemoryLayoutConfig,
    LayoutStats,
    measure_packing_cost,
    estimate_locality,
)

__all__ = [
    "cluster_tokens",
    "semantic_cluster",
    "graph_cluster",
    "Permutation",
    "build_permutation",
    "identity_permutation",
    "pack_kv",
    "unpack_kv",
    "translate_indices",
    "MemoryLayoutConfig",
    "LayoutStats",
    "measure_packing_cost",
    "estimate_locality",
]
