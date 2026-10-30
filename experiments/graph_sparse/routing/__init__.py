"""Graph-based sparse attention routing.

Public API for the routing subsystem. Callers should use `route_layer` for the
full pipeline (summarize → score → classify → sample → expand → deduplicate),
or import individual functions for testing.
"""
from .candidates import (
    build_block_summaries,
    build_query_summaries,
    causal_block_mask,
    compute_similarity,
    find_candidates,
    project_for_routing,
)
from .sample import (
    deduplicate_and_sort,
    deduplicate_and_sort_batched,
    expand_blocks_to_indices,
    sample_semantic_edges,
    sample_semantic_edges_batched,
)
from .semantic import (
    classify_semantic_distance,
    classify_semantic_distance_batched,
)
from .rewire import rewire_edges, compute_edge_overlap
from .walks import (
    Walk,
    advance_walks,
    inject_walk_edges,
    spawn_walks,
    walk_diagnostics,
    random_walk,
    semantic_walk,
    novelty_walk,
)

__all__ = [
    "build_block_summaries",
    "build_query_summaries",
    "causal_block_mask",
    "compute_similarity",
    "find_candidates",
    "project_for_routing",
    "classify_semantic_distance",
    "classify_semantic_distance_batched",
    "sample_semantic_edges",
    "sample_semantic_edges_batched",
    "expand_blocks_to_indices",
    "deduplicate_and_sort",
    "deduplicate_and_sort_batched",
    "rewire_edges",
    "compute_edge_overlap",
    "Walk",
    "advance_walks",
    "inject_walk_edges",
    "spawn_walks",
    "walk_diagnostics",
    "random_walk",
    "semantic_walk",
    "novelty_walk",
]
