"""Configuration for graph-aware sparse attention."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

MEMORY_STRATEGIES = frozenset({
    "original", "sorted_indices", "semantic_clustered", "graph_clustered",
})


@dataclass
class SemanticConfig:
    near_quantile: Tuple[float, float] = (0.0, 0.20)
    medium_quantile: Tuple[float, float] = (0.20, 0.70)
    far_quantile: Tuple[float, float] = (0.70, 1.0)
    near_blocks: int = 3
    medium_blocks: int = 2
    far_blocks: int = 2
    random_blocks: int = 1
    sampling_temperature: float = 0.0
    candidate_count: int = 0


@dataclass
class RewireConfig:
    mode: str = "partial_semantic"
    interval_layers: int = 1
    replace_probability: float = 0.5
    preserve_edge_fraction: float = 0.0
    sampling_temperature: float = 0.0
    distance_schedule: Optional[List[dict]] = None
    preserve_walk_edges: bool = True
    seed: int = 42


@dataclass
class WalkConfig:
    enabled: bool = False
    walk_type: str = "random"
    walk_length: int = 3
    walks_per_start: int = 1
    continuation_probability: float = 0.8
    restart_probability: float = 0.1
    avoid_revisits: bool = True
    reserved_edge_budget: int = 1
    semantic_distance_schedule: Optional[List[str]] = None
    seed: int = 42


@dataclass
class MemoryLayoutConfig:
    strategy: str = "original"
    cluster_size: int = 64
    clustering_method: str = "semantic"
    cluster_refresh_layers: int = 0
    pack_each_layer: bool = False
    graph_history_decay: float = 0.9
    reorder_threshold: float = 0.0
    seed: int = 42


@dataclass
class GraphConfig:
    mode: str = "strict_causal"
    query_chunk_size: int = 512
    key_block_size: int = 64
    routed_blocks: int = 8
    routing_dim: int = 64
    routing_representation: str = "hidden_states"
    seed: int = 42
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    rewiring: RewireConfig = field(default_factory=RewireConfig)
    walks: WalkConfig = field(default_factory=WalkConfig)
    memory_layout: MemoryLayoutConfig = field(default_factory=MemoryLayoutConfig)

    @property
    def routed_keys(self) -> int:
        return self.routed_blocks * self.key_block_size

    @property
    def total_semantic_blocks(self) -> int:
        s = self.semantic
        return s.near_blocks + s.medium_blocks + s.far_blocks + s.random_blocks

    def validate(self):
        """Check configuration consistency. Raises ValueError on invalid configs."""
        s = self.semantic
        total = self.total_semantic_blocks
        if total != self.routed_blocks:
            raise ValueError(
                f"Semantic block budget ({s.near_blocks}+{s.medium_blocks}"
                f"+{s.far_blocks}+{s.random_blocks}={total}) "
                f"!= routed_blocks ({self.routed_blocks})")
        if self.query_chunk_size < 1 or self.key_block_size < 1:
            raise ValueError("query_chunk_size and key_block_size must be positive")
        if self.routing_dim < 1:
            raise ValueError("routing_dim must be positive")
        if self.mode not in ("strict_causal", "offline"):
            raise ValueError(f"Unknown mode {self.mode!r}, expected 'strict_causal' or 'offline'")
        for name, (lo, hi) in [("near", s.near_quantile), ("medium", s.medium_quantile),
                                ("far", s.far_quantile)]:
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(f"{name}_quantile ({lo}, {hi}) must satisfy 0 <= lo < hi <= 1")
        if s.sampling_temperature < 0:
            raise ValueError("sampling_temperature must be >= 0")
        if s.candidate_count < 0:
            raise ValueError("candidate_count must be >= 0")
        REWIRE_MODES = {"none", "static", "random", "partial_random",
                         "semantic", "partial_semantic", "distance_schedule"}
        rew = self.rewiring
        if rew.mode not in REWIRE_MODES:
            raise ValueError(
                f"Unknown rewiring mode {rew.mode!r}, "
                f"expected one of {sorted(REWIRE_MODES)}")
        if not (0.0 <= rew.replace_probability <= 1.0):
            raise ValueError("replace_probability must be in [0, 1]")
        if not (0.0 <= rew.preserve_edge_fraction <= 1.0):
            raise ValueError("preserve_edge_fraction must be in [0, 1]")
        if rew.sampling_temperature < 0:
            raise ValueError("rewiring sampling_temperature must be >= 0")
        if rew.interval_layers < 1:
            raise ValueError("interval_layers must be >= 1")
        if rew.distance_schedule is not None:
            for i, tier in enumerate(rew.distance_schedule):
                t = sum(tier.get(k, 0) for k in ("near", "medium", "far", "random"))
                if t != self.routed_blocks:
                    raise ValueError(
                        f"distance_schedule[{i}] budget {t} "
                        f"!= routed_blocks ({self.routed_blocks})")
        w = self.walks
        WALK_TYPES = {"random", "semantic", "novelty"}
        if w.walk_type not in WALK_TYPES:
            raise ValueError(
                f"Unknown walk_type {w.walk_type!r}, "
                f"expected one of {sorted(WALK_TYPES)}")
        if w.walk_length < 1:
            raise ValueError("walk_length must be >= 1")
        if w.walks_per_start < 1:
            raise ValueError("walks_per_start must be >= 1")
        if not (0.0 <= w.continuation_probability <= 1.0):
            raise ValueError("continuation_probability must be in [0, 1]")
        if not (0.0 <= w.restart_probability <= 1.0):
            raise ValueError("restart_probability must be in [0, 1]")
        if w.reserved_edge_budget < 0:
            raise ValueError("reserved_edge_budget must be >= 0")
        if w.reserved_edge_budget > self.routed_blocks:
            raise ValueError(
                f"reserved_edge_budget ({w.reserved_edge_budget}) "
                f"> routed_blocks ({self.routed_blocks})")
        if w.semantic_distance_schedule is not None:
            valid_bands = {"near", "medium", "far", "random"}
            for band in w.semantic_distance_schedule:
                if band not in valid_bands:
                    raise ValueError(
                        f"Invalid band {band!r} in semantic_distance_schedule, "
                        f"expected one of {sorted(valid_bands)}")
        ml = self.memory_layout
        if ml.strategy not in MEMORY_STRATEGIES:
            raise ValueError(
                f"Unknown memory_layout strategy {ml.strategy!r}, "
                f"expected one of {sorted(MEMORY_STRATEGIES)}")
        if ml.cluster_size < 1:
            raise ValueError("memory_layout cluster_size must be >= 1")
        if ml.clustering_method not in ("semantic", "graph"):
            raise ValueError(
                f"Unknown clustering_method {ml.clustering_method!r}")
        if ml.cluster_refresh_layers < 0:
            raise ValueError("cluster_refresh_layers must be >= 0")
        if not (0.0 < ml.graph_history_decay <= 1.0):
            raise ValueError("graph_history_decay must be in (0, 1]")
        if ml.reorder_threshold < 0.0:
            raise ValueError("reorder_threshold must be >= 0")
