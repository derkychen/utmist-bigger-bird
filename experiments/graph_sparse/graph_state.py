"""Cross-layer graph state and coordinator.

The coordinator is shared across all decoder layers. It maintains the graph state
(previous routes, active walks) and builds per-layer routing indices compatible
with the bigger_bird_flash kernel: [H, G, K] int32 indices.
"""
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from .config import GraphConfig
from .routing.candidates import find_candidates
from .routing.semantic import classify_semantic_distance_batched
from .routing.sample import (
    sample_semantic_edges_batched,
    expand_blocks_to_indices,
    deduplicate_and_sort,
)
from .routing.rewire import rewire_edges, compute_edge_overlap
from .routing.walks import (
    Walk, advance_walks, inject_walk_edges, spawn_walks, walk_diagnostics,
)


@dataclass
class TimingRecord:
    """CUDA-event or wall-clock timing for one layer."""
    graph_build_ms: float = 0.0
    rewire_ms: float = 0.0
    walk_ms: float = 0.0
    expand_ms: float = 0.0
    total_ms: float = 0.0


def _cuda_event():
    return torch.cuda.Event(enable_timing=True)


@dataclass
class GraphState:
    previous_block_routes: Optional[torch.Tensor] = None
    active_walks: List[Walk] = field(default_factory=list)
    completed_walks: List[Walk] = field(default_factory=list)
    prefill_id: int = 0
    layer_edge_overlap: List[float] = field(default_factory=list)
    walk_diagnostics_per_layer: List[dict] = field(default_factory=list)
    last_similarity: Optional[torch.Tensor] = None
    timing: List[TimingRecord] = field(default_factory=list)


class GraphCoordinator:
    """Orchestrates graph routing across all layers of the model.

    One instance is shared by all GraphSparseAttention layers. It:
    1. Clears state between prompts (begin_prefill).
    2. Builds per-layer graph routing indices from pre-RoPE hidden states.
    3. Records routes for cross-layer rewiring.
    4. Advances traveling paths and spawns new walks.
    5. Tracks edge overlap and walk diagnostics between adjacent layers.
    """

    def __init__(self, config: GraphConfig, num_heads: int, num_kv_heads: int,
                 head_dim: int, num_layers: int):
        config.validate()
        self.config = config
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.state = GraphState()
        self._proj = None

    def begin_prefill(self):
        """Clear all state from the previous prompt."""
        self.state = GraphState(prefill_id=self.state.prefill_id + 1)

    def _get_projection(self, d: int, device: torch.device) -> Optional[torch.Tensor]:
        """Lazily create a fixed random projection matrix for routing."""
        if self.config.routing_dim >= d:
            return None
        if self._proj is None or self._proj.shape[0] != d or self._proj.device != device:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.config.seed)
            self._proj = torch.randn(d, self.config.routing_dim, generator=gen,
                                     dtype=torch.float32).to(device)
            self._proj = self._proj / (d ** 0.5)
        return self._proj

    def _band_budget(self, layer_idx: int):
        """Return (near_k, medium_k, far_k, random_k) for this layer.

        Uses the base semantic config budgets unless the rewiring mode is
        ``distance_schedule`` and a schedule is provided, in which case the
        schedule tier for this layer is used.
        """
        cfg = self.config
        sem = cfg.semantic
        rew = cfg.rewiring
        if rew.mode == "distance_schedule" and rew.distance_schedule:
            sched = rew.distance_schedule
            tier = min(layer_idx * len(sched) // max(1, self.num_layers), len(sched) - 1)
            alloc = sched[tier]
            return (alloc.get("near", sem.near_blocks),
                    alloc.get("medium", sem.medium_blocks),
                    alloc.get("far", sem.far_blocks),
                    alloc.get("random", sem.random_blocks))
        return sem.near_blocks, sem.medium_blocks, sem.far_blocks, sem.random_blocks

    def build_layer_graph(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """Build routing indices for one layer.

        Args:
            layer_idx: which decoder layer (0-indexed).
            hidden_states: [N, D] pre-RoPE hidden states (batch dim already squeezed).

        Returns:
            indices: [H, G, K] int32 token positions for the bigger_bird_flash kernel.
                G = num_query_chunks, K = routed_blocks * key_block_size.
        """
        cfg = self.config
        rew = cfg.rewiring
        wk = cfg.walks
        n, d = hidden_states.shape
        num_blocks = -(-n // cfg.key_block_size)
        use_cuda = hidden_states.is_cuda
        timing = TimingRecord()

        if use_cuda:
            t_start = _cuda_event()
            t_start.record()

        should_rebuild = (
            self.state.previous_block_routes is None
            or layer_idx % max(1, rew.interval_layers) == 0
        )

        if should_rebuild:
            block_edges, num_q = self._build_fresh(layer_idx, hidden_states)
        else:
            block_edges = self.state.previous_block_routes.clone()
            num_q = block_edges.shape[0]

        if use_cuda:
            t_graph = _cuda_event()
            t_graph.record()

        if wk.enabled and self.state.active_walks:
            block_edges = inject_walk_edges(
                block_edges, self.state.active_walks,
                wk.reserved_edge_budget,
                cfg.query_chunk_size, cfg.key_block_size, n)

        if self.state.previous_block_routes is not None:
            self.state.layer_edge_overlap.append(
                compute_edge_overlap(self.state.previous_block_routes, block_edges))

        self.state.previous_block_routes = block_edges

        if wk.enabled:
            gen_device = hidden_states.device if hidden_states.is_cuda else "cpu"
            walk_gen = torch.Generator(device=gen_device)
            walk_gen.manual_seed(wk.seed + layer_idx)

            self.state.active_walks = advance_walks(
                self.state.active_walks, block_edges, layer_idx,
                cfg.query_chunk_size, cfg.key_block_size, n,
                walk_config=wk,
                similarity=self.state.last_similarity,
                generator=walk_gen)

            dead = [w for w in self.state.active_walks if not w.alive]
            self.state.completed_walks.extend([w for w in dead if w.completed])
            self.state.active_walks = [w for w in self.state.active_walks if w.alive]

            if layer_idx == 0:
                new_walks = spawn_walks(
                    block_edges, num_q, wk, layer_idx, num_blocks,
                    generator=walk_gen)
                self.state.active_walks.extend(new_walks)

            self.state.walk_diagnostics_per_layer.append(
                walk_diagnostics(self.state.active_walks, self.state.completed_walks))

        if use_cuda:
            t_walk = _cuda_event()
            t_walk.record()

        token_indices = expand_blocks_to_indices(block_edges, cfg.key_block_size, n)
        token_indices = deduplicate_and_sort(token_indices)

        K = cfg.routed_keys
        Q_chunks = num_q
        if token_indices.shape[1] < K:
            pad = torch.full((Q_chunks, K - token_indices.shape[1]), -1,
                             dtype=token_indices.dtype, device=token_indices.device)
            token_indices = torch.cat([token_indices, pad], 1)
        elif token_indices.shape[1] > K:
            token_indices = token_indices[:, :K]

        indices = token_indices[None, :, :].expand(self.num_heads, -1, -1).contiguous().to(torch.int32)

        if use_cuda:
            t_end = _cuda_event()
            t_end.record()
            torch.cuda.synchronize()
            timing.graph_build_ms = t_start.elapsed_time(t_graph)
            timing.walk_ms = t_graph.elapsed_time(t_walk)
            timing.expand_ms = t_walk.elapsed_time(t_end)
            timing.total_ms = t_start.elapsed_time(t_end)

        self.state.timing.append(timing)
        return indices

    def _build_fresh(self, layer_idx: int, hidden_states: torch.Tensor):
        """Run the full routing pipeline: candidates -> bands -> sample -> rewire."""
        cfg = self.config
        rew = cfg.rewiring
        sem = cfg.semantic
        n, d = hidden_states.shape
        proj = self._get_projection(d, hidden_states.device)

        sim, causal_mask, num_q, num_b = find_candidates(
            hidden_states, cfg, proj, causal=(cfg.mode == "strict_causal"))

        if cfg.walks.enabled and cfg.walks.walk_type == "semantic":
            self.state.last_similarity = sim

        bands = classify_semantic_distance_batched(
            sim, causal_mask, sem.near_quantile, sem.medium_quantile, sem.far_quantile)

        gen_device = hidden_states.device if hidden_states.is_cuda else "cpu"
        sample_gen = torch.Generator(device=gen_device)
        sample_gen.manual_seed(cfg.seed + layer_idx)

        near_k, medium_k, far_k, random_k = self._band_budget(layer_idx)

        block_edges = sample_semantic_edges_batched(
            bands, near_k, medium_k, far_k, random_k,
            causal_mask, generator=sample_gen,
            similarity=sim, temperature=sem.sampling_temperature)

        if rew.mode != "none" and self.state.previous_block_routes is not None:
            rewire_gen = torch.Generator(device=gen_device)
            rewire_gen.manual_seed(cfg.seed + layer_idx + self.num_layers)
            block_edges = rewire_edges(
                block_edges, self.state.previous_block_routes, causal_mask,
                mode=rew.mode, replace_prob=rew.replace_probability,
                preserve_frac=rew.preserve_edge_fraction,
                generator=rewire_gen)

        return block_edges, num_q

    def finish_layer(self, layer_idx: int, routes: torch.Tensor):
        """Hook for any post-layer bookkeeping (currently a no-op; state is updated in build)."""
        pass
