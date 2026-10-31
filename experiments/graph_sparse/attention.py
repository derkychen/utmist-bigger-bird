"""Graph-aware sparse attention layer and model installer.

GraphSparseAttention overrides RDAttention's routing to use the semantic graph
coordinator instead of the global top-V softmax routing. The existing Triton
kernel (bigger_bird_flash) is reused without changes — the graph router produces
indices in the same [H, G, K] format.

Two routing modes are supported via ``GraphConfig.mode``:

* **strict_causal** — routing for query chunk *q* uses only blocks whose last
  token is before *q* × chunk_size (the causal block mask). Exact-tail dense
  attention is disabled because it uses window = n, which depends on the full
  sequence length and is therefore not prefix-invariant. This mode is suitable
  for streaming / incremental inference where the routing decision for a prefix
  must not change when more tokens are appended.

* **offline** — all blocks are eligible candidates for every query chunk (the
  kernel still enforces per-token causal masking). Exact-tail is allowed. This
  mode is comparable to BB v3 whose routing is conditioned on the last n_route
  queries of the full prompt.

  **Offline prompt conditioning is NOT equivalent to streaming causal attention.**
  The routing decisions for early query chunks depend on blocks that appear later
  in the prompt, so the output for a prefix will change if different tokens are
  appended. Use this mode only when the full prompt is available before inference
  begins (e.g. RULER evaluation, offline benchmarks).
"""
import time

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from modal_rd.attention import RDAttention, bigger_bird_flash, rope_inplace, ROPE_CHUNK

from .config import GraphConfig
from .graph_state import GraphCoordinator


class GraphSparseAttention(RDAttention):
    """Sparse attention with semantic graph routing.

    Routing is done on pre-RoPE hidden states via the shared GraphCoordinator.
    The attention computation uses the same bigger_bird_flash kernel as BB v3.

    Debug modes (set via ``_debug_config`` dict):
      route_mode="bb_v3": bypass graph routing, use BB v3 QK top-k instead.
      route_mode="question_conditioned": graph routing with question-aware summaries.
      track_coverage=True + needle_span=(lo, hi): record needle coverage per chunk/head.
    """

    def __init__(self, base, *, coordinator: GraphCoordinator, graph_config: GraphConfig,
                 sink=64, window=4096, exact_tail=256, block_m=64, block_n=64,
                 num_warps=4, num_stages=2):
        if graph_config.query_chunk_size % block_m:
            raise ValueError(f"query_chunk_size ({graph_config.query_chunk_size}) must be "
                             f"a multiple of block_m ({block_m})")
        super().__init__(
            base,
            sink=sink, window=window, vertical=0, n_route=0,
            exact_tail=exact_tail, kv_evict=False,
            block_m=block_m, block_n=block_n,
            num_warps=num_warps, num_stages=num_stages,
        )
        self.coordinator = coordinator
        self.graph_config = graph_config
        self._routing_states = None
        self.last_graph_build_ms = 0.0
        self.last_attention_ms = 0.0
        self._debug_config = {}
        self._last_coverage = None

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        b, t, _ = hidden_states.shape
        self._routing_states = hidden_states[0]

        q = self.q_proj(hidden_states).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if torch.is_grad_enabled() and q.requires_grad:
            q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        else:
            q, k = rope_inplace(q, *position_embeddings), rope_inplace(k, *position_embeddings)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        n = k.shape[-2]
        if b != 1:
            raise ValueError("batch size 1 only")
        if t < n or n <= self.sink + self.window:
            out = self.exact_attention(q, k, v, t, n)
        else:
            out = self.graph_sparse_prefill(q, k, v)
        self._routing_states = None
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out), None

    def graph_sparse_prefill(self, q, k, v):
        """Sparse prefill using graph-routed indices.

        When ``_debug_config["route_mode"]`` is set:
          "bb_v3"   — bypass the graph coordinator and compute BB v3 QK top-k
                      routes, replicated across query groups.
          "hybrid"  — merge QK top-k routes with graph semantic routes.
          None      — normal graph routing via the coordinator.
        """
        n = k.shape[-2]
        debug = self._debug_config

        if debug.get("route_mode") == "bb_v3":
            idx, chunk = self._bb_v3_routes(q, k, n, debug)
            self.last_graph_build_ms = 0.0
        elif debug.get("route_mode") == "hybrid":
            idx, chunk = self._hybrid_routes(q, k, n, debug)
        else:
            t0 = time.perf_counter()
            idx = self.coordinator.build_layer_graph(self.layer_idx, self._routing_states)
            self.last_graph_build_ms = (time.perf_counter() - t0) * 1000
            chunk = self.graph_config.query_chunk_size

        self.last_routes = idx

        Q, K_t, V_t = q[0], k[0], v[0]
        flash = dict(num_heads=self.num_heads, kv_groups=self.groups, scale=self.scaling,
                     block_m=self.block_m, block_n=self.block_n,
                     num_warps=self.num_warps, num_stages=self.num_stages)

        t1 = time.perf_counter()
        out = bigger_bird_flash(Q, K_t, V_t, idx.contiguous(), front=self.sink, window=self.window,
                                token_mask=None, route_chunk=chunk, **flash)[None]

        r = min(self.exact_tail, n) if self.graph_config.mode == "offline" else 0
        if r:
            bh = self.num_heads
            none = torch.full((bh, 1, 1), -1, dtype=torch.int32, device=q.device)
            out[:, :, n - r:] = bigger_bird_flash(Q[:, n - r:], K_t, V_t, none, front=0, window=n,
                                                  token_mask=None, route_chunk=None, **flash)[None]
        self.last_attention_ms = (time.perf_counter() - t1) * 1000

        if debug.get("track_coverage"):
            self._last_coverage = self._compute_coverage(idx, chunk, n, debug)

        return out

    def _bb_v3_routes(self, q, k, n, debug):
        """Compute BB v3 content-based QK routes, replicated across query groups."""
        budget = debug.get("vertical", 512)
        n_route = debug.get("n_route", 64)
        chunk = self.graph_config.query_chunk_size
        G = -(-n // chunk)

        old_vert, old_sqrt, old_nroute = self.vertical, self.vertical_sqrt, self.n_route
        self.vertical, self.vertical_sqrt, self.n_route = budget, 0.0, n_route
        try:
            per_head = self.routes(q, k)
        finally:
            self.vertical, self.vertical_sqrt, self.n_route = old_vert, old_sqrt, old_nroute

        idx = per_head[:, None, :].expand(-1, G, -1).contiguous()
        return idx, chunk

    def _hybrid_routes(self, q, k, n, debug):
        """Merge QK top-k routes with graph semantic routes.

        debug config keys:
          qk_budget:    number of QK-routed token positions per head (0 = pure graph).
          graph_budget: number of graph-routed token positions, head-shared (0 = pure QK).
                        The graph coordinator runs with its full config; the budget just
                        truncates the resulting indices to at most graph_budget valid tokens.
          n_route:      queries used for QK routing (default 64).
          total_budget: total routed token slots (default 512). QK + graph may produce
                        fewer unique tokens than total_budget after deduplication;
                        remaining slots are filled from QK overflow.
        """
        qk_budget = debug.get("qk_budget", 0)
        graph_budget = debug.get("graph_budget", 0)
        total_budget = debug.get("total_budget", 512)
        n_route = debug.get("n_route", 64)
        chunk = self.graph_config.query_chunk_size
        G = -(-n // chunk)
        H = self.num_heads
        device = q.device

        t0 = time.perf_counter()

        # --- QK routes: [H, qk_budget] per head ---
        qk_indices = None
        qk_overflow = None
        if qk_budget > 0:
            old_vert, old_sqrt, old_nroute = self.vertical, self.vertical_sqrt, self.n_route
            extra = min(qk_budget, total_budget)
            self.vertical, self.vertical_sqrt, self.n_route = extra, 0.0, n_route
            try:
                qk_full = self.routes(q, k)  # [H, extra]
            finally:
                self.vertical, self.vertical_sqrt, self.n_route = old_vert, old_sqrt, old_nroute
            qk_indices = qk_full[:, :qk_budget]  # [H, qk_budget]
            if extra > qk_budget:
                qk_overflow = qk_full[:, qk_budget:]  # [H, extra - qk_budget]

        t_qk = time.perf_counter()

        # --- Graph routes: [H, G, K_full] head-shared, truncated to graph_budget ---
        graph_indices = None
        if graph_budget > 0:
            graph_idx = self.coordinator.build_layer_graph(
                self.layer_idx, self._routing_states)  # [H, G, K_full]
            # Truncate each group to at most graph_budget valid tokens
            K_full = graph_idx.shape[-1]
            if K_full > graph_budget:
                graph_idx = graph_idx[:, :, :graph_budget]
            graph_indices = graph_idx

        t_graph = time.perf_counter()

        # --- Merge: per head, per query group ---
        merged = torch.full((H, G, total_budget), -1, dtype=torch.int32, device=device)

        for h in range(H):
            for g in range(G):
                pieces = []
                if qk_indices is not None:
                    pieces.append(qk_indices[h])  # [qk_budget]
                if graph_indices is not None:
                    pieces.append(graph_indices[h, g])  # [K_graph]

                if not pieces:
                    continue

                cat = torch.cat(pieces)
                valid = cat[cat >= 0]
                unique = valid.unique(sorted=True)

                # Fill remaining slots from QK overflow if available
                fill = total_budget - unique.numel()
                if fill > 0 and qk_overflow is not None:
                    extra_valid = qk_overflow[h][qk_overflow[h] >= 0]
                    mask = ~torch.isin(extra_valid, unique)
                    extra_new = extra_valid[mask][:fill]
                    if extra_new.numel() > 0:
                        unique = torch.cat([unique, extra_new]).unique(sorted=True)

                count = min(unique.numel(), total_budget)
                merged[h, g, :count] = unique[:count].to(torch.int32)

        self.last_graph_build_ms = (time.perf_counter() - t0) * 1000

        # --- Diagnostics ---
        if debug.get("track_hybrid"):
            qk_count = qk_indices[qk_indices >= 0].numel() // H if qk_indices is not None else 0
            graph_count = 0
            if graph_indices is not None:
                for g in range(G):
                    graph_count += (graph_indices[0, g] >= 0).sum().item()
                graph_count = graph_count // max(1, G)
            valid_merged = (merged >= 0).sum(dim=-1).float()
            self._last_hybrid_diag = {
                "layer": self.layer_idx,
                "qk_count_per_head": qk_count,
                "graph_count_per_group": graph_count,
                "merged_mean": valid_merged.mean().item(),
                "qk_ms": (t_qk - t0) * 1000,
                "graph_ms": (t_graph - t_qk) * 1000,
                "merge_ms": (time.perf_counter() - t_graph) * 1000,
            }
            if qk_indices is not None and graph_indices is not None:
                overlap_total = 0
                for h in range(H):
                    qk_set = set(qk_indices[h][qk_indices[h] >= 0].tolist())
                    for g in range(G):
                        gr_set = set(graph_indices[h, g][graph_indices[h, g] >= 0].tolist())
                        overlap_total += len(qk_set & gr_set)
                self._last_hybrid_diag["overlap_mean"] = overlap_total / (H * G)

        return merged, chunk

    def _compute_coverage(self, idx, chunk, n, debug):
        """Compute routing coverage diagnostics for the current layer."""
        needle_span = debug.get("needle_span")
        if needle_span is None:
            return None
        lo, hi = needle_span
        H = idx.shape[0]
        G = idx.shape[1] if idx.ndim == 3 else 1
        flat = idx.view(H, G, -1) if idx.ndim == 3 else idx[:, None, :]
        K = flat.shape[-1]

        hit = (flat >= lo) & (flat < hi)
        needle_any = hit.any(dim=-1)

        valid = flat >= 0
        in_sink = (flat >= 0) & (flat < self.sink)
        nonlocal_count = (valid & ~in_sink).sum(dim=-1).float()

        final_g = G - 1
        return {
            "layer": self.layer_idx,
            "needle_hit_any_head_per_chunk": needle_any.any(dim=0).tolist(),
            "needle_frac_heads_per_chunk": needle_any.float().mean(dim=0).tolist(),
            "final_chunk_hit": needle_any[:, final_g].any().item(),
            "final_chunk_per_head": needle_any[:, final_g].tolist(),
            "nonlocal_per_chunk_mean": nonlocal_count.mean(dim=0).tolist(),
            "unique_routed_per_head": [
                flat[h][flat[h] >= 0].unique().numel() for h in range(H)
            ],
        }


def install_graph(model, config: GraphConfig, *, sink=64, window=4096, exact_tail=256,
                  block_m=64, block_n=64, num_warps=4, num_stages=2):
    """Replace every LlamaAttention layer with GraphSparseAttention.

    Registers a forward pre-hook on the model to call coordinator.begin_prefill()
    at the start of each full-sequence forward pass (t == n), ensuring graph state
    is cleared between prompts.

    Routing mode is controlled by ``config.mode``:

    * ``strict_causal``: prefix-invariant routing (causal block mask, no exact-tail).
    * ``offline``: full-prompt-conditioned routing (all blocks eligible, exact-tail
      allowed). NOT equivalent to streaming causal attention — early query chunks'
      routes depend on later blocks, so the output for a prefix changes when the
      suffix changes.

    Args:
        model: a HuggingFace LlamaForCausalLM (or compatible).
        config: GraphConfig for graph routing.
        sink, window, exact_tail: attention budget parameters.

    Returns:
        The modified model (in place).
    """
    cfg = model.config
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // num_heads
    num_layers = cfg.num_hidden_layers

    coordinator = GraphCoordinator(config, num_heads, num_kv_heads, head_dim, num_layers)

    for layer in model.model.layers:
        base = getattr(layer.self_attn, "_rd_base", layer.self_attn)
        graph_attn = GraphSparseAttention(
            base,
            coordinator=coordinator,
            graph_config=config,
            sink=sink, window=window, exact_tail=exact_tail,
            block_m=block_m, block_n=block_n,
            num_warps=num_warps, num_stages=num_stages,
        )
        graph_attn._rd_base = base
        layer.self_attn = graph_attn

    def _begin_prefill_hook(module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is not None and ids.shape[1] > 1:
            coordinator.begin_prefill()

    model._graph_coordinator = coordinator
    model._graph_prefill_hook = model.register_forward_pre_hook(_begin_prefill_hook, with_kwargs=True)
    return model
