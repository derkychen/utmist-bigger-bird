"""Experiment arm definitions for graph-sparse attention evaluation.

Each arm is a dict with:
  name:         short identifier for results files and tables.
  arm:          eval_niah --arm value ("dense", "vs", "graph").
  attn:         JSON attn config for --arm vs.
  graph_config: JSON graph config for --arm graph.
  description:  human-readable summary.

Arms are organized in stages to avoid a full Cartesian product:
  Stage 0: baselines (dense, BB v3 variants).
  Stage 1: static graph ablations (block budget, band allocation, temperature).
  Stage 2: rewiring ablations (mode, interval, replace probability, schedule).
  Stage 3: walk ablations (type, length, continuation probability).
  Stage 4: memory layout (cluster method, cluster size, packing).
  Stage 5: best graph + memory combined.
"""

# ── BB v3 configs ───────────────────────────────────────────────────

BB_V3 = {"sink": 64, "window": 16384, "vertical": 512, "exact_tail": 256}
BB_V3_MATCHED = {"sink": 64, "window": 4096, "vertical": 512, "exact_tail": 256}

# ── Graph base configs ──────────────────────────────────────────────

GRAPH_BASE = {
    "mode": "offline",
    "query_chunk_size": 512,
    "key_block_size": 64,
    "routed_blocks": 8,
    "routing_dim": 64,
    "seed": 42,
    "semantic": {
        "near_blocks": 3,
        "medium_blocks": 2,
        "far_blocks": 2,
        "random_blocks": 1,
        "sampling_temperature": 0.0,
    },
    "rewiring": {"mode": "none"},
    "walks": {"enabled": False},
    "sink": 64,
    "window": 4096,
    "exact_tail": 256,
}


def _graph(overrides=None, sem_overrides=None, rew_overrides=None,
           walk_overrides=None, mem_overrides=None, attn_overrides=None):
    """Build a graph config dict with optional overrides."""
    import copy
    cfg = copy.deepcopy(GRAPH_BASE)
    if sem_overrides:
        cfg["semantic"].update(sem_overrides)
    if rew_overrides:
        cfg["rewiring"].update(rew_overrides)
    if walk_overrides:
        cfg["walks"].update(walk_overrides)
    if mem_overrides:
        cfg.setdefault("memory_layout", {}).update(mem_overrides)
    if attn_overrides:
        for k in ("sink", "window", "exact_tail"):
            if k in attn_overrides:
                cfg[k] = attn_overrides.pop(k)
    if overrides:
        for k, v in overrides.items():
            if k not in ("semantic", "rewiring", "walks", "memory_layout"):
                cfg[k] = v
    return cfg


# ═══════════════════════════════════════════════════════════════════
# Stage 0: Baselines
# ═══════════════════════════════════════════════════════════════════

STAGE_0 = [
    {
        "name": "dense",
        "arm": "dense",
        "attn": {},
        "graph_config": None,
        "description": "Dense FlashAttention (full quadratic).",
    },
    {
        "name": "bb_v3",
        "arm": "vs",
        "attn": BB_V3,
        "graph_config": None,
        "description": "Bigger Bird v3: sink64 + window16K + vert512 + tail256.",
    },
    {
        "name": "bb_v3_matched",
        "arm": "vs",
        "attn": BB_V3_MATCHED,
        "graph_config": None,
        "description": "BB v3 with window=4096 (matched to graph window budget).",
    },
]

# ═══════════════════════════════════════════════════════════════════
# Stage 1: Static graph ablations
# ═══════════════════════════════════════════════════════════════════

STAGE_1 = [
    {
        "name": "graph_static_b4",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            overrides={"routed_blocks": 4},
            sem_overrides={"near_blocks": 2, "medium_blocks": 1,
                           "far_blocks": 0, "random_blocks": 1},
        ),
        "description": "Static graph, 4 routed blocks.",
    },
    {
        "name": "graph_static_b8",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(),
        "description": "Static graph, 8 routed blocks (default).",
    },
    {
        "name": "graph_static_b16",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            overrides={"routed_blocks": 16},
            sem_overrides={"near_blocks": 6, "medium_blocks": 4,
                           "far_blocks": 4, "random_blocks": 2},
        ),
        "description": "Static graph, 16 routed blocks.",
    },
    {
        "name": "graph_near_heavy",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            sem_overrides={"near_blocks": 5, "medium_blocks": 1,
                           "far_blocks": 1, "random_blocks": 1},
        ),
        "description": "Static graph, near-heavy allocation (5/1/1/1).",
    },
    {
        "name": "graph_far_heavy",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            sem_overrides={"near_blocks": 1, "medium_blocks": 2,
                           "far_blocks": 4, "random_blocks": 1},
        ),
        "description": "Static graph, far-heavy allocation (1/2/4/1).",
    },
    {
        "name": "graph_temp_05",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            sem_overrides={"sampling_temperature": 0.5},
        ),
        "description": "Static graph, sampling temperature=0.5.",
    },
    {
        "name": "graph_temp_10",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            sem_overrides={"sampling_temperature": 1.0},
        ),
        "description": "Static graph, sampling temperature=1.0.",
    },
]

# ═══════════════════════════════════════════════════════════════════
# Stage 2: Rewiring ablations
# ═══════════════════════════════════════════════════════════════════

STAGE_2 = [
    {
        "name": "graph_rewire_random",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(rew_overrides={"mode": "random"}),
        "description": "Random rewiring each layer.",
    },
    {
        "name": "graph_rewire_static",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(rew_overrides={"mode": "static"}),
        "description": "Static rewiring (keep valid previous edges).",
    },
    {
        "name": "graph_rewire_sem_03",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            rew_overrides={"mode": "semantic", "replace_probability": 0.3},
        ),
        "description": "Semantic rewiring, replace_prob=0.3.",
    },
    {
        "name": "graph_rewire_sem_05",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            rew_overrides={"mode": "semantic", "replace_probability": 0.5},
        ),
        "description": "Semantic rewiring, replace_prob=0.5 (default).",
    },
    {
        "name": "graph_rewire_sem_07",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            rew_overrides={"mode": "semantic", "replace_probability": 0.7},
        ),
        "description": "Semantic rewiring, replace_prob=0.7.",
    },
    {
        "name": "graph_rewire_interval_4",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            rew_overrides={"mode": "semantic", "replace_probability": 0.5,
                           "interval_layers": 4},
        ),
        "description": "Semantic rewiring, recompute every 4 layers.",
    },
    {
        "name": "graph_rewire_schedule",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            rew_overrides={
                "mode": "distance_schedule",
                "distance_schedule": [
                    {"near": 5, "medium": 1, "far": 1, "random": 1},
                    {"near": 3, "medium": 2, "far": 2, "random": 1},
                    {"near": 1, "medium": 2, "far": 4, "random": 1},
                ],
            },
        ),
        "description": "Distance schedule: near-heavy early, far-heavy late.",
    },
]

# ═══════════════════════════════════════════════════════════════════
# Stage 3: Walk ablations (built on best rewiring from stage 2)
# ═══════════════════════════════════════════════════════════════════

_BEST_REW = {"mode": "semantic", "replace_probability": 0.5}

STAGE_3 = [
    {
        "name": "graph_walk_random_l3",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            rew_overrides=_BEST_REW,
            walk_overrides={"enabled": True, "walk_type": "random",
                            "walk_length": 3, "walks_per_start": 1,
                            "reserved_edge_budget": 1},
        ),
        "description": "Semantic rewiring + random walks (length=3).",
    },
    {
        "name": "graph_walk_random_l5",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            rew_overrides=_BEST_REW,
            walk_overrides={"enabled": True, "walk_type": "random",
                            "walk_length": 5, "walks_per_start": 1,
                            "reserved_edge_budget": 1},
        ),
        "description": "Semantic rewiring + random walks (length=5).",
    },
    {
        "name": "graph_walk_semantic_l5",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            rew_overrides=_BEST_REW,
            walk_overrides={"enabled": True, "walk_type": "semantic",
                            "walk_length": 5, "walks_per_start": 1,
                            "reserved_edge_budget": 1},
        ),
        "description": "Semantic rewiring + semantic walks (length=5).",
    },
    {
        "name": "graph_walk_novelty_l5",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            rew_overrides=_BEST_REW,
            walk_overrides={"enabled": True, "walk_type": "novelty",
                            "walk_length": 5, "walks_per_start": 1,
                            "reserved_edge_budget": 1},
        ),
        "description": "Semantic rewiring + novelty walks (length=5).",
    },
    {
        "name": "graph_walk_random_x2",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            rew_overrides=_BEST_REW,
            walk_overrides={"enabled": True, "walk_type": "random",
                            "walk_length": 5, "walks_per_start": 2,
                            "reserved_edge_budget": 2},
        ),
        "description": "Semantic rewiring + 2 random walks per chunk.",
    },
    {
        "name": "graph_walk_random_cont06",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            rew_overrides=_BEST_REW,
            walk_overrides={"enabled": True, "walk_type": "random",
                            "walk_length": 5, "walks_per_start": 1,
                            "reserved_edge_budget": 1,
                            "continuation_probability": 0.6},
        ),
        "description": "Random walks, continuation_prob=0.6 (shorter paths).",
    },
]

# ═══════════════════════════════════════════════════════════════════
# Stage 4: Memory layout (built on best graph config from stages 1-3)
# ═══════════════════════════════════════════════════════════════════

_BEST_GRAPH_BASE = {
    "rew_overrides": _BEST_REW,
    "walk_overrides": {"enabled": True, "walk_type": "random",
                       "walk_length": 5, "walks_per_start": 1,
                       "reserved_edge_budget": 1},
}

STAGE_4 = [
    {
        "name": "graph_mem_original",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            **_BEST_GRAPH_BASE,
            mem_overrides={"strategy": "original"},
        ),
        "description": "Best graph + original memory layout (baseline).",
    },
    {
        "name": "graph_mem_sorted",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            **_BEST_GRAPH_BASE,
            mem_overrides={"strategy": "sorted_indices", "cluster_size": 64},
        ),
        "description": "Best graph + sorted indices (no K/V data movement).",
    },
    {
        "name": "graph_mem_semantic_32",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            **_BEST_GRAPH_BASE,
            mem_overrides={"strategy": "semantic_clustered", "cluster_size": 32},
        ),
        "description": "Best graph + semantic clustering, cluster_size=32.",
    },
    {
        "name": "graph_mem_semantic_64",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            **_BEST_GRAPH_BASE,
            mem_overrides={"strategy": "semantic_clustered", "cluster_size": 64},
        ),
        "description": "Best graph + semantic clustering, cluster_size=64.",
    },
    {
        "name": "graph_mem_graph_64",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            **_BEST_GRAPH_BASE,
            mem_overrides={"strategy": "graph_clustered", "cluster_size": 64,
                           "clustering_method": "graph"},
        ),
        "description": "Best graph + graph-based clustering, cluster_size=64.",
    },
]

# ═══════════════════════════════════════════════════════════════════
# Stage 1b: Debug/diagnostic arms for integration testing
# ═══════════════════════════════════════════════════════════════════

STAGE_1B = [
    {
        "name": "graph_inject_bb_v3",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(),
        "debug": {"route_mode": "bb_v3", "vertical": 512, "n_route": 64},
        "description": "Graph kernel path with injected BB v3 QK routes.",
    },
    {
        "name": "graph_question_cond",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            sem_overrides={"question_conditioned": True, "question_tokens": 64},
        ),
        "description": "Graph with question-conditioned semantic routing.",
    },
    {
        "name": "graph_budget_32",
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(
            overrides={"routed_blocks": 32},
            sem_overrides={"near_blocks": 12, "medium_blocks": 8,
                           "far_blocks": 8, "random_blocks": 4},
        ),
        "description": "Graph semantic routing, 32 routed blocks (2048 tokens).",
    },
]

# ═══════════════════════════════════════════════════════════════════
# Stage 1c: Hybrid QK + semantic graph routing ablation
# ═══════════════════════════════════════════════════════════════════

def _hybrid_arm(name, qk, graph, desc):
    """Build a hybrid arm: QK budget + graph budget summing to 512 tokens.

    The graph config always uses the default 8 routed blocks (512 tokens).
    The graph_budget in debug config truncates graph indices after routing.
    """
    return {
        "name": name,
        "arm": "graph",
        "attn": {},
        "graph_config": _graph(),
        "debug": {
            "route_mode": "hybrid",
            "qk_budget": qk,
            "graph_budget": graph,
            "total_budget": 512,
            "n_route": 64,
            "track_hybrid": True,
        },
        "description": desc,
    }


STAGE_1C = [
    _hybrid_arm("hybrid_qk512_g0", 512, 0,
                "Pure QK routing (512 tokens) — must match BB v3 matched."),
    _hybrid_arm("hybrid_qk448_g64", 448, 64,
                "Hybrid: 448 QK + 64 graph (1 block)."),
    _hybrid_arm("hybrid_qk384_g128", 384, 128,
                "Hybrid: 384 QK + 128 graph (2 blocks)."),
    _hybrid_arm("hybrid_qk256_g256", 256, 256,
                "Hybrid: 256 QK + 256 graph (4 blocks)."),
    _hybrid_arm("hybrid_qk128_g384", 128, 384,
                "Hybrid: 128 QK + 384 graph (6 blocks)."),
    _hybrid_arm("hybrid_qk0_g512", 0, 512,
                "Pure graph routing (512 tokens, 8 blocks) — matches Stage 1 default."),
]


# ═══════════════════════════════════════════════════════════════════
# All stages
# ═══════════════════════════════════════════════════════════════════

ALL_ARMS = STAGE_0 + STAGE_1 + STAGE_2 + STAGE_3 + STAGE_4

STAGES = {
    0: STAGE_0,
    1: STAGE_1,
    2: STAGE_2,
    3: STAGE_3,
    4: STAGE_4,
    "1b": STAGE_1B,
    "1c": STAGE_1C,
}


def get_arms(stage=None, names=None):
    """Filter arms by stage number (int) or key (str like '1b'), or name list."""
    if stage is not None:
        pool = STAGES.get(stage, STAGES.get(str(stage), ALL_ARMS))
    else:
        pool = ALL_ARMS
    if names:
        name_set = set(names)
        pool = [a for a in pool if a["name"] in name_set]
    return pool
