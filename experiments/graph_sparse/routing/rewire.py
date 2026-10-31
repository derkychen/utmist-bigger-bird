"""Cross-layer edge rewiring: control how graph connections evolve across layers.

Modes:
    none              -- no rewiring, each layer builds an independent graph.
    static            -- keep all causally-valid previous edges, fill gaps from fresh sample.
    random            -- uniform random from the causal pool each layer (ignores previous).
    partial_random    -- keep (1-replace_prob) of previous, replace rest with uniform random.
    semantic          -- keep (1-replace_prob) of previous, replace rest from semantic sample.
    partial_semantic  -- alias for semantic (backward-compat).
    distance_schedule -- same merge as semantic, but band budgets change per layer (config).

``preserve_edge_fraction`` protects the first N edge columns unconditionally so that
high-priority connections (e.g. walk edges) survive across layers.
"""
import torch


REWIRE_MODES = frozenset({
    "none", "static", "random", "partial_random",
    "semantic", "partial_semantic", "distance_schedule",
})


def rewire_edges(new_edges: torch.Tensor,
                 previous_routes: torch.Tensor,
                 causal_mask: torch.Tensor,
                 *,
                 mode: str,
                 replace_prob: float = 0.5,
                 preserve_frac: float = 0.0,
                 generator: torch.Generator = None) -> torch.Tensor:
    """Merge new_edges with previous_routes according to the rewiring mode.

    Args:
        new_edges: [Q, total_blocks] freshly sampled block indices from semantic pipeline.
        previous_routes: [Q, total_blocks] block indices from the previous layer (or None).
        causal_mask: [Q, B] boolean -- True where a key block is causally eligible.
        mode: rewiring strategy (see module doc).
        replace_prob: probability of replacing a non-protected previous edge.
        preserve_frac: fraction of edge columns (first N) unconditionally protected.
        generator: for reproducibility.

    Returns:
        [Q, total_blocks] rewired block indices.
    """
    if previous_routes is None:
        if mode == "random":
            return _uniform_random_edges(causal_mask, new_edges.shape[1], generator)
        return new_edges

    if mode == "static":
        return _static_rewire(new_edges, previous_routes, causal_mask, generator)
    if mode == "random":
        return _uniform_random_edges(causal_mask, new_edges.shape[1], generator)
    if mode == "partial_random":
        random_fill = _uniform_random_edges(causal_mask, new_edges.shape[1], generator)
        return _partial_rewire(random_fill, previous_routes, causal_mask,
                               replace_prob, preserve_frac, generator)
    if mode in ("semantic", "partial_semantic", "distance_schedule"):
        return _partial_rewire(new_edges, previous_routes, causal_mask,
                               replace_prob, preserve_frac, generator)
    return new_edges


def _prev_validity_mask(previous_routes: torch.Tensor,
                        causal_mask: torch.Tensor) -> torch.Tensor:
    """Boolean [Q, total]: True where a previous edge is >= 0 and causally valid."""
    B = causal_mask.shape[1]
    in_range = (previous_routes >= 0) & (previous_routes < B)
    safe_idx = previous_routes.clamp(0, B - 1)
    causal_ok = causal_mask.gather(1, safe_idx)
    return in_range & causal_ok


def _static_rewire(new_edges: torch.Tensor,
                   previous_routes: torch.Tensor,
                   causal_mask: torch.Tensor,
                   generator: torch.Generator = None) -> torch.Tensor:
    """Keep all valid previous edges; fill invalid or missing slots from new_edges."""
    valid = _prev_validity_mask(previous_routes, causal_mask)
    result = torch.where(valid, previous_routes, new_edges)
    return _dedup_and_replenish(result, causal_mask, generator)


def _uniform_random_edges(causal_mask: torch.Tensor,
                          total_blocks: int,
                          generator: torch.Generator = None) -> torch.Tensor:
    """Sample block indices uniformly at random from the causal pool (Gumbel-top-k)."""
    Q, B = causal_mask.shape
    noise = torch.empty(Q, B, device=causal_mask.device).exponential_(
        1.0, generator=generator).clamp(min=1e-20).log().neg()
    noisy = torch.where(causal_mask, noise,
                        torch.tensor(float("-inf"), device=causal_mask.device))
    k = min(total_blocks, B)
    _, topk_idx = noisy.topk(k, dim=1)
    valid = causal_mask.gather(1, topk_idx)
    neg = torch.tensor(-1, dtype=torch.long, device=causal_mask.device)
    result = topk_idx.where(valid, neg)
    if k < total_blocks:
        pad = torch.full((Q, total_blocks - k), -1, dtype=torch.long,
                         device=causal_mask.device)
        result = torch.cat([result, pad], 1)
    return result


def _partial_rewire(replacement_edges: torch.Tensor,
                    previous_routes: torch.Tensor,
                    causal_mask: torch.Tensor,
                    replace_prob: float,
                    preserve_frac: float,
                    generator: torch.Generator = None) -> torch.Tensor:
    """Probabilistically keep previous edges, replace rest with replacement_edges.

    The first floor(preserve_frac * total) edge columns are unconditionally kept
    (if causally valid).  Remaining columns are replaced with probability replace_prob.
    Invalid previous edges (-1 or causally ineligible) are always replaced.
    """
    Q, total = replacement_edges.shape
    prev_valid = _prev_validity_mask(previous_routes, causal_mask)

    n_preserved = int(preserve_frac * total)
    keep = torch.zeros(Q, total, dtype=torch.bool, device=replacement_edges.device)
    if n_preserved > 0:
        keep[:, :n_preserved] = True
    if n_preserved < total:
        keep[:, n_preserved:] = torch.rand(
            Q, total - n_preserved, generator=generator,
            device=replacement_edges.device) > replace_prob
    keep = keep & prev_valid

    result = replacement_edges.clone()
    result[keep] = previous_routes[keep]
    return _dedup_and_replenish(result, causal_mask, generator)


def _dedup_and_replenish(edges: torch.Tensor,
                         causal_mask: torch.Tensor,
                         generator: torch.Generator = None) -> torch.Tensor:
    """Remove duplicate block indices per query chunk and replenish from causal pool.

    For each query chunk, any block that appears more than once has its duplicate
    slots replaced with unused causally-valid blocks (or -1 if the pool is exhausted).
    Also enforces causal validity: any edge that is not causally valid is replaced.
    """
    Q, total = edges.shape
    B = causal_mask.shape[1]
    result = edges.clone()
    for q in range(Q):
        row = result[q]
        causal_ok = (row >= 0) & (row < B)
        safe = row.clamp(0, B - 1)
        causal_ok = causal_ok & causal_mask[q].gather(0, safe)
        row[~causal_ok] = -1

        seen = set()
        dup_slots = []
        for i in range(total):
            val = row[i].item()
            if val < 0:
                dup_slots.append(i)
            elif val in seen:
                row[i] = -1
                dup_slots.append(i)
            else:
                seen.add(val)
        if not dup_slots:
            continue
        pool = causal_mask[q].nonzero(as_tuple=True)[0]
        pool = pool[~torch.isin(pool, torch.tensor(list(seen), dtype=pool.dtype,
                                                     device=pool.device))]
        fill_count = min(len(dup_slots), pool.numel())
        if fill_count > 0:
            perm = torch.randperm(pool.numel(), device=pool.device,
                                  generator=generator)[:fill_count]
            for i, slot in enumerate(dup_slots[:fill_count]):
                row[slot] = pool[perm[i]]
    return result


def compute_edge_overlap(prev_edges: torch.Tensor,
                         new_edges: torch.Tensor) -> float:
    """Fraction of new edges that also appeared in prev_edges (set overlap, mean over Q).

    Block-level comparison: compares *sets* of routed blocks per query chunk,
    not positional alignment within the edge array.
    """
    Q = prev_edges.shape[0]
    total = 0.0
    counted = 0
    for q in range(Q):
        prev_valid = prev_edges[q][prev_edges[q] >= 0]
        new_valid = new_edges[q][new_edges[q] >= 0]
        if new_valid.numel() == 0:
            continue
        prev_set = set(prev_valid.tolist())
        new_set = set(new_valid.tolist())
        total += len(prev_set & new_set) / len(new_set)
        counted += 1
    return total / counted if counted else 0.0
