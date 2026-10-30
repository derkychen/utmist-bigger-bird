"""Classify candidate key blocks into semantic distance bands and sample edges."""
import torch


def classify_semantic_distance(similarity: torch.Tensor, causal_mask: torch.Tensor,
                               near_q: tuple, medium_q: tuple, far_q: tuple):
    """Partition eligible candidates into near/medium/far bands by similarity quantile.

    For each query chunk, eligible candidates (causal_mask=True, similarity > -inf) are
    ranked by similarity. The top near_q fraction are "near", the next medium_q fraction
    are "medium", and the rest are "far".

    Args:
        similarity: [Q, B] similarity scores (-inf for ineligible).
        causal_mask: [Q, B] boolean.
        near_q: (lo, hi) quantile range for near, e.g. (0.0, 0.20).
        medium_q: (lo, hi) quantile range for medium.
        far_q: (lo, hi) quantile range for far.

    Returns:
        bands: dict with keys "near", "medium", "far", each [Q, B] boolean mask.
    """
    Q, B = similarity.shape
    eligible = causal_mask & (similarity > float("-inf"))

    near_mask = torch.zeros(Q, B, dtype=torch.bool, device=similarity.device)
    medium_mask = torch.zeros_like(near_mask)
    far_mask = torch.zeros_like(near_mask)

    for q in range(Q):
        elig = eligible[q]
        if not elig.any():
            continue
        vals = similarity[q, elig]
        n_elig = vals.numel()
        sorted_idx = vals.argsort(descending=True)
        elig_positions = elig.nonzero(as_tuple=True)[0]
        ordered = elig_positions[sorted_idx]

        near_lo = int(near_q[0] * n_elig)
        near_hi = max(near_lo + 1, int(near_q[1] * n_elig))
        med_lo = near_hi
        med_hi = max(med_lo + 1, int(medium_q[1] * n_elig))
        far_lo = med_hi
        far_hi = n_elig

        if near_hi > near_lo:
            near_mask[q, ordered[near_lo:near_hi]] = True
        if med_hi > med_lo:
            medium_mask[q, ordered[med_lo:med_hi]] = True
        if far_hi > far_lo:
            far_mask[q, ordered[far_lo:far_hi]] = True

    return {"near": near_mask, "medium": medium_mask, "far": far_mask}


def classify_semantic_distance_batched(similarity: torch.Tensor, causal_mask: torch.Tensor,
                                       near_q: tuple, medium_q: tuple, far_q: tuple):
    """Vectorized version of classify_semantic_distance (no Python loop over Q).

    Uses argsort + fractional index boundaries per row.
    """
    Q, B = similarity.shape
    eligible = causal_mask & (similarity > float("-inf"))
    counts = eligible.sum(1)

    sim_work = similarity.clone()
    sim_work[~eligible] = float("-inf")
    order = sim_work.argsort(dim=1, descending=True)

    rank = torch.zeros_like(order)
    rank.scatter_(1, order, torch.arange(B, device=similarity.device).expand(Q, -1))

    near_hi = (near_q[1] * counts.float()).long().clamp(min=1)
    med_hi = (medium_q[1] * counts.float()).long().clamp(min=1)

    near_mask = eligible & (rank < near_hi[:, None])
    medium_mask = eligible & (rank >= near_hi[:, None]) & (rank < med_hi[:, None])
    far_mask = eligible & (rank >= med_hi[:, None])

    return {"near": near_mask, "medium": medium_mask, "far": far_mask}
