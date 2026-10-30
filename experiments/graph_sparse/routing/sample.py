"""Sample edges from semantic bands and expand blocks to token indices."""
import torch


def sample_semantic_edges(bands: dict, near_k: int, medium_k: int, far_k: int,
                          random_k: int, causal_mask: torch.Tensor,
                          generator: torch.Generator = None,
                          similarity: torch.Tensor = None,
                          temperature: float = 0.0) -> torch.Tensor:
    """Sample block indices from each band for every query chunk.

    For each query chunk, samples near_k blocks from the "near" band, medium_k from "medium",
    far_k from "far", and random_k from all eligible candidates uniformly.
    When a band has fewer candidates than requested, all available are taken.

    With temperature > 0, sampling is weighted by softmax(similarity / temperature) within
    each band instead of uniform. temperature=0 means uniform random (the default).

    Args:
        bands: {"near": [Q, B], "medium": [Q, B], "far": [Q, B]} boolean masks.
        near_k, medium_k, far_k, random_k: blocks to sample from each band.
        causal_mask: [Q, B] for random sampling pool.
        generator: optional torch.Generator for reproducibility.
        similarity: optional [Q, B] scores for temperature-weighted sampling.
        temperature: > 0 enables similarity-weighted sampling within bands.

    Returns:
        [Q, total_blocks] int64 block indices, padded with -1.
    """
    Q, B = causal_mask.shape
    total = near_k + medium_k + far_k + random_k
    result = torch.full((Q, total), -1, dtype=torch.long, device=causal_mask.device)

    for q in range(Q):
        col = 0
        for band_name, budget in [("near", near_k), ("medium", medium_k), ("far", far_k)]:
            if budget == 0:
                continue
            mask = bands[band_name][q]
            candidates = mask.nonzero(as_tuple=True)[0]
            if candidates.numel() == 0:
                col += budget
                continue
            take = min(budget, candidates.numel())
            if temperature > 0 and similarity is not None:
                weights = (similarity[q, candidates] / temperature).softmax(0)
                perm = torch.multinomial(weights, take, replacement=False, generator=generator)
            else:
                perm = torch.randperm(candidates.numel(), generator=generator, device=causal_mask.device)[:take]
            result[q, col:col + take] = candidates[perm]
            col += budget

        already = set(result[q, :col][result[q, :col] >= 0].tolist())
        eligible = causal_mask[q].clone()
        for idx in already:
            eligible[idx] = False
        rand_cands = eligible.nonzero(as_tuple=True)[0]
        if rand_cands.numel() > 0 and random_k > 0:
            take = min(random_k, rand_cands.numel())
            perm = torch.randperm(rand_cands.numel(), generator=generator, device=causal_mask.device)[:take]
            result[q, col:col + take] = rand_cands[perm]

    return result


def sample_semantic_edges_batched(bands: dict, near_k: int, medium_k: int, far_k: int,
                                  random_k: int, causal_mask: torch.Tensor,
                                  generator: torch.Generator = None,
                                  similarity: torch.Tensor = None,
                                  temperature: float = 0.0) -> torch.Tensor:
    """Vectorized sampling using Gumbel-top-k (no Python loop over Q).

    Draws from each band in parallel by adding Gumbel noise to a 0/−inf mask and taking topk.
    With temperature > 0, noise is scaled so higher-similarity blocks are more likely selected.
    """
    Q, B = causal_mask.shape
    total = near_k + medium_k + far_k + random_k
    result = torch.full((Q, total), -1, dtype=torch.long, device=causal_mask.device)

    def _gumbel_topk(mask: torch.Tensor, k: int, band_sim: bool = False) -> torch.Tensor:
        if k == 0:
            return torch.empty(Q, 0, dtype=torch.long, device=mask.device)
        if temperature > 0 and similarity is not None and band_sim:
            log_probs = torch.where(mask, similarity / temperature, torch.tensor(float("-inf"), device=mask.device))
        else:
            log_probs = torch.where(mask, torch.zeros((), device=mask.device),
                                    torch.tensor(float("-inf"), device=mask.device))
        noise = torch.empty_like(log_probs).exponential_(1.0, generator=generator).clamp(min=1e-20).log().neg()
        noisy = log_probs + noise
        actual_k = min(k, B)
        _, topk_idx = noisy.topk(actual_k, dim=1)
        valid = mask.gather(1, topk_idx)
        return topk_idx.where(valid, torch.tensor(-1, dtype=torch.long, device=mask.device))

    col = 0
    for band_name, budget in [("near", near_k), ("medium", medium_k), ("far", far_k)]:
        sel = _gumbel_topk(bands[band_name], budget, band_sim=True)
        w = sel.shape[1]
        result[:, col:col + w] = sel
        col += budget

    already = result[:, :col]
    used = torch.zeros(Q, B, dtype=torch.bool, device=causal_mask.device)
    valid_already = already.clamp(min=0)
    used.scatter_(1, valid_already, already >= 0)
    rand_pool = causal_mask & ~used
    sel = _gumbel_topk(rand_pool, random_k, band_sim=False)
    w = sel.shape[1]
    result[:, col:col + w] = sel

    return result


def expand_blocks_to_indices(block_indices: torch.Tensor, key_block_size: int,
                             seq_len: int) -> torch.Tensor:
    """Expand selected block indices to individual token positions.

    Args:
        block_indices: [Q, num_blocks] block indices (-1 = padding).
        key_block_size: tokens per block.
        seq_len: total sequence length (for clamping the last block).

    Returns:
        [Q, num_blocks * key_block_size] token positions, -1 for invalid.
    """
    Q, num_blocks = block_indices.shape
    offsets = torch.arange(key_block_size, device=block_indices.device)
    base = block_indices[:, :, None] * key_block_size + offsets[None, None, :]
    base = base.view(Q, num_blocks * key_block_size)
    invalid = (block_indices[:, :, None].expand(-1, -1, key_block_size).reshape(Q, -1) < 0)
    base[invalid] = -1
    base[(base >= seq_len) & ~invalid] = -1
    return base


def deduplicate_and_sort(indices: torch.Tensor) -> torch.Tensor:
    """Sort token indices per row, pushing -1 to the end and removing duplicates.

    Args:
        indices: [Q, K] token positions (-1 = invalid).

    Returns:
        [Q, K] sorted, deduplicated indices with -1 padding at the end.
    """
    Q, K = indices.shape
    result = torch.full_like(indices, -1)
    for q in range(Q):
        row = indices[q]
        valid = row[row >= 0].unique(sorted=True)
        result[q, :valid.numel()] = valid
    return result


def deduplicate_and_sort_batched(indices: torch.Tensor) -> torch.Tensor:
    """Vectorized dedup+sort: scatter into a wide mask, then compact. No Python loops.

    Works well when seq_len is moderate; for very long sequences, the per-row
    loop version may use less memory.

    Complexity: O(Q * max_val) space for the membership mask plus O(Q * max_val * log(max_val))
    for the sort. For 128K sequences with block_size=64 this is O(Q * 2048).
    """
    Q, K = indices.shape
    if K == 0:
        return indices
    max_val = indices.max().item()
    if max_val < 0:
        return torch.full_like(indices, -1)
    width = max_val + 1
    member = torch.zeros(Q, width, dtype=torch.bool, device=indices.device)
    valid = indices >= 0
    if valid.any():
        flat_q = torch.arange(Q, device=indices.device)[:, None].expand_as(indices)[valid]
        flat_idx = indices[valid]
        member[flat_q, flat_idx] = True
    counts = member.sum(1)
    all_positions = torch.arange(width, device=indices.device).expand(Q, -1)
    masked = all_positions.where(member, torch.tensor(width, dtype=torch.long, device=indices.device))
    sorted_pos, _ = masked.sort(dim=1)
    result = torch.full((Q, K), -1, dtype=indices.dtype, device=indices.device)
    col_idx = torch.arange(K, device=indices.device).expand(Q, -1)
    fill = col_idx < counts[:, None].clamp(max=K)
    result[fill] = sorted_pos[:, :K][fill].to(indices.dtype)
    return result
