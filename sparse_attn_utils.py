"""Efficient sparse attention helpers (no BH x T x T expand)."""

from typing import Optional

import torch
import torch.nn.functional as F


def effective_top_k(top_k: int, seq_len: int, min_k: int = 32, ratio: int = 8) -> int:
    """Scale k down on short sequences so sparsity is meaningful."""
    return min(top_k, max(min_k, seq_len // ratio))


def token_mask_1d(attention_mask, bsz: int, src_len: int, device) -> Optional[torch.Tensor]:
    """[B, src_len] bool mask from HF attention_mask.

    Handles 2D padding masks [B, T], 3D [B, 1, T], and 4D causal+padding
    masks [B, 1, T, T]. For 4D causal masks, the padding component is
    extracted by checking which key positions are valid for ANY query
    position (``.any(dim=1)``), since the causal triangle allows the
    last query to attend to all valid keys.
    """
    if attention_mask is None:
        return None
    am_bool = (
        attention_mask
        if attention_mask.dtype == torch.bool
        else (attention_mask > 0)
    )
    if am_bool.dim() == 4:
        # Causal+padding mask [B, 1, T, T]: extract padding mask via any()
        return am_bool[:, 0, :, :].any(dim=1)  # [B, T]
    if am_bool.dim() == 3:
        return am_bool[:, 0, :]
    return am_bool


def apply_token_mask_scores(scores: torch.Tensor, token_mask, bsz: int, num_heads: int) -> torch.Tensor:
    """Mask scores [BH, ...] with [B, src_len] padding mask."""
    if token_mask is None:
        return scores
    BH = scores.size(0)
    src_len = token_mask.size(-1)
    if scores.dim() == 2:
        # [BH, src_len]
        me = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
        return scores.masked_fill(~me, -1e9)
    # [BH, Tq, src_len] or [BH, Tq, k]
    tgt_len = scores.size(1)
    me = token_mask.unsqueeze(1).unsqueeze(1).expand(bsz, num_heads, tgt_len, src_len)
    me = me.reshape(BH, tgt_len, src_len)
    return scores.masked_fill(~me, -1e9)


def _gather_kv(K: torch.Tensor, V: torch.Tensor, indices: torch.Tensor):
    """Advanced index gather: K [BH,S,d], indices [BH,T,k] -> [BH,T,k,d]."""
    BH = K.size(0)
    bh = torch.arange(BH, device=K.device).view(BH, 1, 1)
    K_sel = K[bh, indices, :]
    V_sel = V[bh, indices, :]
    return K_sel, V_sel


def sparse_attention_from_indices(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    indices: torch.Tensor,
    dropout: float,
    training: bool,
    token_mask=None,
    bsz: int = 1,
    num_heads: int = 12,
) -> torch.Tensor:
    """
    Q, K, V: [BH, T, d]
    indices: [BH, T, k] or [BH, k] (broadcast over T)
    Returns [BH, T, d]
    """
    BH, tgt_len, d = Q.shape
    src_len = K.size(1)

    if indices.dim() == 2:
        indices = indices.unsqueeze(1).expand(-1, tgt_len, -1)

    k = indices.size(-1)
    K_sel, V_sel = _gather_kv(K, V, indices)

    scores = torch.matmul(Q.unsqueeze(2), K_sel.transpose(-1, -2)).squeeze(2)

    if token_mask is not None:
        am = token_mask.unsqueeze(1).unsqueeze(1).expand(bsz, num_heads, tgt_len, src_len)
        am = am.reshape(BH, tgt_len, src_len)
        allowed = torch.gather(am, 2, indices)
        scores = scores.masked_fill(~allowed, -1e9)

    attn = F.softmax(scores, dim=-1)
    attn = F.dropout(attn, p=dropout, training=training)
    return torch.bmm(
        attn.reshape(BH * tgt_len, 1, k),
        V_sel.reshape(BH * tgt_len, k, d),
    ).reshape(BH, tgt_len, d)


def sparse_attention_head_shared(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    indices: torch.Tensor,
    dropout: float,
    training: bool,
    token_mask=None,
    bsz: int = 1,
    num_heads: int = 12,
    is_causal: bool = False,
    key_positions=None,
) -> torch.Tensor:
    """indices [BH, k] shared across all query positions.

    When ``is_causal=True``, masks out keys at positions > query position.
    ``key_positions`` [BH, src_len] maps an index into K to its original
    position in the full sequence (needed when K is a subset, e.g. after
    token dropping). If None, indices are assumed to be original positions.
    """
    BH, tgt_len, d = Q.shape
    src_len = K.size(1)
    k = indices.size(-1)
    idx = indices.unsqueeze(-1).expand(-1, -1, d)
    K_sel = torch.gather(K, 1, idx)
    V_sel = torch.gather(V, 1, idx)
    scores = torch.bmm(Q, K_sel.transpose(1, 2))  # [BH, T, k]
    if token_mask is not None:
        am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
        allowed = torch.gather(am, 1, indices)  # [BH, k]
        scores = scores.masked_fill(~allowed.unsqueeze(1), -1e9)
    if is_causal:
        # Get original positions of selected keys
        if key_positions is not None:
            idx_pos = torch.gather(key_positions, 1, indices)  # [BH, k]
        else:
            idx_pos = indices  # indices are already original positions
        idx_pos = idx_pos.unsqueeze(1)  # [BH, 1, k]
        q_pos = torch.arange(tgt_len, device=Q.device).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        causal_allowed = idx_pos <= q_pos  # [BH, T, k]
        scores = scores.masked_fill(~causal_allowed, -1e9)
    attn = F.softmax(scores, dim=-1)
    attn = F.dropout(attn, p=dropout, training=training)
    return torch.bmm(attn, V_sel)


def head_shared_topk_indices(
    Q_low: torch.Tensor,
    K_low: torch.Tensor,
    top_k: int,
    token_mask=None,
    bsz: int = 1,
    num_heads: int = 12,
) -> torch.Tensor:
    """
    One top-k set per head (not per query): [BH, k].
    Routing cost O(BH * src_len * d_low), not O(BH * T * src_len).
    """
    BH, tgt_len, d_low = Q_low.shape
    src_len = K_low.size(1)
    q_mean = Q_low.mean(dim=1, keepdim=True)
    rough = torch.bmm(q_mean, K_low.transpose(1, 2)).squeeze(1) / (d_low ** 0.5)
    rough = apply_token_mask_scores(rough, token_mask, bsz, num_heads)
    k = min(top_k, src_len)
    _, idx = torch.topk(rough, k=k, dim=-1)
    return idx


def gather_attention_triton_or_none(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    indices: torch.Tensor,
    attention_mask,
    bsz: int,
    num_heads: int,
    use_triton: bool,
    training: bool,
    scale: float = 1.0,
) -> Optional[torch.Tensor]:
    """Inference-only fused gather attention; returns None to signal PyTorch fallback.

    ``indices`` may be head-shared ``[BH, M]`` (broadcast across queries) or
    per-query ``[BH, T, M]``. Q is assumed pre-scaled, so ``scale`` defaults to 1.0.

    At inference this uses the forward-only fused kernel. During training, if the
    experimental training-kernel flag is enabled, it uses the autograd-capable
    kernel (which carries gradients); otherwise it returns None to fall back.
    """
    if not use_triton:
        return None
    try:
        from kernels import (
            build_gather_key_mask,
            gather_attention_autograd,
            should_use_train_kernel,
            should_use_triton,
            sparse_gather_attention,
        )
    except ImportError:
        return None

    tgt_len = Q.size(1)

    def _token_idx():
        idx = indices
        if idx.dim() == 2:
            idx = idx.unsqueeze(1).expand(-1, tgt_len, -1)
        return idx

    if training:
        if not should_use_train_kernel(use_triton, Q):
            return None
        try:
            token_idx = _token_idx()
            key_mask = build_gather_key_mask(attention_mask, bsz, num_heads, tgt_len, token_idx)
            return gather_attention_autograd(Q, K, V, token_idx, key_mask, scale=scale)
        except Exception:
            return None

    if not should_use_triton(use_triton, Q, training=training):
        return None
    try:
        token_idx = _token_idx()
        key_mask = build_gather_key_mask(attention_mask, bsz, num_heads, tgt_len, token_idx)
        return sparse_gather_attention(Q, K, V, token_idx, key_mask, scale=scale)
    except Exception:
        return None


def sdpa_head_shared_or_none(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    indices: torch.Tensor,
    attention_mask,
    bsz: int,
    num_heads: int,
    enabled: bool,
    training: bool,
    is_causal: bool = False,
    key_positions=None,
) -> Optional[torch.Tensor]:
    """Fused attention over a head-shared key set via F.scaled_dot_product_attention.

    ``indices`` is ``[BH, k]`` (one key set per head, shared across queries). Q is
    pre-scaled, so ``scale=1.0``. Inference-only; returns None to fall back.

    When ``is_causal=True``, applies a causal mask so query at position q only
    attends to selected keys at positions <= q.
    ``key_positions`` [BH, src_len] maps an index into K to its original
    position (needed when K is a subset). If None, indices are original positions.
    """
    if not enabled or training:
        return None
    BH, tgt_len, d = Q.shape
    src_len = K.size(1)
    idx = indices.unsqueeze(-1).expand(-1, -1, d)
    K_sel = torch.gather(K, 1, idx)
    V_sel = torch.gather(V, 1, idx)

    # Build attention mask [BH, T, k] (True means "attend" for bool masks)
    attn_mask = None
    token_mask = token_mask_1d(attention_mask, bsz, src_len, Q.device)
    if token_mask is not None:
        am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
        allowed = torch.gather(am, 1, indices)  # [BH, k] bool
        attn_mask = allowed.unsqueeze(1)  # [BH, 1, k], broadcast over queries

    if is_causal:
        # Get original positions of selected keys
        if key_positions is not None:
            idx_pos = torch.gather(key_positions, 1, indices)  # [BH, k]
        else:
            idx_pos = indices
        idx_pos = idx_pos.unsqueeze(1)  # [BH, 1, k]
        q_pos = torch.arange(tgt_len, device=Q.device).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        causal_allowed = idx_pos <= q_pos  # [BH, T, k]
        if attn_mask is not None:
            attn_mask = attn_mask & causal_allowed
        else:
            attn_mask = causal_allowed

    return F.scaled_dot_product_attention(
        Q, K_sel, V_sel, attn_mask=attn_mask, scale=1.0,
    )


def sdpa_dense_or_none(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attention_mask,
    bsz: int,
    num_heads: int,
    enabled: bool,
    training: bool,
) -> Optional[torch.Tensor]:
    """Fused full attention via F.scaled_dot_product_attention (Flash/mem-efficient).

    Q is pre-scaled, so ``scale=1.0``. Inference-only; returns None to fall back.
    """
    if not enabled or training:
        return None
    BH = Q.size(0)
    src_len = K.size(1)
    attn_mask = None
    token_mask = token_mask_1d(attention_mask, bsz, src_len, Q.device)
    if token_mask is not None:
        am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
        attn_mask = am.unsqueeze(1)  # [BH, 1, src_len], broadcast over queries
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, scale=1.0)


def dense_self_attention(
    Q, K, V, attention_mask, bsz, num_heads, dropout, training, is_causal=False
) -> torch.Tensor:
    """Standard dense attention [BH, T, d] using fused SDPA.

    Uses torch.nn.functional.scaled_dot_product_attention to avoid
    materializing the full [BH, T, T] score/mask tensors, which OOMs
    on long sequences (e.g. 4096 tokens x 256 heads = 8GB just for
    scores on a 40GB MIG slice). SDPA's FlashAttention backend computes
    the same result in O(T) memory.

    Q/K/V arrive as [BH, T, d] (batch*heads merged). We reshape to
    [B, H, T, d] for SDPA, which is the shape FlashAttention expects.
    Q is already pre-scaled by the caller, so we pass scale=1.0 to
    avoid double-scaling.

    If ``is_causal=True``, applies a causal mask so position i can only
    attend to positions <= i. This matches how Llama was pretrained and
    is critical for last-token pooling to work — the last token's
    representation becomes a summary of everything before it.
    """
    BH, tgt_len, d = Q.shape
    src_len = K.size(1)
    Q4d = Q.reshape(bsz, num_heads, tgt_len, d)
    K4d = K.reshape(bsz, num_heads, src_len, d)
    V4d = V.reshape(bsz, num_heads, src_len, d)

    token_mask = token_mask_1d(attention_mask, bsz, src_len, Q.device)
    if token_mask is not None:
        # [B, T] -> [B, 1, 1, T] bool mask (True = padding, don't attend)
        attn_mask = ~token_mask.unsqueeze(1).unsqueeze(2)
    else:
        attn_mask = None

    out = F.scaled_dot_product_attention(
        Q4d, K4d, V4d, attn_mask=attn_mask, is_causal=is_causal, scale=1.0,
        dropout_p=dropout if training else 0.0,
    )
    return out.reshape(BH, tgt_len, d)


def causal_sparse_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    routed_indices: torch.Tensor,
    local_window: int = 256,
    token_mask=None,
    bsz: int = 1,
    num_heads: int = 32,
    query_chunk: int = 256,
) -> torch.Tensor:
    """Causal sparse attention with local window + head-shared routed keys.

    Each query position q attends to:
      1. Local window: keys at positions [max(0, q-W+1), q]  (always valid)
      2. Routed keys: head-shared top-k keys (causal-masked, may be future → masked out)

    This ensures every query has at least W valid keys, even if all routed
    keys are in the future. The routed keys provide long-range content-based
    retrieval, while the local window provides causal context.

    Total keys per query: W + k (with possible overlap, deduplicated by softmax masking).
    Complexity: O(N * (W + k)) = O(N) when W and k are fixed constants.

    Args:
        Q, K, V: [BH, T, d] — pre-projected, RoPE'd, GQA-expanded
        routed_indices: [BH, k] — head-shared key indices from routing
        local_window: W — size of causal local window per query
        token_mask: [B, src_len] bool padding mask (optional)
        bsz, num_heads: for unflattening
        query_chunk: process this many queries at a time to limit memory

    Returns: [BH, T, d]
    """
    BH, tgt_len, d = Q.shape
    src_len = K.size(1)
    k = routed_indices.size(-1)
    W = local_window
    M = W + k

    # Always execute the gather-based sparse path, including short sequences.
    # This keeps experiment models sparse-only instead of silently switching to SDPA.

    # Precompute local window offsets (shared across all chunks)
    local_offsets = torch.arange(W, device=Q.device)  # [W]

    # Precompute padding mask if needed
    if token_mask is not None:
        am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
    else:
        am = None

    # Process queries in chunks to limit peak memory
    # Build indices on-the-fly per chunk to avoid [BH, T, M] tensor
    out_chunks = []
    for q_start in range(0, tgt_len, query_chunk):
        q_end = min(q_start + query_chunk, tgt_len)
        chunk_len = q_end - q_start
        Q_chunk = Q[:, q_start:q_end, :]  # [BH, chunk, d]

        # Build local window indices for this chunk: [chunk, W]
        q_positions = torch.arange(q_start, q_end, device=Q.device)
        local_start = (q_positions - W + 1).clamp(min=0)
        local_idx_chunk = (local_start.unsqueeze(1) + local_offsets.unsqueeze(0)).clamp(max=src_len - 1)

        # Build full index set for this chunk: [BH, chunk, W+k]
        local_expanded = local_idx_chunk.unsqueeze(0).expand(BH, -1, -1)  # [BH, chunk, W]
        routed_expanded = routed_indices.unsqueeze(1).expand(-1, chunk_len, -1)  # [BH, chunk, k]
        idx_chunk = torch.cat([local_expanded, routed_expanded], dim=-1)  # [BH, chunk, M]

        # Gather K, V: [BH, chunk, M, d]
        bh_arange = torch.arange(BH, device=Q.device).view(BH, 1, 1)
        k_sel = K[bh_arange, idx_chunk, :]
        v_sel = V[bh_arange, idx_chunk, :]

        # Scores: [BH, chunk, M]
        scores = torch.matmul(Q_chunk.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2)

        # Causal mask: key position must be <= query position
        q_pos = q_positions.unsqueeze(0).unsqueeze(-1)  # [1, chunk, 1]
        causal_allowed = idx_chunk <= q_pos  # [BH, chunk, M]
        scores = scores.masked_fill(~causal_allowed, torch.finfo(scores.dtype).min)

        # Padding mask
        if am is not None:
            allowed_pad = torch.gather(
                am.unsqueeze(1).expand(-1, chunk_len, -1), 2, idx_chunk
            )
            scores = scores.masked_fill(~allowed_pad, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        chunk_out = torch.bmm(
            attn.reshape(BH * chunk_len, 1, M),
            v_sel.reshape(BH * chunk_len, M, d),
        ).reshape(BH, chunk_len, d)
        out_chunks.append(chunk_out)

    return torch.cat(out_chunks, dim=1)


def last_query_topk_indices(
    Q_low: torch.Tensor,
    K_low: torch.Tensor,
    top_k: int,
    token_mask=None,
    bsz: int = 1,
    num_heads: int = 32,
) -> torch.Tensor:
    """Select top-k keys using the LAST query position (causal-safe routing).

    During causal generation, the last query position is the "current" token
    being generated. It can attend to all past keys. Using it for routing
    gives a key set that's valid for the last position, and we apply causal
    masking for earlier positions during attention.

    This is O(N * d_low) for routing — truly linear.

    Args:
        Q_low: [BH, T, d_low] — low-rank query projection
        K_low: [BH, S, d_low] — low-rank key projection
        top_k: number of keys to select per head
        token_mask: [B, S] padding mask
        bsz, num_heads: for unflattening

    Returns: [BH, k] key indices per head
    """
    BH, tgt_len, d_low = Q_low.shape
    src_len = K_low.size(1)

    # Use last query position for routing
    q_last = Q_low[:, -1:, :]  # [BH, 1, d_low]
    scores = torch.bmm(q_last, K_low.transpose(1, 2)).squeeze(1) / (d_low ** 0.5)  # [BH, S]

    if token_mask is not None:
        am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
        scores = scores.masked_fill(~am, torch.finfo(scores.dtype).min)

    k = min(top_k, src_len)
    _, idx = torch.topk(scores, k=k, dim=-1)  # [BH, k]
    return idx


def multi_query_topk_indices(
    Q_low: torch.Tensor,
    K_low: torch.Tensor,
    top_k: int,
    num_queries: int = 4,
    token_mask=None,
    bsz: int = 1,
    num_heads: int = 32,
) -> torch.Tensor:
    """Select top-k keys using the LAST N query positions (causal-safe routing).

    Uses the last ``num_queries`` query positions to compute routing scores.
    For each key, takes the MAX score across the N queries (equivalent to
    union of per-query top-k sets, then re-ranking by best score).  This
    catches the needle even if one query position's low-rank similarity
    misses it — the max across N queries has higher recall.

    Cost: O(N_queries * N * d_low) for routing — still linear in N.

    Args:
        Q_low: [BH, T, d_low] — low-rank query projection
        K_low: [BH, S, d_low] — low-rank key projection
        top_k: number of keys to select per head
        num_queries: number of last query positions to use for routing
        token_mask: [B, S] padding mask
        bsz, num_heads: for unflattening

    Returns: [BH, k] key indices per head
    """
    BH, tgt_len, d_low = Q_low.shape
    src_len = K_low.size(1)

    n = min(num_queries, tgt_len)
    # Use last n query positions: [BH, n, d_low]
    q_last_n = Q_low[:, -n:, :]
    # Scores: [BH, n, S]
    scores = torch.bmm(q_last_n, K_low.transpose(1, 2)) / (d_low ** 0.5)

    if token_mask is not None:
        am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
        scores = scores.masked_fill(~am.unsqueeze(1), torch.finfo(scores.dtype).min)

    # Max score across query positions: [BH, S]
    best_scores = scores.max(dim=1).values

    k = min(top_k, src_len)
    _, best_idx = torch.topk(best_scores, k=k, dim=-1)  # [BH, k]
    return best_idx


def novelty_topk_indices(
    K: torch.Tensor,
    novelty_ratio: float = 0.02,
    novelty_window: int = 64,
    min_k: int = 64,
    token_mask=None,
    bsz: int = 1,
    num_heads: int = 32,
) -> torch.Tensor:
    """Select keys by local novelty — how different each key is from its neighborhood.

    The needle (a random number) is an outlier in the key distribution.
    This routing signal doesn't compete with distractors in a ranked QK list,
    so it doesn't have the phase-transition problem of top-k routing.

    Budget scales automatically with sequence length:
      4K  -> ~80 tokens  (0.02 * 4096)
      64K -> ~1280 tokens
      128K -> ~2560 tokens

    Args:
        K: [BH, S, d] — full-dim keys (not low-rank)
        novelty_ratio: fraction of tokens to select (budget = ratio * seq_len)
        novelty_window: neighborhood size for computing local distinctiveness
        min_k: minimum budget for short sequences
        token_mask: [B, S] padding mask
        bsz, num_heads: for unflattening

    Returns: [BH, k] key indices per head, sorted by position
    """
    BH, src_len, d = K.shape

    # Compute local neighborhood mean using unfold.
    # Pad K so every position has a full neighborhood, then unfold with
    # stride=1.  Padding by (novelty_window-1) on the left and 0 on the right
    # would misalign; instead pad symmetrically and crop the extra window.
    half_w = novelty_window // 2
    # Pad left by half_w, right by half_w.  After unfold we get
    # (S + 2*half_w) - novelty_window + 1 = S + 1 windows.  Crop to S.
    K_padded = F.pad(K, (0, 0, half_w, half_w), mode="replicate")  # [BH, S+2w, d]
    K_windows = K_padded.unfold(1, novelty_window, 1)  # [BH, S+1, d, w]
    K_windows = K_windows[:, :src_len, :, :]  # crop to [BH, S, d, w]
    K_windows = K_windows.transpose(-1, -2)  # [BH, S, w, d]

    # Local mean (includes self — that's fine, novelty is still dominated
    # by the outlier when the window is small relative to seq_len)
    neighborhood_mean = K_windows.mean(dim=2)  # [BH, S, d]

    # Novelty = distance from local mean
    novelty = (K - neighborhood_mean).norm(dim=-1)  # [BH, S]

    if token_mask is not None:
        am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
        novelty = novelty.masked_fill(~am, torch.finfo(novelty.dtype).min)

    # Budget scales with sequence length — no hard-coded number
    k = max(min_k, int(src_len * novelty_ratio))
    k = min(k, src_len)

    _, idx = torch.topk(novelty, k=k, dim=-1)  # [BH, k]
    # Sort by position for causal kernel compatibility
    idx, _ = idx.sort(dim=-1)
    return idx


def hybrid_topk_indices(
    Q_low: torch.Tensor,
    K: torch.Tensor,
    K_low: torch.Tensor,
    top_k: int,
    novelty_ratio: float = 0.01,
    novelty_window: int = 64,
    num_queries: int = 4,
    token_mask=None,
    bsz: int = 1,
    num_heads: int = 32,
) -> torch.Tensor:
    """Scale-invariant routing: QK similarity + novelty boost, ratio-based budget.

    No hard-coded budget. The number of selected tokens is ``ratio * seq_len``,
    which grows automatically with context length.

    The needle is found via two complementary signals:
    1. QK similarity (query-aware — finds semantically relevant tokens)
    2. Novelty (content-distinctiveness — the needle is an outlier)

    Combined score = normalized QK + novelty_weight * normalized novelty.
    Select top (ratio * seq_len) by combined score.

    Args:
        Q_low: [BH, T, d_low] — low-rank queries
        K: [BH, S, d] — full-dim keys for novelty
        K_low: [BH, S, d_low] — low-rank keys for QK
        top_k: ignored (kept for API compat). Budget is ratio-based.
        novelty_ratio: fraction of seq_len to select (the budget)
        novelty_window: neighborhood size for novelty computation
        num_queries: multi-query positions for QK
        token_mask: [B, S] padding mask
        bsz, num_heads: for unflattening

    Returns: [BH, k] key indices per head, sorted by position
    """
    BH, tgt_len, d_low = Q_low.shape
    src_len = K.size(1)

    # --- QK scores (query-aware, multi-query) ---
    q_last_n = Q_low[:, -min(num_queries, tgt_len):, :]  # [BH, nq, d_low]
    qk_scores = torch.bmm(q_last_n, K_low.transpose(1, 2)).max(dim=1).values  # [BH, S]
    qk_scores = qk_scores / (d_low ** 0.5)

    # --- Novelty scores (content-distinctiveness) ---
    half_w = novelty_window // 2
    K_padded = F.pad(K, (0, 0, half_w, half_w), mode="replicate")
    K_windows = K_padded.unfold(1, novelty_window, 1)[:, :src_len, :, :]
    K_windows = K_windows.transpose(-1, -2)  # [BH, S, w, d]
    neighborhood_mean = K_windows.mean(dim=2)
    novelty_scores = (K - neighborhood_mean).norm(dim=-1)  # [BH, S]

    # --- Normalize both to [0, 1] per head so they're comparable ---
    qk_norm = qk_scores - qk_scores.amin(dim=-1, keepdim=True)
    qk_norm = qk_norm / (qk_norm.amax(dim=-1, keepdim=True) + 1e-6)
    nov_norm = novelty_scores - novelty_scores.amin(dim=-1, keepdim=True)
    nov_norm = nov_norm / (nov_norm.amax(dim=-1, keepdim=True) + 1e-6)

    # --- Combined score: QK primary, novelty as boost ---
    combined = qk_norm + nov_norm  # [BH, S]

    if token_mask is not None:
        am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
        combined = combined.masked_fill(~am, torch.finfo(combined.dtype).min)

    # --- Ratio-based budget: scales with sequence length ---
    k = max(64, int(src_len * novelty_ratio))
    k = min(k, src_len)
    _, best_idx = torch.topk(combined, k=k, dim=-1)
    best_idx, _ = best_idx.sort(dim=-1)
    return best_idx


def last_query_block_topk_indices(
    Q_low: torch.Tensor,
    K_low: torch.Tensor,
    block_size: int,
    num_blocks: int,
    token_mask=None,
    bsz: int = 1,
    num_heads: int = 32,
) -> torch.Tensor:
    """Causal-safe block-level top-k routing using the LAST query position.

    Scores blocks by mean-pooling keys within each block and comparing to the
    last query position.  Selects the top ``num_blocks`` blocks, expands to
    token indices, and returns them sorted.

    Args:
        Q_low: [BH, T, d_low]
        K_low: [BH, S, d_low]
        block_size: tokens per block
        num_blocks: how many blocks to select
        token_mask: [B, S] padding mask
        bsz, num_heads: for unflattening

    Returns: [BH, k] token indices (sorted), where k = num_blocks * block_size
    """
    BH, tgt_len, d_low = Q_low.shape
    src_len = K_low.size(1)

    n_blocks_k = max(1, (src_len + block_size - 1) // block_size)
    usable_src = n_blocks_k * block_size
    K_used = F.pad(K_low, (0, 0, 0, usable_src - src_len))
    K_blocks = K_used.view(BH, n_blocks_k, block_size, d_low).mean(dim=2)  # [BH, n_blocks_k, d_low]

    # Use last query position for routing (causal-safe)
    q_last = Q_low[:, -1:, :]  # [BH, 1, d_low]
    block_scores = torch.bmm(q_last, K_blocks.transpose(1, 2)).squeeze(1) / (d_low ** 0.5)  # [BH, n_blocks_k]

    if token_mask is not None:
        padded_mask = F.pad(token_mask, (0, usable_src - src_len), value=False)
        block_mask = padded_mask.view(bsz, n_blocks_k, block_size).any(dim=-1)
        block_mask = block_mask.unsqueeze(1).expand(bsz, num_heads, n_blocks_k).reshape(BH, n_blocks_k)
        block_scores = block_scores.masked_fill(~block_mask, torch.finfo(block_scores.dtype).min)

    M = min(num_blocks, n_blocks_k)
    _, top_blocks = torch.topk(block_scores, k=M, dim=-1)  # [BH, M]

    # Expand block indices to token indices
    offs = torch.arange(block_size, device=Q_low.device)
    base = top_blocks.unsqueeze(-1) * block_size  # [BH, M, 1]
    top_idx = (base + offs.view(1, 1, -1)).reshape(BH, M * block_size)
    top_idx = top_idx.clamp(max=src_len - 1)
    top_idx, _ = torch.sort(top_idx, dim=-1)
    return top_idx


def causal_linear_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    token_mask=None,
    bsz: int = 1,
    num_heads: int = 32,
) -> torch.Tensor:
    """Causal ELU-feature linear attention via cumulative sums: O(T * d^2).

    For each position t, the output is:
        O_t = phi(Q_t) * (sum_{s<=t} phi(K_s) * V_s^T) / (phi(Q_t) * sum_{s<=t} phi(K_s))

    where phi(x) = ELU(x) + 1.

    Args:
        Q, K, V: [BH, T, d]
        token_mask: [B, S] padding mask
        bsz, num_heads: for unflattening

    Returns: [BH, T, d]
    """
    BH, tgt_len, d = Q.shape
    src_len = K.size(1)

    Q_f = F.elu(Q) + 1.0  # [BH, T, d]
    K_f = F.elu(K) + 1.0  # [BH, T, d]

    if token_mask is not None:
        am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
        K_f = K_f * am.unsqueeze(-1)

    # Causal cumulative sums: S_t = sum_{s<=t} phi(K_s) * V_s^T
    # KV_t = phi(K_t).unsqueeze(-1) * V_t.unsqueeze(-2) -> [BH, T, d, d]
    # But that's d^2 per step. Instead, compute cumsum of phi(K) * V^T:
    # KV_cum[t] = sum_{s<=t} phi(K_s) outer V_s  -> [BH, T, d, d]
    # We compute this as: cumsum over dim=1 of (K_f.unsqueeze(-1) * V.unsqueeze(-2))
    # But d=128, d^2=16384 per head — too much memory for long sequences.
    # Instead, compute per-step: Num_t = Q_f[t] @ KV_cum[t], Den_t = Q_f[t] @ Z_cum[t]
    # Use chunked cumulative sum to limit memory.

    # Z_cum = cumsum of K_f: [BH, T, d]
    Z_cum = torch.cumsum(K_f, dim=1)

    # KV_cum = cumsum of (K_f outer V): [BH, T, d, d]
    # For d=128, this is BH * T * 16384 * 2 bytes — at T=131072, BH=32, that's 137GB. Too much.
    # Instead, compute the numerator incrementally:
    # Num_t = Q_f[t] @ (sum_{s<=t} K_f[s] outer V[s])
    #       = sum_{s<=t} (Q_f[t] @ K_f[s]) * V[s]
    # But that's O(T^2 * d).
    #
    # Better: use the "left chunk" trick. Process in chunks of C:
    # For each chunk, compute intra-chunk causal attention + contribution from all past chunks.
    # Past chunk contribution = Q_f[chunk] @ KV_past, where KV_past = sum of K_f outer V for past.
    # This is O(T * d^2) total with O(C * d^2) memory.

    chunk_size = min(1024, tgt_len)
    out_chunks = []
    KV_acc = torch.zeros(BH, d, d, device=Q.device, dtype=Q.dtype)  # running sum of K_f outer V
    Z_acc = torch.zeros(BH, 1, d, device=Q.device, dtype=Q.dtype)   # running sum of K_f

    for start in range(0, tgt_len, chunk_size):
        end = min(start + chunk_size, tgt_len)
        Q_c = Q_f[:, start:end, :]  # [BH, C, d]
        K_c = K_f[:, start:end, :]  # [BH, C, d]
        V_c = V[:, start:end, :]    # [BH, C, d]
        C = end - start

        # Contribution from past chunks: [BH, C, d]
        Num_past = torch.bmm(Q_c, KV_acc)  # [BH, C, d]
        Den_past = (Q_c * Z_acc).sum(dim=-1, keepdim=True)  # [BH, C, 1]

        # Intra-chunk causal contribution (cumsum within chunk)
        # KV_intra[t] = sum_{s<=t, s in chunk} K_f[s] outer V[s]
        K_outer_V = K_c.unsqueeze(-1) * V_c.unsqueeze(-2)  # [BH, C, d, d]
        KV_intra = torch.cumsum(K_outer_V, dim=1)  # [BH, C, d, d]
        Z_intra = torch.cumsum(K_c, dim=1)  # [BH, C, d]

        Num_intra = torch.einsum("bcd,bcde->bce", Q_c, KV_intra)  # [BH, C, d]
        Den_intra = (Q_c * Z_intra).sum(dim=-1, keepdim=True)  # [BH, C, 1]

        Num = Num_past + Num_intra
        Den = Den_past + Den_intra
        out_c = Num / (Den + 1e-6)
        out_chunks.append(out_c)

        # Update running sums
        KV_acc = KV_acc + K_outer_V.sum(dim=1)  # [BH, d, d]
        Z_acc = Z_acc + K_c.sum(dim=1, keepdim=True)  # [BH, 1, d]

    return torch.cat(out_chunks, dim=1)


def causal_sparse_attention_with_indices(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    all_idx: torch.Tensor,
    token_mask=None,
    bsz: int = 1,
    num_heads: int = 32,
    query_chunk: int = 256,
) -> torch.Tensor:
    """Causal sparse attention with per-query, per-head arbitrary key indices.

    Unlike ``causal_sparse_attention`` which adds a local window to head-shared
    routed indices, this function takes fully specified per-query indices
    (e.g. from strided patterns, anchor patterns, etc.) and applies causal
    masking on top.

    Args:
        Q, K, V: [BH, T, d]
        all_idx: [BH, T, M] — key indices per query position per head
        token_mask: [B, src_len] padding mask
        bsz, num_heads: for unflattening
        query_chunk: process this many queries at a time

    Returns: [BH, T, d]
    """
    BH, tgt_len, d = Q.shape
    src_len = K.size(1)
    M = all_idx.size(-1)

    if token_mask is not None:
        am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
    else:
        am = None

    out_chunks = []
    for q_start in range(0, tgt_len, query_chunk):
        q_end = min(q_start + query_chunk, tgt_len)
        Q_chunk = Q[:, q_start:q_end, :]  # [BH, chunk, d]
        idx_chunk = all_idx[:, q_start:q_end, :]  # [BH, chunk, M]
        chunk_len = q_end - q_start

        bh_arange = torch.arange(BH, device=Q.device).view(BH, 1, 1)
        k_sel = K[bh_arange, idx_chunk, :]  # [BH, chunk, M, d]
        v_sel = V[bh_arange, idx_chunk, :]

        scores = torch.matmul(Q_chunk.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2)  # [BH, chunk, M]

        # Causal mask: key position must be <= query position
        q_pos = torch.arange(q_start, q_end, device=Q.device).unsqueeze(0).unsqueeze(-1)  # [1, chunk, 1]
        causal_allowed = idx_chunk <= q_pos  # [BH, chunk, M]
        scores = scores.masked_fill(~causal_allowed, torch.finfo(scores.dtype).min)

        # Padding mask
        if am is not None:
            allowed_pad = torch.gather(
                am.unsqueeze(1).expand(-1, chunk_len, -1), 2, idx_chunk
            )
            scores = scores.masked_fill(~allowed_pad, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        chunk_out = torch.bmm(
            attn.reshape(BH * chunk_len, 1, M),
            v_sel.reshape(BH * chunk_len, M, d),
        ).reshape(BH, chunk_len, d)
        out_chunks.append(chunk_out)

    return torch.cat(out_chunks, dim=1)
