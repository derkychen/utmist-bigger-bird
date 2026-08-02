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
    from .kernels import (
        build_gather_key_mask,
        gather_attention_autograd,
        should_use_train_kernel,
        should_use_triton,
        sparse_gather_attention,
    )

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

    # Short-circuit: if sequence is short enough, use dense
    if src_len <= W + k:
        return dense_self_attention(
            Q, K, V, token_mask, bsz, num_heads, 0.0, False, is_causal=True,
        )

    # Build local window indices: [T, W]
    # For query q: positions max(0, q-W+1) to q
    positions = torch.arange(tgt_len, device=Q.device)
    local_start = (positions - W + 1).clamp(min=0)
    local_offsets = torch.arange(W, device=Q.device).unsqueeze(0)  # [1, W]
    local_idx = (local_start.unsqueeze(1) + local_offsets).clamp(max=src_len - 1)  # [T, W]

    # Expand to per-query indices: [BH, T, W+k]
    local_expanded = local_idx.unsqueeze(0).expand(BH, -1, -1)  # [BH, T, W]
    routed_expanded = routed_indices.unsqueeze(1).expand(-1, tgt_len, -1)  # [BH, T, k]
    all_idx = torch.cat([local_expanded, routed_expanded], dim=-1)  # [BH, T, W+k]
    M = all_idx.size(-1)

    # Precompute padding mask if needed
    if token_mask is not None:
        am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
    else:
        am = None

    # Process queries in chunks to limit peak memory
    out_chunks = []
    for q_start in range(0, tgt_len, query_chunk):
        q_end = min(q_start + query_chunk, tgt_len)
        Q_chunk = Q[:, q_start:q_end, :]  # [BH, chunk, d]
        idx_chunk = all_idx[:, q_start:q_end, :]  # [BH, chunk, M]
        chunk_len = q_end - q_start

        # Gather K, V: [BH, chunk, M, d]
        bh_arange = torch.arange(BH, device=Q.device).view(BH, 1, 1)
        k_sel = K[bh_arange, idx_chunk, :]
        v_sel = V[bh_arange, idx_chunk, :]

        # Scores: [BH, chunk, M]
        scores = torch.matmul(Q_chunk.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2)

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
