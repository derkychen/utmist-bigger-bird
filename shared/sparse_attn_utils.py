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
