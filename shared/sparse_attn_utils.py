"""Efficient sparse attention helpers (no BH x T x T expand)."""

from typing import Optional

import torch
import torch.nn.functional as F


def effective_top_k(top_k: int, seq_len: int, min_k: int = 32, ratio: int = 8) -> int:
    """Scale k down on short sequences so sparsity is meaningful."""
    return min(top_k, max(min_k, seq_len // ratio))


def token_mask_1d(attention_mask, bsz: int, src_len: int, device) -> Optional[torch.Tensor]:
    """[B, src_len] bool mask from HF attention_mask."""
    if attention_mask is None:
        return None
    am_bool = (
        attention_mask
        if attention_mask.dtype == torch.bool
        else (attention_mask > -1e-8)
    )
    if am_bool.dim() == 4:
        return am_bool[:, 0, 0, :]
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
) -> torch.Tensor:
    """indices [BH, k] shared across all query positions."""
    BH, tgt_len, d = Q.shape
    src_len = K.size(1)
    k = indices.size(-1)
    idx = indices.unsqueeze(-1).expand(-1, -1, d)
    K_sel = torch.gather(K, 1, idx)
    V_sel = torch.gather(V, 1, idx)
    scores = torch.bmm(Q, K_sel.transpose(1, 2))
    if token_mask is not None:
        am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
        allowed = torch.gather(am, 1, indices)
        scores = scores.masked_fill(~allowed.unsqueeze(1), -1e9)
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
) -> Optional[torch.Tensor]:
    """Fused attention over a head-shared key set via F.scaled_dot_product_attention.

    ``indices`` is ``[BH, k]`` (one key set per head, shared across queries). Q is
    pre-scaled, so ``scale=1.0``. Inference-only; returns None to fall back.
    """
    if not enabled or training:
        return None
    BH, _, d = Q.shape
    idx = indices.unsqueeze(-1).expand(-1, -1, d)
    K_sel = torch.gather(K, 1, idx)
    V_sel = torch.gather(V, 1, idx)
    attn_mask = None
    token_mask = token_mask_1d(attention_mask, bsz, K.size(1), Q.device)
    if token_mask is not None:
        src_len = token_mask.size(-1)
        am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
        allowed = torch.gather(am, 1, indices)  # [BH, k] bool
        attn_mask = allowed.unsqueeze(1)  # [BH, 1, k], broadcast over queries
    return F.scaled_dot_product_attention(Q, K_sel, V_sel, attn_mask=attn_mask, scale=1.0)


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
    Q, K, V, attention_mask, bsz, num_heads, dropout, training
) -> torch.Tensor:
    """Standard dense attention [BH, T, d]."""
    BH, tgt_len, _ = Q.shape
    src_len = K.size(1)
    scores = torch.bmm(Q, K.transpose(1, 2))
    token_mask = token_mask_1d(attention_mask, bsz, src_len, Q.device)
    scores = apply_token_mask_scores(scores, token_mask, bsz, num_heads)
    attn = F.softmax(scores, dim=-1)
    attn = F.dropout(attn, p=dropout, training=training)
    return torch.bmm(attn, V)
