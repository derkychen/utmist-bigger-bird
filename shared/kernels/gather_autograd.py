"""Training-capable gathered sparse attention (forward + Triton backward).

Unlike :func:`shared.kernels.gather.sparse_gather_attention` (inference-only,
no autograd), this exposes a ``torch.autograd.Function`` with a hand-written
Triton backward so the fused gather attention can be used during training.

Each query t attends to a fixed set of M gathered keys given by
``token_idx[BH, T, M]`` (indices into the K/V sequence). The forward saves the
per-row logsumexp; the backward recomputes the softmax probabilities and emits
dQ (one row per program) and dK/dV (scatter via atomics, since multiple queries
may select the same key index).

NOTE: attention dropout is *not* applied inside the kernel. It is only correct
to use this when the module's attention dropout is 0 (the BART-base default).
"""

from __future__ import annotations

import torch

from .common import MASK_SCORE_FP32, _TRITON_IMPORTED, ceil_pow2, triton_available

if _TRITON_IMPORTED:
    import triton
    import triton.language as tl

    @triton.jit
    def _gather_fwd_kernel(
        Q, K, V, Out, L, Idx, Mask,
        sqb, sqt, sqd,
        skb, sks, skd,
        svb, svs, svd,
        sob, sot, sod,
        slb, slt,
        sib, sit, sim,
        smb, smt, smm,
        seq_len,
        n_keys: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_MASK: tl.constexpr,
        SCALE,
        MASK_VAL: tl.constexpr,
    ):
        bh = tl.program_id(0)
        t = tl.program_id(1)
        d = tl.arange(0, BLOCK_D)
        dm = d < head_dim

        q = tl.load(Q + bh * sqb + t * sqt + d * sqd, mask=dm, other=0.0).to(tl.float32)

        m_i = MASK_VAL
        l_i = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for j in range(n_keys):
            idx = tl.load(Idx + bh * sib + t * sit + j * sim).to(tl.int32)
            idx = tl.minimum(tl.maximum(idx, 0), seq_len - 1)
            k = tl.load(K + bh * skb + idx * sks + d * skd, mask=dm, other=0.0).to(tl.float32)
            v = tl.load(V + bh * svb + idx * svs + d * svd, mask=dm, other=0.0).to(tl.float32)
            s = tl.sum(q * k, axis=0) * SCALE
            if HAS_MASK:
                keep = tl.load(Mask + bh * smb + t * smt + j * smm).to(tl.int1)
                s = tl.where(keep, s, MASK_VAL)
            m_new = tl.maximum(m_i, s)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + p
            acc = acc * alpha + p * v
            m_i = m_new

        l_safe = tl.where(l_i > 0.0, l_i, 1.0)
        out = acc / l_safe
        tl.store(Out + bh * sob + t * sot + d * sod, out, mask=dm)
        tl.store(L + bh * slb + t * slt, m_i + tl.log(l_safe))

    @triton.jit
    def _gather_bwd_kernel(
        Q, K, V, Out, L, dOut, dQ, dK, dV, Idx, Mask,
        sqb, sqt, sqd,
        skb, sks, skd,
        svb, svs, svd,
        sob, sot, sod,
        slb, slt,
        sdqb, sdqt, sdqd,
        sib, sit, sim,
        smb, smt, smm,
        seq_len,
        n_keys: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_MASK: tl.constexpr,
        SCALE,
        MASK_VAL: tl.constexpr,
    ):
        bh = tl.program_id(0)
        t = tl.program_id(1)
        d = tl.arange(0, BLOCK_D)
        dm = d < head_dim

        q = tl.load(Q + bh * sqb + t * sqt + d * sqd, mask=dm, other=0.0).to(tl.float32)
        dout = tl.load(dOut + bh * sob + t * sot + d * sod, mask=dm, other=0.0).to(tl.float32)
        out = tl.load(Out + bh * sob + t * sot + d * sod, mask=dm, other=0.0).to(tl.float32)
        L_i = tl.load(L + bh * slb + t * slt)
        delta = tl.sum(dout * out, axis=0)

        dq = tl.zeros([BLOCK_D], dtype=tl.float32)
        for j in range(n_keys):
            idx = tl.load(Idx + bh * sib + t * sit + j * sim).to(tl.int32)
            idx = tl.minimum(tl.maximum(idx, 0), seq_len - 1)
            k = tl.load(K + bh * skb + idx * sks + d * skd, mask=dm, other=0.0).to(tl.float32)
            v = tl.load(V + bh * svb + idx * svs + d * svd, mask=dm, other=0.0).to(tl.float32)
            s = tl.sum(q * k, axis=0) * SCALE
            if HAS_MASK:
                keep = tl.load(Mask + bh * smb + t * smt + j * smm).to(tl.int1)
                s = tl.where(keep, s, MASK_VAL)
            p = tl.exp(s - L_i)
            dp = tl.sum(dout * v, axis=0)
            ds = p * (dp - delta)
            dq += ds * SCALE * k
            tl.atomic_add(dV + bh * svb + idx * svs + d * svd, p * dout, mask=dm)
            tl.atomic_add(dK + bh * skb + idx * sks + d * skd, ds * SCALE * q, mask=dm)

        tl.store(dQ + bh * sdqb + t * sdqt + d * sdqd, dq, mask=dm)


def _fwd(q, k, v, token_idx, key_mask, scale):
    bh, tgt_len, head_dim = q.shape
    seq_len = k.size(1)
    n_keys = token_idx.size(-1)
    out = torch.empty_like(q)
    L = torch.empty(bh, tgt_len, device=q.device, dtype=torch.float32)
    block_d = ceil_pow2(head_dim)
    has_mask = key_mask is not None
    mask_i8 = key_mask.to(torch.int8) if has_mask else q
    _gather_fwd_kernel[(bh, tgt_len)](
        q, k, v, out, L, token_idx, mask_i8,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        L.stride(0), L.stride(1),
        token_idx.stride(0), token_idx.stride(1), token_idx.stride(2),
        mask_i8.stride(0) if has_mask else 0,
        mask_i8.stride(1) if has_mask else 0,
        mask_i8.stride(2) if has_mask else 0,
        seq_len,
        n_keys=n_keys,
        head_dim=head_dim,
        BLOCK_D=block_d,
        HAS_MASK=has_mask,
        SCALE=scale,
        MASK_VAL=MASK_SCORE_FP32,
        num_warps=4 if head_dim <= 64 else 8,
    )
    return out, L


def _bwd(q, k, v, out, L, dout, token_idx, key_mask, scale):
    bh, tgt_len, head_dim = q.shape
    seq_len = k.size(1)
    n_keys = token_idx.size(-1)
    dq = torch.zeros(bh, tgt_len, head_dim, device=q.device, dtype=torch.float32)
    dk = torch.zeros(bh, seq_len, head_dim, device=q.device, dtype=torch.float32)
    dv = torch.zeros(bh, seq_len, head_dim, device=q.device, dtype=torch.float32)
    block_d = ceil_pow2(head_dim)
    has_mask = key_mask is not None
    mask_i8 = key_mask.to(torch.int8) if has_mask else q
    _gather_bwd_kernel[(bh, tgt_len)](
        q, k, v, out, L, dout, dq, dk, dv, token_idx, mask_i8,
        q.stride(0), q.stride(1), q.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        L.stride(0), L.stride(1),
        dq.stride(0), dq.stride(1), dq.stride(2),
        token_idx.stride(0), token_idx.stride(1), token_idx.stride(2),
        mask_i8.stride(0) if has_mask else 0,
        mask_i8.stride(1) if has_mask else 0,
        mask_i8.stride(2) if has_mask else 0,
        seq_len,
        n_keys=n_keys,
        head_dim=head_dim,
        BLOCK_D=block_d,
        HAS_MASK=has_mask,
        SCALE=scale,
        MASK_VAL=MASK_SCORE_FP32,
        num_warps=4 if head_dim <= 64 else 8,
    )
    return dq, dk, dv


class _GatherAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, token_idx, key_mask, scale):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        token_idx = token_idx.contiguous().to(torch.int32)
        if key_mask is not None:
            key_mask = key_mask.contiguous()
        out, L = _fwd(q, k, v, token_idx, key_mask, scale)
        ctx.scale = scale
        ctx.has_mask = key_mask is not None
        if key_mask is not None:
            ctx.save_for_backward(q, k, v, out, L, token_idx, key_mask)
        else:
            ctx.save_for_backward(q, k, v, out, L, token_idx)
        return out

    @staticmethod
    def backward(ctx, dout):
        if ctx.has_mask:
            q, k, v, out, L, token_idx, key_mask = ctx.saved_tensors
        else:
            q, k, v, out, L, token_idx = ctx.saved_tensors
            key_mask = None
        dq, dk, dv = _bwd(q, k, v, out, L, dout.contiguous(), token_idx, key_mask, ctx.scale)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


def gather_attention_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    token_idx: torch.Tensor,
    key_mask: torch.Tensor | None = None,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Autograd-enabled fused gather attention. q/k/v [BH,T,D]; token_idx [BH,T,M]."""
    if not triton_available():
        raise RuntimeError("Triton CUDA kernels are not available")
    return _GatherAttentionFn.apply(q, k, v, token_idx, key_mask, scale)
