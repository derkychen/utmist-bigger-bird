"""Autograd-capable fused local plus head-shared routed attention."""

from __future__ import annotations

import torch

from .common import MASK_SCORE_FP32, _TRITON_IMPORTED, ceil_pow2, triton_available

if _TRITON_IMPORTED:
    import triton
    import triton.language as tl

    @triton.jit
    def _routed_window_fwd(
        Q,
        K,
        V,
        Out,
        LSE,
        Route,
        Mask,
        sqb,
        sqt,
        sqd,
        skb,
        sks,
        skd,
        svb,
        svs,
        svd,
        sob,
        sot,
        sod,
        slb,
        slt,
        srb,
        srk,
        smb,
        smt,
        seq_len,
        window_size: tl.constexpr,
        route_k: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
        RADIUS: tl.constexpr,
        HAS_MASK: tl.constexpr,
        SCALE,
    ):
        bh = tl.program_id(0)
        t = tl.program_id(1)
        d = tl.arange(0, BLOCK_D)
        dm = d < head_dim
        q = tl.load(Q + bh * sqb + t * sqt + d * sqd, mask=dm, other=0.0).to(tl.float32)

        m_i = MASK_SCORE_FP32
        l_i = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for j in range(window_size + route_k):
            if j < window_size:
                if CAUSAL:
                    raw = tl.maximum(0, t - window_size + 1) + j
                    idx = tl.minimum(raw, seq_len - 1)
                    valid = idx <= t
                else:
                    raw = tl.maximum(0, t - RADIUS) + j
                    idx = tl.minimum(raw, seq_len - 1)
                    valid = (raw <= t + RADIUS) and (raw <= seq_len - 1)
            else:
                idx = tl.load(Route + bh * srb + (j - window_size) * srk).to(tl.int32)
                idx = tl.minimum(tl.maximum(idx, 0), seq_len - 1)
                if CAUSAL:
                    valid = idx <= t
                else:
                    valid = True

            k = tl.load(K + bh * skb + idx * sks + d * skd, mask=dm, other=0.0).to(tl.float32)
            v = tl.load(V + bh * svb + idx * svs + d * svd, mask=dm, other=0.0).to(tl.float32)
            score = tl.sum(q * k, axis=0) * SCALE
            score = tl.where(valid, score, MASK_SCORE_FP32)
            if HAS_MASK:
                keep = tl.load(Mask + bh * smb + idx * smt).to(tl.int1)
                score = tl.where(keep, score, MASK_SCORE_FP32)

            m_new = tl.maximum(m_i, score)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(score - m_new)
            l_i = l_i * alpha + p
            acc = acc * alpha + p * v
            m_i = m_new

        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        tl.store(Out + bh * sob + t * sot + d * sod, acc / safe_l, mask=dm)
        tl.store(LSE + bh * slb + t * slt, m_i + tl.log(safe_l))

    @triton.jit
    def _routed_window_bwd(
        Q,
        K,
        V,
        Out,
        LSE,
        dOut,
        dQ,
        dK,
        dV,
        Route,
        Mask,
        sqb,
        sqt,
        sqd,
        skb,
        sks,
        skd,
        svb,
        svs,
        svd,
        sob,
        sot,
        sod,
        slb,
        slt,
        sdqb,
        sdqt,
        sdqd,
        srb,
        srk,
        smb,
        smt,
        seq_len,
        window_size: tl.constexpr,
        route_k: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
        RADIUS: tl.constexpr,
        HAS_MASK: tl.constexpr,
        SCALE,
    ):
        bh = tl.program_id(0)
        t = tl.program_id(1)
        d = tl.arange(0, BLOCK_D)
        dm = d < head_dim
        q = tl.load(Q + bh * sqb + t * sqt + d * sqd, mask=dm, other=0.0).to(tl.float32)
        dout = tl.load(dOut + bh * sob + t * sot + d * sod, mask=dm, other=0.0).to(tl.float32)
        out = tl.load(Out + bh * sob + t * sot + d * sod, mask=dm, other=0.0).to(tl.float32)
        lse = tl.load(LSE + bh * slb + t * slt)
        delta = tl.sum(dout * out, axis=0)
        dq = tl.zeros([BLOCK_D], dtype=tl.float32)

        for j in range(window_size + route_k):
            if j < window_size:
                if CAUSAL:
                    raw = tl.maximum(0, t - window_size + 1) + j
                    idx = tl.minimum(raw, seq_len - 1)
                    valid = idx <= t
                else:
                    raw = tl.maximum(0, t - RADIUS) + j
                    idx = tl.minimum(raw, seq_len - 1)
                    valid = (raw <= t + RADIUS) and (raw <= seq_len - 1)
            else:
                idx = tl.load(Route + bh * srb + (j - window_size) * srk).to(tl.int32)
                idx = tl.minimum(tl.maximum(idx, 0), seq_len - 1)
                if CAUSAL:
                    valid = idx <= t
                else:
                    valid = True

            k = tl.load(K + bh * skb + idx * sks + d * skd, mask=dm, other=0.0).to(tl.float32)
            v = tl.load(V + bh * svb + idx * svs + d * svd, mask=dm, other=0.0).to(tl.float32)
            score = tl.sum(q * k, axis=0) * SCALE
            score = tl.where(valid, score, MASK_SCORE_FP32)
            if HAS_MASK:
                keep = tl.load(Mask + bh * smb + idx * smt).to(tl.int1)
                score = tl.where(keep, score, MASK_SCORE_FP32)

            p = tl.exp(score - lse)
            dp = tl.sum(dout * v, axis=0)
            ds = p * (dp - delta)
            dq += ds * SCALE * k
            tl.atomic_add(dV + bh * svb + idx * svs + d * svd, p * dout, mask=dm)
            tl.atomic_add(dK + bh * skb + idx * sks + d * skd, ds * SCALE * q, mask=dm)

        tl.store(dQ + bh * sdqb + t * sdqt + d * sdqd, dq, mask=dm)


class _RoutedWindowFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, route, key_mask, window_size, causal, scale):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        route = route.contiguous().to(torch.int32)
        if key_mask is not None:
            key_mask = key_mask.contiguous()
        bh, tgt_len, head_dim = q.shape
        out = torch.empty_like(q)
        lse = torch.empty(bh, tgt_len, device=q.device, dtype=torch.float32)
        block_d = ceil_pow2(head_dim)
        width = min(int(window_size), k.size(1)) if causal else min(2 * (int(window_size) // 2) + 1, k.size(1))
        radius = width // 2
        has_mask = key_mask is not None
        mask_i8 = key_mask.to(torch.int8) if has_mask else q
        _routed_window_fwd[(bh, tgt_len)](
            q, k, v, out, lse, route, mask_i8,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            lse.stride(0), lse.stride(1),
            route.stride(0), route.stride(1),
            mask_i8.stride(0) if has_mask else 0,
            mask_i8.stride(1) if has_mask else 0,
            k.size(1),
            window_size=width,
            route_k=route.size(-1),
            head_dim=head_dim,
            BLOCK_D=block_d,
            CAUSAL=bool(causal),
            RADIUS=radius,
            HAS_MASK=has_mask,
            SCALE=scale,
            num_warps=8 if head_dim >= 64 else 4,
        )
        ctx.window_size = width
        ctx.causal = bool(causal)
        ctx.scale = scale
        ctx.has_mask = has_mask
        if has_mask:
            ctx.save_for_backward(q, k, v, out, lse, route, key_mask)
        else:
            ctx.save_for_backward(q, k, v, out, lse, route)
        return out

    @staticmethod
    def backward(ctx, dout):
        if ctx.has_mask:
            q, k, v, out, lse, route, key_mask = ctx.saved_tensors
        else:
            q, k, v, out, lse, route = ctx.saved_tensors
            key_mask = None
        dout = dout.contiguous()
        bh, tgt_len, head_dim = q.shape
        dQ = torch.zeros_like(q, dtype=torch.float32)
        dK = torch.zeros_like(k, dtype=torch.float32)
        dV = torch.zeros_like(v, dtype=torch.float32)
        block_d = ceil_pow2(head_dim)
        mask_i8 = key_mask.to(torch.int8) if key_mask is not None else q
        _routed_window_bwd[(bh, tgt_len)](
            q, k, v, out, lse, dout, dQ, dK, dV, route, mask_i8,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            lse.stride(0), lse.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            route.stride(0), route.stride(1),
            mask_i8.stride(0) if key_mask is not None else 0,
            mask_i8.stride(1) if key_mask is not None else 0,
            k.size(1),
            window_size=ctx.window_size,
            route_k=route.size(-1),
            head_dim=head_dim,
            BLOCK_D=block_d,
            CAUSAL=ctx.causal,
            RADIUS=ctx.window_size // 2,
            HAS_MASK=key_mask is not None,
            SCALE=ctx.scale,
            num_warps=8 if head_dim >= 64 else 4,
        )
        return dQ.to(q.dtype), dK.to(k.dtype), dV.to(v.dtype), None, None, None, None, None


def routed_window_attention_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    routed_indices: torch.Tensor,
    window_size: int,
    key_mask: torch.Tensor | None = None,
    *,
    causal: bool = True,
    scale: float = 1.0,
) -> torch.Tensor:
    """Autograd-enabled fused local plus routed attention."""
    if not triton_available():
        raise RuntimeError("Triton CUDA kernels are not available")
    return _RoutedWindowFn.apply(
        q, k, v, routed_indices, key_mask, int(window_size), bool(causal), float(scale)
    )

