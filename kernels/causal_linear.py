"""Causal low-rank linear-attention kernel for Exp 18."""

from __future__ import annotations

import torch

from .common import _TRITON_IMPORTED, ceil_pow2, triton_available

if _TRITON_IMPORTED:
    import triton
    import triton.language as tl

    @triton.jit
    def _causal_linear_kernel(
        Q,
        K,
        V,
        Out,
        Mask,
        stride_qb,
        stride_qt,
        stride_qf,
        stride_kb,
        stride_kt,
        stride_kf,
        stride_vb,
        stride_vt,
        stride_vd,
        stride_ob,
        stride_ot,
        stride_od,
        stride_mb,
        stride_mt,
        seq_len,
        feature_dim: tl.constexpr,
        value_dim: tl.constexpr,
        BLOCK_F: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_MASK: tl.constexpr,
        EPS,
    ):
        bh = tl.program_id(0)
        f = tl.arange(0, BLOCK_F)
        d = tl.arange(0, BLOCK_D)
        fm = f < feature_dim
        dm = d < value_dim

        state = tl.zeros((BLOCK_F, BLOCK_D), dtype=tl.float32)
        z = tl.zeros((BLOCK_F,), dtype=tl.float32)

        for t in range(seq_len):
            k = tl.load(
                K + bh * stride_kb + t * stride_kt + f * stride_kf,
                mask=fm,
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                V + bh * stride_vb + t * stride_vt + d * stride_vd,
                mask=dm,
                other=0.0,
            ).to(tl.float32)
            if HAS_MASK:
                keep = tl.load(Mask + bh * stride_mb + t * stride_mt).to(tl.int1)
                k = tl.where(keep, k, 0.0)

            state += k[:, None] * v[None, :]
            z += k

            q = tl.load(
                Q + bh * stride_qb + t * stride_qt + f * stride_qf,
                mask=fm,
                other=0.0,
            ).to(tl.float32)
            numerator = tl.sum(q[:, None] * state, axis=0)
            denominator = tl.sum(q * z, axis=0)
            out = numerator / (denominator + EPS)
            tl.store(
                Out + bh * stride_ob + t * stride_ot + d * stride_od,
                out,
                mask=dm,
            )


def causal_linear_attention(
    q_features: torch.Tensor,
    k_features: torch.Tensor,
    values: torch.Tensor,
    key_mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute causal linear attention from precomputed positive features.

    Inputs are ``q_features``/``k_features`` with shape ``[BH, T, F]`` and
    ``values`` with shape ``[BH, T, D]``. This function is inference-only;
    Exp 18 uses the differentiable reference path when training kernels are
    unavailable.
    """
    if not triton_available():
        raise RuntimeError("Triton CUDA kernels are not available")
    q_features = q_features.contiguous()
    k_features = k_features.contiguous()
    values = values.contiguous()
    bh, tgt_len, feature_dim = q_features.shape
    if k_features.shape[:2] != (bh, tgt_len):
        raise ValueError("causal linear attention requires matching Q/K lengths")
    value_dim = values.size(-1)
    out = torch.empty_like(values)
    block_f = ceil_pow2(feature_dim)
    block_d = ceil_pow2(value_dim)
    has_mask = key_mask is not None
    mask_i8 = key_mask.contiguous().to(torch.int8) if has_mask else q_features

    _causal_linear_kernel[(bh,)](
        q_features,
        k_features,
        values,
        out,
        mask_i8,
        q_features.stride(0),
        q_features.stride(1),
        q_features.stride(2),
        k_features.stride(0),
        k_features.stride(1),
        k_features.stride(2),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        mask_i8.stride(0) if has_mask else 0,
        mask_i8.stride(1) if has_mask else 0,
        tgt_len,
        feature_dim=feature_dim,
        value_dim=value_dim,
        BLOCK_F=block_f,
        BLOCK_D=block_d,
        HAS_MASK=has_mask,
        EPS=eps,
        num_warps=8 if value_dim >= 64 else 4,
    )
    return out
