"""ELU-feature-map linear attention (inter-block long-range branch).

Implements the kernelized linear attention used by exp_2's global branch:

    phi(x) = elu(x) + 1
    KV     = sum_s phi(K_s) outer V_s        # [D, D] per (batch*head)
    Z      = sum_s phi(K_s)                   # [D]
    out_t  = (phi(Q_t) @ KV) / (phi(Q_t) @ Z + eps)

The accumulation is O(S * D^2) instead of the O(S^2) of softmax attention.
"""

from __future__ import annotations

import torch

from .common import _TRITON_IMPORTED, ceil_pow2, triton_available

LINEAR_EPS = 1e-6

if _TRITON_IMPORTED:
    import triton
    import triton.language as tl

    @triton.jit
    def _elu_feature(x):
        # phi(x) = elu(x) + 1 = x + 1 if x > 0 else exp(x)
        return tl.where(x > 0, x + 1.0, tl.exp(x))

    @triton.jit
    def _linear_kv_kernel(
        K,
        V,
        Mask,
        KV,
        Z,
        stride_kb,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vd,
        stride_mb,
        stride_ms,
        stride_kvb,
        stride_kvi,
        stride_kvj,
        stride_zb,
        stride_zd,
        src_len,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        bh = tl.program_id(0)
        d1 = tl.arange(0, BLOCK_D)
        d2 = tl.arange(0, BLOCK_D)
        dm1 = d1 < head_dim
        dm2 = d2 < head_dim

        kv = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
        z = tl.zeros([BLOCK_D], dtype=tl.float32)

        for s in range(src_len):
            k = tl.load(
                K + bh * stride_kb + s * stride_ks + d1 * stride_kd, mask=dm1, other=0.0
            ).to(tl.float32)
            kf = _elu_feature(k)
            if HAS_MASK:
                keep = tl.load(Mask + bh * stride_mb + s * stride_ms).to(tl.int1)
                kf = tl.where(keep, kf, 0.0)
            v = tl.load(
                V + bh * stride_vb + s * stride_vs + d2 * stride_vd, mask=dm2, other=0.0
            ).to(tl.float32)
            kv += kf[:, None] * v[None, :]
            z += kf

        kv_ptr = KV + bh * stride_kvb + d1[:, None] * stride_kvi + d2[None, :] * stride_kvj
        tl.store(kv_ptr, kv, mask=dm1[:, None] & dm2[None, :])
        tl.store(Z + bh * stride_zb + d1 * stride_zd, z, mask=dm1)

    @triton.jit
    def _linear_out_kernel(
        Q,
        KV,
        Z,
        Out,
        stride_qb,
        stride_qt,
        stride_qd,
        stride_kvb,
        stride_kvi,
        stride_kvj,
        stride_zb,
        stride_zd,
        stride_ob,
        stride_ot,
        stride_od,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
        EPS,
    ):
        bh = tl.program_id(0)
        t = tl.program_id(1)
        d1 = tl.arange(0, BLOCK_D)
        d2 = tl.arange(0, BLOCK_D)
        dm1 = d1 < head_dim
        dm2 = d2 < head_dim

        q = tl.load(
            Q + bh * stride_qb + t * stride_qt + d1 * stride_qd, mask=dm1, other=0.0
        ).to(tl.float32)
        qf = _elu_feature(q)

        kv = tl.load(
            KV + bh * stride_kvb + d1[:, None] * stride_kvi + d2[None, :] * stride_kvj,
            mask=dm1[:, None] & dm2[None, :],
            other=0.0,
        )
        z = tl.load(Z + bh * stride_zb + d1 * stride_zd, mask=dm1, other=0.0)

        num = tl.sum(qf[:, None] * kv, axis=0)
        den = tl.sum(qf * z, axis=0)
        out = num / (den + EPS)
        tl.store(Out + bh * stride_ob + t * stride_ot + d2 * stride_od, out, mask=dm2)


@torch.inference_mode()
def elu_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor | None = None,
    *,
    eps: float = LINEAR_EPS,
) -> torch.Tensor:
    """Fused ELU linear attention. q/k/v [BH,T,D]; key_mask optional [BH,S]."""
    if not triton_available():
        raise RuntimeError("Triton CUDA kernels are not available")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    bh, tgt_len, head_dim = q.shape
    src_len = k.size(1)
    block_d = ceil_pow2(head_dim)
    has_mask = key_mask is not None
    mask_i8 = key_mask.contiguous().to(torch.int8) if has_mask else q

    kv = torch.empty(bh, head_dim, head_dim, device=q.device, dtype=torch.float32)
    z = torch.empty(bh, head_dim, device=q.device, dtype=torch.float32)

    _linear_kv_kernel[(bh,)](
        k,
        v,
        mask_i8,
        kv,
        z,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        mask_i8.stride(0) if has_mask else 0,
        mask_i8.stride(1) if has_mask else 0,
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        z.stride(0),
        z.stride(1),
        src_len,
        head_dim=head_dim,
        BLOCK_D=block_d,
        HAS_MASK=has_mask,
        num_warps=4 if head_dim <= 64 else 8,
    )

    out = torch.empty_like(q)
    _linear_out_kernel[(bh, tgt_len)](
        q,
        kv,
        z,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        z.stride(0),
        z.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        head_dim=head_dim,
        BLOCK_D=block_d,
        EPS=eps,
        num_warps=4 if head_dim <= 64 else 8,
    )
    return out
