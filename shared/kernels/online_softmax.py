"""Unified online-softmax attention kernel (gather / window / compressed modes)."""

from __future__ import annotations

import torch

from .common import MASK_SCORE_FP32, MODE_GATHER, MODE_WINDOW, ceil_pow2, _TRITON_IMPORTED

if _TRITON_IMPORTED:
    import triton
    import triton.language as tl

    @triton.jit
    def _attn_fwd_kernel(
        Q,
        K,
        V,
        Out,
        Idx,
        Mask,
        stride_qb,
        stride_qt,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_ot,
        stride_od,
        stride_ib,
        stride_it,
        stride_im,
        stride_mb,
        stride_mt,
        stride_mm,
        seq_len,
        n_keys: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MODE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        window_size: tl.constexpr,
        block_size: tl.constexpr,
        stride_blocks: tl.constexpr,
        radius: tl.constexpr,
        SCALE,
        MASK_VAL: tl.constexpr,
    ):
        bh = tl.program_id(0)
        t = tl.program_id(1)

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < head_dim

        q_ptr = Q + bh * stride_qb + t * stride_qt + d_offs * stride_qd
        q = tl.load(q_ptr, mask=d_mask, other=0.0).to(tl.float32)

        m_i = MASK_VAL
        l_i = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for j in range(n_keys):
            # MODE: 0=gather, 1=window, 2=compressed, 3=band
            if MODE == 0:
                idx = tl.load(Idx + bh * stride_ib + t * stride_it + j * stride_im).to(tl.int32)
                idx = tl.minimum(tl.maximum(idx, 0), seq_len - 1)
                valid = True
            elif MODE == 1:
                start = tl.maximum(0, t - window_size + 1)
                idx = start + j
                idx = tl.minimum(idx, seq_len - 1)
                valid = True
            elif MODE == 2:
                idx = j
                block_end = j * stride_blocks + block_size
                valid = block_end <= t
            else:
                # Symmetric band: query t attends to keys in [t-radius, t+radius].
                raw = tl.maximum(0, t - radius) + j
                valid = (raw <= t + radius) and (raw <= seq_len - 1)
                idx = tl.minimum(raw, seq_len - 1)

            k_ptr = K + bh * stride_kb + idx * stride_ks + d_offs * stride_kd
            v_ptr = V + bh * stride_vb + idx * stride_vs + d_offs * stride_vd
            k = tl.load(k_ptr, mask=d_mask, other=0.0).to(tl.float32)
            v = tl.load(v_ptr, mask=d_mask, other=0.0).to(tl.float32)

            s = tl.sum(q * k, axis=0) * SCALE
            s = tl.where(valid, s, MASK_VAL)

            if HAS_MASK:
                if MODE == 0:
                    keep = tl.load(Mask + bh * stride_mb + t * stride_mt + j * stride_mm).to(tl.int1)
                elif MODE == 1:
                    keep = tl.load(Mask + bh * stride_mb + idx * stride_mt).to(tl.int1)
                elif MODE == 2:
                    keep = tl.load(Mask + bh * stride_mb + j * stride_mt).to(tl.int1)
                else:
                    keep = tl.load(Mask + bh * stride_mb + idx * stride_mt).to(tl.int1)
                s = tl.where(keep, s, MASK_VAL)

            m_new = tl.maximum(m_i, s)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + p
            acc = acc * alpha + p * v
            m_i = m_new

        out = acc / l_i
        o_ptr = Out + bh * stride_ob + t * stride_ot + d_offs * stride_od
        tl.store(o_ptr, out, mask=d_mask)

    def _launch(
        mode: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n_keys: int,
        idx: torch.Tensor | None,
        mask: torch.Tensor | None,
        *,
        scale: float,
        window_size: int = 1,
        block_size: int = 1,
        stride_blocks: int = 1,
        radius: int = 0,
    ) -> torch.Tensor:
        bh, tgt_len, head_dim = q.shape
        out = torch.empty_like(q)
        block_d = ceil_pow2(head_dim)
        has_mask = mask is not None
        idx_t = idx if idx is not None else q
        mask_i8 = mask.to(torch.int8) if has_mask else q

        _attn_fwd_kernel[(bh, tgt_len)](
            q,
            k,
            v,
            out,
            idx_t,
            mask_i8,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            idx_t.stride(0) if mode == MODE_GATHER else 0,
            idx_t.stride(1) if mode == MODE_GATHER else 0,
            idx_t.stride(2) if mode == MODE_GATHER else 0,
            mask_i8.stride(0) if has_mask else 0,
            mask_i8.stride(1) if has_mask else 0,
            mask_i8.stride(2) if has_mask and mode == MODE_GATHER else 0,
            k.size(1),
            n_keys=n_keys,
            head_dim=head_dim,
            BLOCK_D=block_d,
            MODE=mode,
            HAS_MASK=has_mask,
            window_size=window_size,
            block_size=block_size,
            stride_blocks=stride_blocks,
            radius=radius,
            SCALE=scale,
            MASK_VAL=MASK_SCORE_FP32,
            num_warps=4 if head_dim <= 64 else 8,
        )
        return out

else:

    def _launch(*_args, **_kwargs):  # type: ignore[misc]
        raise RuntimeError("Triton is not installed")
