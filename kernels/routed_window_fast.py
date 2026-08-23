"""Block-parallel fused causal local-window plus routed attention.

This kernel processes keys in blocks of BLOCK_K instead of one at a time,
reducing kernel launch overhead and enabling vectorized memory access.
"""

from __future__ import annotations

import torch

from .common import ceil_pow2, _TRITON_IMPORTED

if _TRITON_IMPORTED:
    import triton
    import triton.language as tl

    @triton.jit
    def _routed_window_fwd_kernel(
        Q,
        K,
        V,
        Out,
        Idx,  # [BH, K] routed indices (head-shared)
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
        stride_im,
        seq_len,
        tgt_len,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        ROUTE_K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SCALE,
    ):
        pid = tl.program_id(0)
        bh = pid // tgt_len
        t = pid % tgt_len

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < head_dim

        q_ptr = Q + bh * stride_qb + t * stride_qt + d_offs * stride_qd
        q = tl.load(q_ptr, mask=d_mask, other=0.0).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        # Phase 1: local window keys [max(0, t-W+1), t]
        local_start = tl.maximum(0, t - WINDOW_SIZE + 1)
        local_count = t - local_start + 1  # number of valid local keys

        for j_start in range(0, WINDOW_SIZE, BLOCK_K):
            j_offs = j_start + tl.arange(0, BLOCK_K)
            # Local key index = local_start + j
            idx = local_start + j_offs
            idx = tl.minimum(idx, seq_len - 1)
            # Valid if j < local_count (i.e., idx <= t)
            valid = (j_offs < local_count) & (idx <= t)

            k_ptrs = K + bh * stride_kb + idx[:, None] * stride_ks + d_offs[None, :] * stride_kd
            v_ptrs = V + bh * stride_vb + idx[:, None] * stride_vs + d_offs[None, :] * stride_vd
            k = tl.load(k_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

            # Scores: [BLOCK_K]
            s = tl.sum(q[None, :] * k, axis=1) * SCALE
            s = tl.where(valid, s, -float("inf"))

            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(s))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + tl.sum(p)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        # Phase 2: routed keys (head-shared indices)
        for j_start in range(0, ROUTE_K, BLOCK_K):
            j_offs = j_start + tl.arange(0, BLOCK_K)
            # Load routed indices
            idx = tl.load(
                Idx + bh * stride_ib + j_offs * stride_im,
                mask=j_offs < ROUTE_K,
                other=0,
            ).to(tl.int32)
            idx = tl.minimum(tl.maximum(idx, 0), seq_len - 1)
            # Causal: routed key must be <= t
            valid = (j_offs < ROUTE_K) & (idx <= t)

            k_ptrs = K + bh * stride_kb + idx[:, None] * stride_ks + d_offs[None, :] * stride_kd
            v_ptrs = V + bh * stride_vb + idx[:, None] * stride_vs + d_offs[None, :] * stride_vd
            k = tl.load(k_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

            s = tl.sum(q[None, :] * k, axis=1) * SCALE
            s = tl.where(valid, s, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(s))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + tl.sum(p)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        out = acc / l_i
        o_ptr = Out + bh * stride_ob + t * stride_ot + d_offs * stride_od
        tl.store(o_ptr, out, mask=d_mask)

    @torch.inference_mode()
    def routed_window_attention_fast(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        routed_indices: torch.Tensor,
        window_size: int,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Fast block-parallel causal local + routed attention.

        Args:
            q, k, v: [BH, T, D]
            routed_indices: [BH, K] head-shared key indices
            window_size: local causal window size
            scale: softmax scale factor

        Returns: [BH, T, D]
        """
        bh, tgt_len, head_dim = q.shape
        out = torch.empty_like(q)
        route_k = routed_indices.size(-1)
        block_d = ceil_pow2(head_dim)
        block_k = 64  # keys per block

        # Pad route_k to block_k multiple for the loop
        routed_indices = routed_indices.contiguous().to(torch.int32)

        _routed_window_fwd_kernel[(bh * tgt_len,)](
            q,
            k,
            v,
            out,
            routed_indices,
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
            routed_indices.stride(0),
            routed_indices.stride(1),
            k.size(1),
            tgt_len,
            head_dim=head_dim,
            BLOCK_D=block_d,
            WINDOW_SIZE=window_size,
            ROUTE_K=route_k,
            BLOCK_K=block_k,
            SCALE=scale,
            num_warps=4 if head_dim <= 64 else 8,
        )
        return out

else:
    def routed_window_attention_fast(*args, **kwargs):
        raise RuntimeError("Triton is not available")
