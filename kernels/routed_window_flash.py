"""Tiled (flash-style) causal local-window plus head-shared routed attention.

Each program owns a BLOCK_M tile of queries for one head and accumulates an
online softmax over

  1. the contiguous key tiles covering the tile's local windows, and
  2. tiles of routed keys gathered by index,

using ``tl.dot`` so both phases run on tensor cores. Routed indices are sorted
ascending on the host, and each query tile only loops over the prefix of
routed keys that can be causally visible to it.

Matches ``sparse_attn_utils.causal_sparse_attention`` semantics, including
``token_mask``, ``routed_mask`` and ``dedupe_local``. Duplicate routed indices
are each counted, as in the PyTorch path.
"""

from __future__ import annotations

import torch

from .common import _TRITON_IMPORTED, ceil_pow2

if _TRITON_IMPORTED:
    import triton
    import triton.language as tl

    @triton.jit
    def _routed_window_flash_fwd(
        Q, K, V, Out,
        Idx, RMask, KeyMask, RCount,
        stride_qb, stride_qt,
        stride_kb, stride_ks,
        stride_vb, stride_vs,
        stride_ob, stride_ot,
        stride_ib,
        stride_kmb,
        stride_cb,
        tgt_len, src_len, route_k, num_heads,
        qk_scale,
        WINDOW: tl.constexpr,
        ROUTE_OFFSET: tl.constexpr,
        KV_GROUPS: tl.constexpr,
        HAS_RMASK: tl.constexpr,
        HAS_KEYMASK: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        bh = tl.program_id(1)
        kv_bh = bh // KV_GROUPS
        batch = bh // num_heads

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        d_ok = offs_d < HEAD_DIM
        m_ok = offs_m < tgt_len

        q = tl.load(
            Q + bh * stride_qb + offs_m[:, None] * stride_qt + offs_d[None, :],
            mask=m_ok[:, None] & d_ok[None, :], other=0.0,
        )

        NEG: tl.constexpr = -1.0e30
        m_i = tl.full([BLOCK_M], NEG, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        k_base = K + kv_bh * stride_kb
        v_base = V + kv_bh * stride_vb

        # Phase 1: contiguous keys covering every local window in this tile.
        lo = tl.maximum(pid_m * BLOCK_M - WINDOW + 1, 0)
        hi = tl.minimum(pid_m * BLOCK_M + BLOCK_M, src_len)
        for start_n in range(lo, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_ok = offs_n < hi
            k = tl.load(k_base + offs_n[:, None] * stride_ks + offs_d[None, :],
                        mask=n_ok[:, None] & d_ok[None, :], other=0.0)
            v = tl.load(v_base + offs_n[:, None] * stride_vs + offs_d[None, :],
                        mask=n_ok[:, None] & d_ok[None, :], other=0.0)
            rel = offs_m[:, None] - offs_n[None, :]
            valid = (rel >= 0) & (rel < WINDOW) & n_ok[None, :]
            if HAS_KEYMASK:
                keep = tl.load(KeyMask + batch * stride_kmb + offs_n, mask=n_ok, other=0)
                valid = valid & (keep != 0)[None, :]

            s = tl.dot(q, tl.trans(k)) * qk_scale
            s = tl.where(valid, s, NEG)
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        # Phase 2: routed keys (sorted ascending); only the causally visible prefix.
        count = tl.load(RCount + bh * stride_cb + pid_m)
        for start_r in range(0, count, BLOCK_N):
            offs_r = start_r + tl.arange(0, BLOCK_N)
            r_ok = offs_r < count
            idx = tl.load(Idx + bh * stride_ib + offs_r, mask=r_ok, other=0)
            k = tl.load(k_base + idx[:, None] * stride_ks + offs_d[None, :],
                        mask=r_ok[:, None] & d_ok[None, :], other=0.0)
            v = tl.load(v_base + idx[:, None] * stride_vs + offs_d[None, :],
                        mask=r_ok[:, None] & d_ok[None, :], other=0.0)
            valid = (idx[None, :] <= offs_m[:, None] - ROUTE_OFFSET) & r_ok[None, :]
            if HAS_RMASK:
                rkeep = tl.load(RMask + bh * stride_ib + offs_r, mask=r_ok, other=0)
                valid = valid & (rkeep != 0)[None, :]
            if HAS_KEYMASK:
                keep = tl.load(KeyMask + batch * stride_kmb + idx, mask=r_ok, other=0)
                valid = valid & (keep != 0)[None, :]

            s = tl.dot(q, tl.trans(k)) * qk_scale
            s = tl.where(valid, s, NEG)
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        out = acc / l_i[:, None]
        tl.store(
            Out + bh * stride_ob + offs_m[:, None] * stride_ot + offs_d[None, :],
            out.to(Out.dtype.element_ty),
            mask=m_ok[:, None] & d_ok[None, :],
        )

    @torch.no_grad()
    def routed_window_attention_flash(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        routed_indices: torch.Tensor,
        window_size: int,
        token_mask: torch.Tensor | None = None,
        num_heads: int = 1,
        routed_mask: torch.Tensor | None = None,
        dedupe_local: bool = False,
        scale: float = 1.0,
        block_m: int = 64,
        block_n: int = 64,
    ) -> torch.Tensor:
        """Causal local-window + routed attention (inference only).

        Args:
            q: [BH, T, D]
            k, v: [BH // kv_groups, S, D] — pass un-expanded GQA K/V to skip
                repeat_interleave; kv_groups is inferred from the leading dims
            routed_indices: [BH, R] head-shared key indices
            window_size: local causal window size
            token_mask: [B, S] bool padding mask, B = BH // num_heads
            num_heads: query heads per batch element (for token_mask)
            routed_mask: [BH, R] bool — False entries are ignored
            dedupe_local: skip routed keys that are inside a query's window
            scale: multiplier applied to QK^T

        Returns: [BH, T, D]
        """
        bh, tgt_len, head_dim = q.shape
        src_len = k.size(1)
        if bh % k.size(0) != 0:
            raise ValueError(f"q heads {bh} not a multiple of kv heads {k.size(0)}")
        kv_groups = bh // k.size(0)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = torch.empty_like(q)

        idx_sorted, order = routed_indices.sort(dim=-1)
        idx_sorted = idx_sorted.to(torch.int32).contiguous()
        rmask = None
        if routed_mask is not None:
            rmask = torch.gather(routed_mask, 1, order).to(torch.int8).contiguous()

        route_offset = int(window_size) if dedupe_local else 0
        n_mblocks = triton.cdiv(tgt_len, block_m)
        last_q = (torch.arange(n_mblocks, device=q.device, dtype=torch.int32) + 1) * block_m - 1
        last_q = last_q.clamp(max=tgt_len - 1) - route_offset
        rcount = torch.searchsorted(
            idx_sorted, last_q.unsqueeze(0).expand(bh, -1).contiguous(), right=True,
        ).to(torch.int32).contiguous()

        keymask = None
        if token_mask is not None:
            keymask = token_mask.to(torch.int8).contiguous()

        dummy = idx_sorted
        _routed_window_flash_fwd[(n_mblocks, bh)](
            q, k, v, out,
            idx_sorted,
            rmask if rmask is not None else dummy,
            keymask if keymask is not None else dummy,
            rcount,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            out.stride(0), out.stride(1),
            idx_sorted.stride(0),
            keymask.stride(0) if keymask is not None else 0,
            rcount.stride(0),
            tgt_len, src_len, idx_sorted.size(1), num_heads,
            scale * 1.4426950408889634,  # log2(e): kernel uses exp2
            WINDOW=int(window_size),
            ROUTE_OFFSET=route_offset,
            KV_GROUPS=kv_groups,
            HAS_RMASK=rmask is not None,
            HAS_KEYMASK=keymask is not None,
            HEAD_DIM=head_dim,
            BLOCK_D=max(16, ceil_pow2(head_dim)),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=2,
        )
        return out

else:

    def routed_window_attention_flash(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("Triton is not available")
