"""Exp 16 — Parameter-Free NSA on R1-Distill-Llama-8B.

A training-free variant of NSA (exp 11) that replaces all learned components
with parameter-free alternatives:

  (1) Compressed branch: mean-pool block compression instead of learned φ
  (2) Selected branch: block-mean QK routing (same as exp 11 but no learned gate)
  (3) Sliding window: uses base K/V projections (no separate learned projections)

Branch combination: fixed weights instead of learned gate_mlp.
  - Window: weight 1.0 (always important for local context)
  - Compressed: weight 0.5 (coarse long-range summary)
  - Selected: weight 1.0 (fine-grained long-range retrieval)

This is the key difference from exp 11: in zero-shot RULER eval, exp 11's
randomly initialized compress_k/compress_v/gate_mlp produce random routing,
so the needle is almost never selected. Exp 16 uses the model's existing
Q/K/V weights for all routing decisions, so it works without any training.

Complexity: O(N * (W + topk_blocks * block_size)) = O(N * M) where M is fixed.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from patches.llama.llama_patched_model import (
    LlamaSparseAttention,
    LlamaPatchedModel,
    apply_lora,
)
from sparse_attn_utils import (
    causal_sparse_attention,
    last_query_block_topk_indices,
)


class FreeNSAAttention(LlamaSparseAttention):
    """Parameter-free NSA: mean-pool compression + QK block routing + fixed gates."""

    def __init__(
        self,
        base_attn,
        block_size: int = 64,
        topk_blocks: int = 16,
        window_size: int = 256,
        cmp_weight: float = 0.5,
        slc_weight: float = 1.0,
        win_weight: float = 1.0,
        use_triton: bool = False,
    ):
        super().__init__(base_attn)
        self.block_size = block_size
        self.topk_blocks = topk_blocks
        self.window_size = window_size
        self.cmp_weight = cmp_weight
        self.slc_weight = slc_weight
        self.win_weight = win_weight
        self.use_triton = use_triton

    def forward(self, hidden_states, **kwargs):
        return super().forward(hidden_states, **kwargs)

    def _block_means(self, tensor, blk):
        """Mean-pool [BH, L, D] into block representatives [BH, n_blocks, D]."""
        bh, length, dim = tensor.shape
        pad = (blk - length % blk) % blk
        if pad:
            tensor = F.pad(tensor, (0, 0, 0, pad), value=0.0)
        n_blocks = tensor.size(1) // blk
        return tensor.view(bh, n_blocks, blk, dim).mean(dim=2)

    def _causal_window_branch(self, Q, K, V, bsz, tgt_len, token_mask, query_chunk=256):
        """Causal sliding-window attention using base K/V (no separate projections)."""
        bh = Q.size(0)
        src_len = K.size(1)
        w = min(self.window_size, src_len)
        d = self.head_dim

        if token_mask is not None:
            am = token_mask.unsqueeze(1).expand(bsz, self.num_heads, src_len).reshape(bh, src_len)
        else:
            am = None

        offsets = torch.arange(w, device=Q.device)
        out_chunks = []
        for q_start in range(0, tgt_len, query_chunk):
            q_end = min(q_start + query_chunk, tgt_len)
            chunk_len = q_end - q_start
            q_pos_chunk = torch.arange(q_start, q_end, device=Q.device)
            starts = torch.clamp(q_pos_chunk - w + 1, min=0)
            local_idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(max=src_len - 1)
            local_idx_exp = local_idx.unsqueeze(0).expand(bh, -1, -1)

            Q_chunk = Q[:, q_start:q_end, :]
            bh_arange = torch.arange(bh, device=Q.device).view(bh, 1, 1)
            k_sel = K[bh_arange, local_idx_exp, :]
            v_sel = V[bh_arange, local_idx_exp, :]

            scores = torch.matmul(Q_chunk.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2)

            causal_allowed = local_idx.unsqueeze(0) <= q_pos_chunk.view(1, -1, 1)
            scores = scores.masked_fill(~causal_allowed, torch.finfo(scores.dtype).min)

            if am is not None:
                allowed = torch.gather(am.unsqueeze(1).expand(-1, chunk_len, -1), 2, local_idx_exp)
                scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

            attn = F.softmax(scores, dim=-1)
            out_c = torch.bmm(
                attn.reshape(bh * chunk_len, 1, w),
                v_sel.reshape(bh * chunk_len, w, d),
            ).reshape(bh, chunk_len, d)
            out_chunks.append(out_c)

        return torch.cat(out_chunks, dim=1)

    def _causal_compressed_branch(self, Q, K, V, bsz, tgt_len, token_mask, query_chunk=256):
        """Causal compressed attention: attend to mean-pooled block summaries.

        Parameter-free: block summaries are simple mean-pools of K/V, not learned
        projections. Routing uses Q . block_mean(K) with the last query position.
        """
        bh = Q.size(0)
        d = self.head_dim
        blk = self.block_size

        k_cmp = self._block_means(K, blk)  # [BH, n_cmp, d]
        v_cmp = self._block_means(V, blk)  # [BH, n_cmp, d]
        n_cmp = k_cmp.size(1)

        block_starts = torch.arange(n_cmp, device=Q.device) * blk

        if token_mask is not None:
            block_ok_starts = block_starts.clamp(max=token_mask.size(-1) - 1)
            block_ok = token_mask[:, block_ok_starts]
            block_ok_bh = block_ok.unsqueeze(1).expand(bsz, self.num_heads, n_cmp).reshape(bh, n_cmp)
        else:
            block_ok_bh = None

        out_chunks = []
        for q_start in range(0, tgt_len, query_chunk):
            q_end = min(q_start + query_chunk, tgt_len)
            chunk_len = q_end - q_start
            Q_chunk = Q[:, q_start:q_end, :]

            scores = torch.bmm(Q_chunk, k_cmp.transpose(1, 2))

            q_pos_chunk = torch.arange(q_start, q_end, device=Q.device).unsqueeze(-1)
            causal_allowed = block_starts.unsqueeze(0) <= q_pos_chunk
            scores = scores.masked_fill(~causal_allowed.unsqueeze(0), torch.finfo(scores.dtype).min)

            if block_ok_bh is not None:
                scores = scores.masked_fill(~block_ok_bh.unsqueeze(1), torch.finfo(scores.dtype).min)

            attn = F.softmax(scores, dim=-1)
            out_c = torch.bmm(attn, v_cmp)
            out_chunks.append(out_c)

        return torch.cat(out_chunks, dim=1)

    def _causal_selected_branch(self, Q, K, V, bsz, tgt_len, token_mask):
        """Causal selected attention: top-k blocks via last-query block routing.

        Uses last_query_block_topk_indices (parameter-free QK product routing)
        then causal_sparse_attention with local window.
        """
        bh, src_len, dim = K.shape
        blk = self.block_size
        d_low = self.head_dim

        Q_low = Q[:, :, :d_low]
        K_low = K[:, :, :d_low]
        routed_idx = last_query_block_topk_indices(
            Q_low, K_low, blk, self.topk_blocks,
            token_mask, bsz, self.num_heads,
        )

        return causal_sparse_attention(
            Q, K, V, routed_idx, local_window=self.window_size,
            token_mask=token_mask, bsz=bsz, num_heads=self.num_heads,
        )

    def _window_branch(self, Q, K, V, bsz, tgt_len, token_mask):
        """Bidirectional sliding-window attention using base K/V."""
        bh = Q.size(0)
        src_len = K.size(1)
        w = min(self.window_size, src_len)
        d = self.head_dim

        t = torch.arange(tgt_len, device=Q.device)
        starts = torch.clamp(t - w // 2, min=0, max=max(0, src_len - w))
        offsets = torch.arange(w, device=Q.device)
        local_idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(max=src_len - 1)
        local_idx_exp = local_idx.unsqueeze(0).expand(bh, -1, -1)

        bh_arange = torch.arange(bh, device=Q.device).view(bh, 1, 1)
        k_sel = K[bh_arange, local_idx_exp, :]
        v_sel = V[bh_arange, local_idx_exp, :]

        scores = torch.matmul(Q.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2)

        if token_mask is not None:
            am = token_mask.unsqueeze(1).expand(bsz, self.num_heads, src_len).reshape(bh, src_len)
            allowed = torch.gather(am.unsqueeze(1).expand(bh, tgt_len, -1), 2, local_idx_exp)
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        return torch.bmm(
            attn.reshape(bh * tgt_len, 1, w),
            v_sel.reshape(bh * tgt_len, w, d),
        ).reshape(bh, tgt_len, d)

    def _compressed_branch(self, Q, K, V, bsz, tgt_len, token_mask):
        """Bidirectional compressed attention: attend to mean-pooled block summaries."""
        bh = Q.size(0)
        blk = self.block_size

        k_cmp = self._block_means(K, blk)
        v_cmp = self._block_means(V, blk)
        n_cmp = k_cmp.size(1)

        scores = torch.bmm(Q, k_cmp.transpose(1, 2))

        if token_mask is not None:
            block_starts = torch.arange(n_cmp, device=Q.device) * blk
            block_starts = block_starts.clamp(max=token_mask.size(-1) - 1)
            block_ok = token_mask[:, block_starts]
            block_ok = block_ok.unsqueeze(1).expand(bsz, self.num_heads, n_cmp).reshape(bh, 1, n_cmp).expand(-1, tgt_len, -1)
            scores = scores.masked_fill(~block_ok, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn, v_cmp)

    def _selected_branch(self, Q, K, V, bsz, tgt_len, token_mask):
        """Bidirectional selected attention: top-k blocks via per-query block routing."""
        bh, src_len, dim = K.shape
        blk = self.block_size
        n_blocks = math.ceil(src_len / blk)
        k_blocks = self._block_means(K, blk)

        block_scores = torch.bmm(Q, k_blocks.transpose(1, 2))

        if token_mask is not None:
            block_starts = torch.arange(n_blocks, device=Q.device) * blk
            block_starts = block_starts.clamp(max=token_mask.size(-1) - 1)
            block_ok = token_mask[:, block_starts]
            block_ok = block_ok.unsqueeze(1).expand(bsz, self.num_heads, n_blocks).reshape(bh, 1, n_blocks).expand(-1, tgt_len, -1)
            block_scores = block_scores.masked_fill(~block_ok, torch.finfo(block_scores.dtype).min)

        m = min(self.topk_blocks, n_blocks)
        _, top_blocks = torch.topk(block_scores, k=m, dim=-1)

        block_offset = torch.arange(blk, device=Q.device).view(1, 1, 1, blk)
        base_idx = (top_blocks.unsqueeze(2) * blk).unsqueeze(-1)
        token_idx = (base_idx + block_offset).reshape(bh, tgt_len, m * blk).clamp(max=src_len - 1)

        m_tokens = m * blk
        idx_gather = token_idx.unsqueeze(-1).expand(-1, -1, -1, dim)
        k_sel = torch.gather(K.unsqueeze(1).expand(bh, tgt_len, src_len, dim), 2, idx_gather)
        v_sel = torch.gather(V.unsqueeze(1).expand(bh, tgt_len, src_len, dim), 2, idx_gather)

        scores_sel = torch.matmul(Q.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2)

        if token_mask is not None:
            am = token_mask.unsqueeze(1).expand(bsz, self.num_heads, src_len).reshape(bh, src_len)
            allowed = torch.gather(am.unsqueeze(1).expand(bh, tgt_len, src_len), 2, token_idx)
            scores_sel = scores_sel.masked_fill(~allowed, torch.finfo(scores_sel.dtype).min)

        attn = F.softmax(scores_sel, dim=-1)
        return torch.bmm(
            attn.reshape(bh * tgt_len, 1, m_tokens),
            v_sel.reshape(bh * tgt_len, m_tokens, dim),
        ).reshape(bh, tgt_len, dim)

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        BH, tgt_len, _ = Q.shape

        if is_causal:
            out_win = self._causal_window_branch(Q, K, V, bsz, tgt_len, token_mask)
            out_cmp = self._causal_compressed_branch(Q, K, V, bsz, tgt_len, token_mask)
            out_slc = self._causal_selected_branch(Q, K, V, bsz, tgt_len, token_mask)
        else:
            out_win = self._window_branch(Q, K, V, bsz, tgt_len, token_mask)
            out_cmp = self._compressed_branch(Q, K, V, bsz, tgt_len, token_mask)
            out_slc = self._selected_branch(Q, K, V, bsz, tgt_len, token_mask)

        # Fixed gates (no learned gate_mlp)
        w = self.win_weight
        c = self.cmp_weight
        s = self.slc_weight
        total = w + c + s
        return (w / total) * out_win + (c / total) * out_cmp + (s / total) * out_slc


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    block_size: int = 64,
    topk_blocks: int = 16,
    window_size: int = 256,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=FreeNSAAttention,
        num_labels=num_labels,
        attn_kwargs={
            "block_size": block_size,
            "topk_blocks": topk_blocks,
            "window_size": window_size,
            "use_triton": False,
        },
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
