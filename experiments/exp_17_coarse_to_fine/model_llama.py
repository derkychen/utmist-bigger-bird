"""Exp 17 — Coarse-to-Fine sparse attention on R1-Distill-Llama-8B.

Two-stage routing for better long-context coverage at O(N) cost:

  Stage 1 (coarse): Block-level selection — score blocks of ``block_size``
    tokens using block-mean QK product with the last query. Select the top
    ``topk_blocks`` blocks. This gives O(N) routing with block granularity.

  Stage 2 (fine): Token-level top-k within the selected blocks — from the
    ``topk_blocks * block_size`` candidate tokens, select the top
    ``fine_k`` tokens using full-dim QK product with the last query.

  Additionally, a local sliding window of ``window_size`` is always included
  via causal_sparse_attention, ensuring every query has at least W valid keys.

Total keys per query: W + fine_k (deduplicated by softmax masking).
Complexity: O(N * block_size) for coarse routing + O(topk_blocks * block_size * d)
  for fine routing + O(N * (W + fine_k)) for attention = O(N) when all budgets
  are fixed constants.

This approach has better effective coverage than flat top-k (exp 1) because:
  - Coarse stage screens N tokens at block granularity (cheap)
  - Fine stage only scores the ~topk_blocks * block_size survivors (expensive
    but bounded)
  - At 128K with block_size=128, topk_blocks=32, fine_k=512:
    coarse screens 1024 blocks, fine scores 4096 tokens, attention uses 768 keys
    → effective coverage from 4096 candidates instead of 128K

Parameter-free: uses the model's existing Q/K/V weights for all routing.
No learned components, so it works in zero-shot RULER eval without training.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from patches.llama.llama_patched_model import (
    LlamaSparseAttention,
    LlamaPatchedModel,
    apply_lora,
)
from sparse_attn_utils import causal_sparse_attention, last_query_topk_indices, effective_top_k


class CoarseToFineAttention(LlamaSparseAttention):
    """Two-stage block-then-token sparse attention with local window."""

    def __init__(
        self,
        base_attn,
        block_size: int = 128,
        topk_blocks: int = 32,
        fine_k: int = 512,
        window_size: int = 256,
        use_triton: bool = False,
    ):
        super().__init__(base_attn)
        self.block_size = block_size
        self.topk_blocks = topk_blocks
        self.fine_k = fine_k
        self.window_size = window_size
        self.use_triton = use_triton

    def _coarse_to_fine_indices(self, Q, K, token_mask, bsz, num_heads):
        """Two-stage routing: block selection → token top-k within blocks.

        Returns [BH, fine_k] token indices (unsorted), head-shared.
        """
        BH, tgt_len, d = Q.shape
        src_len = K.size(1)
        blk = self.block_size

        # --- Stage 1: Coarse block selection via block-mean QK ---
        n_blocks = max(1, (src_len + blk - 1) // blk)
        m_blocks = min(self.topk_blocks, n_blocks)

        # Fast path: if all blocks are selected, skip coarse stage and do
        # direct token-level top-k (identical to exp 1's code path).
        # This avoids numerical precision differences from gather + sort.
        if m_blocks >= n_blocks:
            d_low = min(d, 128)
            Q_low = Q[:, :, :d_low]
            K_low = K[:, :, :d_low]
            k_eff = effective_top_k(self.fine_k, src_len, min_k=64, ratio=2)
            return last_query_topk_indices(
                Q_low, K_low, k_eff, token_mask, bsz, num_heads,
            )

        # Coarse stage: score blocks using max-pooling of per-token QK scores.
        # Max-pooling preserves the needle's distinctive signal (a few high-scoring
        # tokens in a block), unlike mean-pooling which dilutes it with filler.
        pad = n_blocks * blk - src_len
        if pad > 0:
            K_padded = F.pad(K, (0, 0, 0, pad), value=0.0)
        else:
            K_padded = K

        q_last = Q[:, -1:, :]  # [BH, 1, d]
        # Per-token QK scores with the last query (low-rank for RoPE robustness at long context)
        d_low = min(d, 128)
        q_last_low = q_last[:, :, :d_low]
        K_low = K_padded[:, :, :d_low]
        token_scores = torch.bmm(q_last_low, K_low.transpose(1, 2)).squeeze(1) / (d_low ** 0.5)  # [BH, n_blocks * blk]
        # Max-pool per block: the block score is the best token score in the block
        block_scores = token_scores.view(BH, n_blocks, blk).max(dim=-1).values  # [BH, n_blocks]

        if token_mask is not None:
            padded_mask = F.pad(token_mask, (0, pad), value=False) if pad > 0 else token_mask
            block_mask = padded_mask.view(bsz, n_blocks, blk).any(dim=-1)
            block_mask = block_mask.unsqueeze(1).expand(bsz, num_heads, n_blocks).reshape(BH, n_blocks)
            block_scores = block_scores.masked_fill(~block_mask, torch.finfo(block_scores.dtype).min)

        _, top_blocks = torch.topk(block_scores, k=m_blocks, dim=-1)

        # Expand block indices to token indices
        offs = torch.arange(blk, device=Q.device)
        base = top_blocks.unsqueeze(-1) * blk
        candidate_idx = (base + offs.view(1, 1, -1)).reshape(BH, m_blocks * blk)
        candidate_idx = candidate_idx.clamp(max=src_len - 1)

        # --- Stage 2: Fine token selection within candidate blocks ---
        n_candidates = candidate_idx.size(-1)
        k_fine = min(self.fine_k, n_candidates)

        if k_fine < n_candidates:
            bh_idx = torch.arange(BH, device=Q.device).view(BH, 1).expand(-1, n_candidates)
            K_cand = K[bh_idx, candidate_idx, :]

            # Use low-rank projection for consistency with exp 1
            d_low = min(d, 128)
            q_last_low = Q[:, -1, :d_low]  # [BH, d_low]
            K_cand_low = K_cand[:, :, :d_low]  # [BH, n_candidates, d_low]
            fine_scores = torch.bmm(q_last_low.unsqueeze(1), K_cand_low.transpose(1, 2)).squeeze(1) / (d_low ** 0.5)

            if token_mask is not None:
                am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
                cand_valid = torch.gather(am, 1, candidate_idx)
                fine_scores = fine_scores.masked_fill(~cand_valid, torch.finfo(fine_scores.dtype).min)

            _, fine_top = torch.topk(fine_scores, k=k_fine, dim=-1)
            routed_idx = torch.gather(candidate_idx, 1, fine_top)
        else:
            routed_idx = candidate_idx

        # Don't sort — keep topk order for numerical consistency with exp 1
        return routed_idx

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        BH, tgt_len, _ = Q.shape
        src_len = K.size(1)

        if is_causal:
            routed_idx = self._coarse_to_fine_indices(Q, K, token_mask, bsz, num_heads)
            return causal_sparse_attention(
                Q, K, V, routed_idx, local_window=self.window_size,
                token_mask=token_mask, bsz=bsz, num_heads=num_heads,
            )

        # Bidirectional mode: per-query coarse-to-fine routing with max-pool block scoring
        d = self.head_dim
        blk = self.block_size
        n_blocks = max(1, (src_len + blk - 1) // blk)
        pad = n_blocks * blk - src_len

        K_padded = F.pad(K, (0, 0, 0, pad), value=0.0) if pad > 0 else K
        # Per-token QK scores: [BH, Tq, n_blocks * blk]
        token_scores = torch.bmm(Q, K_padded.transpose(1, 2)) / (d ** 0.5)
        # Max-pool per block
        block_scores = token_scores.view(BH, tgt_len, n_blocks, blk).max(dim=-1).values  # [BH, Tq, n_blocks]

        if token_mask is not None:
            padded_mask = F.pad(token_mask, (0, pad), value=False) if pad > 0 else token_mask
            block_mask = padded_mask.view(bsz, n_blocks, blk).any(dim=-1)
            block_mask = block_mask.unsqueeze(1).expand(bsz, num_heads, n_blocks).reshape(BH, 1, n_blocks).expand(-1, tgt_len, -1)
            block_scores = block_scores.masked_fill(~block_mask, torch.finfo(block_scores.dtype).min)

        m_blocks = min(self.topk_blocks, n_blocks)
        _, top_blocks = torch.topk(block_scores, k=m_blocks, dim=-1)  # [BH, Tq, m_blocks]

        block_offset = torch.arange(blk, device=Q.device).view(1, 1, 1, blk)
        base_idx = (top_blocks.unsqueeze(2) * blk).unsqueeze(-1)
        candidate_idx = (base_idx + block_offset).reshape(BH, tgt_len, m_blocks * blk).clamp(max=src_len - 1)

        # Fine selection per query
        n_cand = candidate_idx.size(-1)
        k_fine = min(self.fine_k, n_cand)

        if k_fine < n_cand:
            bh_idx_c = torch.arange(BH, device=Q.device).view(BH, 1, 1).expand(-1, tgt_len, n_cand)
            K_cand = K[bh_idx_c, candidate_idx, :]  # [BH, Tq, n_cand, d]
            fine_scores = torch.einsum("btd,btcd->btc", Q, K_cand) / (d ** 0.5)  # [BH, Tq, n_cand]

            if token_mask is not None:
                am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
                cand_valid = torch.gather(am.unsqueeze(1).expand(BH, tgt_len, src_len), 2, candidate_idx)
                fine_scores = fine_scores.masked_fill(~cand_valid, torch.finfo(fine_scores.dtype).min)

            _, fine_top = torch.topk(fine_scores, k=k_fine, dim=-1)  # [BH, Tq, k_fine]
            all_idx = torch.gather(candidate_idx, 2, fine_top)  # [BH, Tq, k_fine]
        else:
            all_idx = candidate_idx

        # Add local window
        t = torch.arange(tgt_len, device=Q.device)
        w = min(self.window_size, src_len)
        starts = torch.clamp(t - w // 2, min=0, max=max(0, src_len - w))
        offsets = torch.arange(w, device=Q.device)
        local_idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(max=src_len - 1)
        local_idx = local_idx.unsqueeze(0).expand(BH, -1, -1)

        all_idx = torch.cat([all_idx, local_idx], dim=-1)  # [BH, Tq, k_fine + w]

        # Gather attention
        M = all_idx.size(-1)
        bh_idx = torch.arange(BH, device=Q.device).view(BH, 1, 1).expand(-1, tgt_len, M)
        k_sel = K[bh_idx, all_idx, :]  # [BH, Tq, M, d]
        v_sel = V[bh_idx, all_idx, :]

        scores = torch.matmul(Q.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2)

        if token_mask is not None:
            am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
            allowed = torch.gather(am.unsqueeze(1).expand(BH, tgt_len, src_len), 2, all_idx)
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        return torch.bmm(
            attn.reshape(BH * tgt_len, 1, M),
            v_sel.reshape(BH * tgt_len, M, d),
        ).reshape(BH, tgt_len, d)


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    block_size: int = 128,
    topk_blocks: int = 32,
    fine_k: int = 512,
    window_size: int = 256,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=CoarseToFineAttention,
        num_labels=num_labels,
        attn_kwargs={
            "block_size": block_size,
            "topk_blocks": topk_blocks,
            "fine_k": fine_k,
            "window_size": window_size,
            "use_triton": False,
        },
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
