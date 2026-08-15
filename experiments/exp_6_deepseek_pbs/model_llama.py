"""Exp 6 — DeepSeek PBS Attention on R1-Distill-Llama-8B.

Combines DeepSeek-style low-rank top-k routing with PBS block selection:
block-level scoring via a low-rank proxy selects the most relevant blocks,
then a secondary top-k refinement keeps only the most relevant tokens within
those blocks. This is the Llama-3 port of the original BART-based experiment.

Key changes from the BART version:
  - Inherits from ``LlamaSparseAttention`` (handles GQA, RoPE, projections)
  - ``sparse_attention()`` receives already-projected, RoPE'd, GQA-expanded
    Q/K/V as [BH, T, d] — Q is pre-scaled by self.scaling
  - Bidirectional attention (no causal mask) for sequence classification
  - LoRA training instead of full fine-tuning
"""

import os
import torch
import torch.nn as nn

from patches.llama.llama_patched_model import (
    LlamaSparseAttention,
    LlamaPatchedModel,
    apply_lora,
)
from sparse_attn_utils import (
    dense_self_attention,
    effective_top_k,
    last_query_block_topk_indices,
    causal_sparse_attention,
    sdpa_head_shared_or_none,
    sparse_attention_head_shared,
)


def _effective_num_blocks(num_blocks: int, seq_len: int, block_size: int) -> int:
    n_blocks_k = max(1, seq_len // block_size)
    return min(num_blocks, max(2, n_blocks_k // 2))


class DeepSeekPBSAttention(LlamaSparseAttention):
    """Low-rank block routing (head-shared) + PBS sorted token indices + sparse softmax."""

    def __init__(
        self,
        base_attn,
        top_k: int = 64,
        low_rank_dim: int = 16,
        block_size: int = 32,
        num_blocks: int = 4,
        use_triton: bool = True,
    ):
        super().__init__(base_attn)
        self.top_k = top_k
        self.low_rank_dim = low_rank_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.use_triton = use_triton

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        BH, tgt_len, _ = Q.shape
        src_len = K.size(1)
        k_eff = effective_top_k(self.top_k, src_len)
        M_blocks = _effective_num_blocks(self.num_blocks, src_len, self.block_size)

        # Causal mode: last-query block routing + local window (O(N) attention)
        if is_causal:
            local_w = self.block_size * 4  # local window = 4 blocks
            sparse_budget = k_eff + local_w
            if src_len <= sparse_budget:
                return dense_self_attention(
                    Q, K, V, token_mask, bsz, num_heads, 0.0, self.training,
                    is_causal=True,
                )
            d_low = min(self.low_rank_dim, self.head_dim)
            Q_low = Q[:, :, :d_low]
            K_low = K[:, :, :d_low]
            routed_idx = last_query_block_topk_indices(
                Q_low, K_low, self.block_size, M_blocks,
                token_mask, bsz, num_heads,
            )
            # Secondary top-k refinement if too many tokens
            if routed_idx.size(-1) > k_eff:
                bh = torch.arange(BH, device=Q.device).view(BH, 1)
                K_sub = K[bh, routed_idx, :d_low]
                q_last = Q_low[:, -1:, :]
                rough_tok = torch.bmm(q_last, K_sub.transpose(1, 2)).squeeze(1)
                _, pick = torch.topk(rough_tok, k=k_eff, dim=-1)
                routed_idx = torch.gather(routed_idx, 1, pick)
                routed_idx, _ = torch.sort(routed_idx, dim=-1)
            return causal_sparse_attention(
                Q, K, V, routed_idx, local_window=local_w,
                token_mask=token_mask, bsz=bsz, num_heads=num_heads,
            )

        if src_len <= k_eff or src_len < self.block_size * 2:
            # Fall back to dense attention on short sequences
            return dense_self_attention(
                Q, K, V, None, bsz, num_heads, 0.0, self.training
            )

        # --- Block-level scoring via low-rank proxy ---
        n_blocks_k = max(1, src_len // self.block_size)
        usable_src = n_blocks_k * self.block_size
        d_low = min(self.low_rank_dim, self.head_dim)
        K_low = K[:, :usable_src, :d_low]
        K_blocks = K_low.view(BH, n_blocks_k, self.block_size, d_low).mean(dim=2)
        q_mean = Q[:, :, :d_low].mean(dim=1, keepdim=True)
        block_scores = (
            torch.bmm(q_mean, K_blocks.transpose(1, 2)).squeeze(1)
            / (d_low ** 0.5)
        )

        if token_mask is not None:
            block_mask = (
                token_mask[:, :usable_src]
                .view(bsz, n_blocks_k, self.block_size)
                .any(dim=-1)
            )
            block_mask = block_mask.unsqueeze(1).expand(
                bsz, num_heads, n_blocks_k
            ).reshape(BH, n_blocks_k)
            block_scores = block_scores.masked_fill(~block_mask, -1e9)

        M = min(M_blocks, n_blocks_k)
        _, top_blocks = torch.topk(block_scores, k=M, dim=-1)

        # --- Expand block indices to token indices ---
        offs = torch.arange(self.block_size, device=Q.device)
        base = top_blocks.unsqueeze(-1) * self.block_size
        top_idx = (base.unsqueeze(-1) + offs.view(1, 1, -1)).reshape(
            BH, M * self.block_size
        )
        top_idx, _ = torch.sort(top_idx, dim=-1)

        # --- Secondary top-k refinement if too many tokens ---
        if top_idx.size(-1) > k_eff:
            d_low_full = min(self.low_rank_dim, self.head_dim)
            bh = torch.arange(BH, device=Q.device).view(BH, 1)
            K_sub = K[bh, top_idx, :d_low_full]
            rough_tok = torch.bmm(
                Q[:, :, :d_low_full].mean(dim=1, keepdim=True),
                K_sub.transpose(1, 2),
            ).squeeze(1)
            _, pick = torch.topk(rough_tok, k=k_eff, dim=-1)
            top_idx = torch.gather(top_idx, 1, pick)
            top_idx, _ = torch.sort(top_idx, dim=-1)

        # --- Sparse attention over selected tokens ---
        out = sdpa_head_shared_or_none(
            Q, K, V, top_idx, None, bsz, num_heads,
            self.use_triton, self.training,
        )
        if out is None:
            out = sparse_attention_head_shared(
                Q, K, V, top_idx, 0.0, self.training, token_mask, bsz, num_heads
            )
        return out


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    top_k: int = 64,
    low_rank_dim: int = 16,
    block_size: int = 32,
    num_blocks: int = 4,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Build the patched R1-8B model with DeepSeek PBS attention + LoRA."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=DeepSeekPBSAttention,
        num_labels=num_labels,
        attn_kwargs={
            "top_k": top_k,
            "low_rank_dim": low_rank_dim,
            "block_size": block_size,
            "num_blocks": num_blocks,
            "use_triton": False,
        },
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
