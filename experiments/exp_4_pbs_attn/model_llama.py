"""Exp 4 — Probabilistic Block Sparse (PBS) Attention on R1-Distill-Llama-8B.

Selects num_blocks blocks of block_size tokens per head using block-level
similarity scoring, then attends only over those tokens. This is the Llama-3
port of the original BART-based experiment.

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
    sdpa_head_shared_or_none,
    sparse_attention_head_shared,
)


def _effective_num_blocks(num_blocks: int, seq_len: int, block_size: int) -> int:
    n_blocks_k = max(1, seq_len // block_size)
    return min(num_blocks, max(2, n_blocks_k // 2))


class PBSAttention(LlamaSparseAttention):
    def __init__(
        self,
        base_attn,
        block_size: int = 64,
        num_blocks: int = 2,
        use_triton: bool = True,
    ):
        super().__init__(base_attn)
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.use_triton = use_triton

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads):
        BH, tgt_len, _ = Q.shape
        src_len = K.size(1)
        M_blocks = _effective_num_blocks(self.num_blocks, src_len, self.block_size)

        if src_len <= self.block_size * M_blocks:
            # Fall back to dense attention on short sequences
            return dense_self_attention(
                Q, K, V, None, bsz, num_heads, 0.0, self.training
            )

        # --- Block-level scoring ---
        n_blocks_q = max(1, tgt_len // self.block_size)
        n_blocks_k = max(1, src_len // self.block_size)
        usable_src = n_blocks_k * self.block_size

        Q_blocks = (
            Q[:, : n_blocks_q * self.block_size, :]
            .view(BH, n_blocks_q, self.block_size, self.head_dim)
            .mean(dim=2)
        )
        K_blocks = (
            K[:, :usable_src, :]
            .view(BH, n_blocks_k, self.block_size, self.head_dim)
            .mean(dim=2)
        )

        q_pool = Q_blocks.mean(dim=1)  # [BH, d]
        block_scores = torch.bmm(
            q_pool.unsqueeze(1), K_blocks.transpose(1, 2)
        ).squeeze(1)  # [BH, n_blocks_k]
        M = min(M_blocks, n_blocks_k)
        _, top_blocks = torch.topk(block_scores, k=M, dim=-1)

        # --- Expand block indices to token indices ---
        offs = torch.arange(self.block_size, device=Q.device)
        base = top_blocks.unsqueeze(-1) * self.block_size
        top_idx = (base.unsqueeze(-1) + offs.view(1, 1, -1)).reshape(
            BH, M * self.block_size
        )
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
    block_size: int = 64,
    num_blocks: int = 2,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Build the patched R1-8B model with PBS attention + LoRA."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=PBSAttention,
        num_labels=num_labels,
        attn_kwargs={
            "block_size": block_size,
            "num_blocks": num_blocks,
            "use_triton": False,
        },
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
