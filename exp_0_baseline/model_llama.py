"""Exp 0 — Dense baseline on R1-Distill-Llama-8B.

Full dense (bidirectional) attention with no sparsity. This is the reference
baseline for comparing sparse attention variants on the same base model.
"""

import os
import torch
from shared.llama_patched_model import (
    LlamaSparseAttention,
    LlamaPatchedModel,
    apply_lora,
)
from shared.sparse_attn_utils import dense_self_attention


class DenseAttention(LlamaSparseAttention):
    """Full dense bidirectional attention — no sparsity."""

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads):
        return dense_self_attention(
            Q, K, V, token_mask, bsz, num_heads, 0.0, self.training
        )


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Build R1-8B with full dense attention + LoRA."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=DenseAttention,
        num_labels=num_labels,
        attn_kwargs={},
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
