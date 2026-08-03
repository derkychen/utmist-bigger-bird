"""Exp 0 — Dense baseline on R1-Distill-Llama-8B.

Full dense attention with no sparsity. This is the reference baseline for
comparing sparse attention variants on the same base model.

Uses **causal** attention (matching how Llama was pretrained) with
last-token pooling for retrieval tasks.  The last token's hidden state
is a natural summary of everything before it — this is how Llama is
designed to be used for classification.
"""

import os
import torch
from patches.llama.llama_patched_model import (
    LlamaSparseAttention,
    LlamaPatchedModel,
    apply_lora,
)
from sparse_attn_utils import dense_self_attention


class DenseAttention(LlamaSparseAttention):
    """Full dense attention — causal to match Llama pretraining."""

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        return dense_self_attention(
            Q, K, V, token_mask, bsz, num_heads, 0.0, self.training,
            is_causal=is_causal,
        )


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    pooling: str = "last",
):
    """Build R1-8B with full dense attention + LoRA."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=DenseAttention,
        num_labels=num_labels,
        attn_kwargs={},
        pooling=pooling,
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
