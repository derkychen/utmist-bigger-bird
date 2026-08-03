"""Exp 8 — Token Dropping on R1-Distill-Llama-8B.

After a few early dense layers, low-importance tokens are dropped from the
attention key set so that subsequent layers attend over a shorter sequence.
Importance is estimated from the L2 norm of the key vectors (a proxy for the
hidden-state norm used in the original BART version).  This is the Llama-3
port of the original BART-based experiment.

Key changes from the BART version:
  - Inherits from ``LlamaSparseAttention`` (handles GQA, RoPE, projections)
  - ``sparse_attention()`` receives already-projected, RoPE'd, GQA-expanded
    Q/K/V as [BH, T, d] -- same interface the sparse_attn_utils expect
  - Token dropping is implemented as attention masking (selecting a subset of
    keys) rather than physically removing tokens from the hidden-state sequence,
    since the HF LlamaModel controls the layer loop
  - ``self.layer_idx`` determines whether this layer is before or after
    ``drop_after_layer``
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


class TokenDropAttention(LlamaSparseAttention):
    """Dense attention for early layers; sparse (token-dropped) attention later.

    Layers with ``layer_idx < drop_after_layer`` use full dense attention.
    Later layers attend only over the top ``(1 - drop_ratio)`` tokens ranked by
    key-vector L2 norm (averaged across heads within each batch element).
    """

    def __init__(
        self,
        base_attn,
        drop_after_layer: int = 3,
        drop_ratio: float = 0.3,
        use_triton: bool = False,
    ):
        super().__init__(base_attn)
        self.drop_after_layer = drop_after_layer
        self.drop_ratio = drop_ratio
        self.use_triton = use_triton

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads):
        BH, tgt_len, d = Q.shape
        src_len = K.size(1)

        # --- Early layers: full dense attention ---
        if self.layer_idx < self.drop_after_layer:
            return dense_self_attention(
                Q, K, V, token_mask, bsz, num_heads, 0.0, self.training
            )

        # --- Later layers: attend over top (1 - drop_ratio) tokens ---
        keep_n = max(1, int(src_len * (1.0 - self.drop_ratio)))
        if keep_n >= src_len:
            return dense_self_attention(
                Q, K, V, token_mask, bsz, num_heads, 0.0, self.training
            )

        # Importance proxy: key-vector L2 norm averaged across heads -> [B, T]
        K_reshaped = K.view(bsz, num_heads, src_len, d)
        norms = K_reshaped.norm(dim=-1).mean(dim=1)  # [B, T]
        if token_mask is not None:
            norms = norms.masked_fill(~token_mask, -1e9)

        _, top_idx = torch.topk(norms, k=keep_n, dim=-1)  # [B, keep_n]
        top_idx, _ = torch.sort(top_idx, dim=-1)  # preserve relative order

        # Expand to [BH, keep_n] -- same indices for all heads within a batch
        indices = top_idx.unsqueeze(1).expand(bsz, num_heads, keep_n).reshape(BH, keep_n)

        out = sdpa_head_shared_or_none(
            Q, K, V, indices, None, bsz, num_heads,
            self.use_triton, self.training,
        )
        if out is None:
            out = sparse_attention_head_shared(
                Q, K, V, indices, 0.0, self.training, token_mask, bsz, num_heads
            )
        return out


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    drop_after_layer: int = 3,
    drop_ratio: float = 0.3,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Build the patched R1-8B model with token-dropping attention + LoRA."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=TokenDropAttention,
        num_labels=num_labels,
        attn_kwargs={
            "drop_after_layer": drop_after_layer,
            "drop_ratio": drop_ratio,
            "use_triton": False,
        },
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
