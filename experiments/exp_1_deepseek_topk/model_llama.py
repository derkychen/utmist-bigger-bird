"""Exp 1 — DeepSeek Top-K sparse attention on R1-Distill-Llama-8B.

Selects the top-k most relevant keys per head (shared across query positions)
using a low-rank proxy, then attends only over those keys. This is the Llama-3
port of the original BART-based experiment. Causal (generative) attention uses
the Triton sparse flash kernel from exp 19 on CUDA fp16/bf16 and falls back to
the PyTorch gather path elsewhere.

Key changes from the BART version:
  - Inherits from ``LlamaSparseAttention`` (handles GQA, RoPE, projections)
  - ``sparse_attention()`` receives already-projected, RoPE'd, GQA-expanded
    Q/K/V as [BH, T, d] — same interface the sparse_attn_utils expect
  - Bidirectional attention (no causal mask) for sequence classification
  - LoRA training instead of full fine-tuning
"""

import os
import torch
import torch.nn as nn

from patches.llama.llama_patched_model import (
    LlamaSparseAttention,
    patch_llama,
    LlamaPatchedModel,
    apply_lora,
)
from sparse_attn_utils import (
    effective_top_k,
    head_shared_topk_indices,
    last_query_topk_indices,
    causal_sparse_attention,
    sdpa_head_shared_or_none,
    sparse_attention_head_shared,
)

try:
    from kernels.bigger_bird_flash import bigger_bird_flash
except ImportError:  # no Triton (e.g. macOS); the torch backend is used
    bigger_bird_flash = None

LOCAL_WINDOW = 256
_FLASH_DTYPES = (torch.float16, torch.bfloat16)


class DeepSeekTopKAttention(LlamaSparseAttention):
    def __init__(self, base_attn, top_k: int = 128, low_rank_dim: int = 16,
                 use_triton: bool = True, query_chunk: int = 256,
                 attn_backend: str = "flash"):
        super().__init__(base_attn)
        if query_chunk < 1:
            raise ValueError("query_chunk must be at least 1")
        if attn_backend not in ("flash", "torch"):
            raise ValueError("attn_backend must be 'flash' or 'torch'")
        self.top_k = top_k
        self.low_rank_dim = low_rank_dim
        self.use_triton = use_triton
        self.query_chunk = query_chunk
        self.attn_backend = attn_backend
        self.last_backend = None  # backend the last causal call actually used
        self.capture_first_route = False
        self.first_routed_indices = None

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        BH, tgt_len, _ = Q.shape
        src_len = K.size(1)

        # Causal mode: use last-query routing + local window (O(N) attention)
        if is_causal:
            k_eff = effective_top_k(self.top_k, src_len, min_k=64, ratio=2)
            d_low = min(self.low_rank_dim, self.head_dim)
            Q_low = Q[:, :, :d_low]
            K_low = K[:, :, :d_low]
            routed_idx = last_query_topk_indices(
                Q_low, K_low, k_eff, token_mask, bsz, num_heads,
            )
            if self.capture_first_route and self.first_routed_indices is None:
                self.first_routed_indices = routed_idx.detach().cpu()
            return self._causal_attention(Q, K, V, routed_idx, token_mask, bsz, num_heads)

        # Bidirectional mode: original head-shared routing
        k_eff = effective_top_k(self.top_k, src_len, min_k=64, ratio=2)

        d_low = min(self.low_rank_dim, self.head_dim)
        Q_low = Q[:, :, :d_low]
        K_low = K[:, :, :d_low]
        topk_idx = head_shared_topk_indices(
            Q_low, K_low, k_eff, token_mask, bsz, num_heads
        )
        out = sdpa_head_shared_or_none(
            Q, K, V, topk_idx, None, bsz, num_heads,
            self.use_triton, self.training, is_causal=is_causal,
        )
        if out is None:
            out = sparse_attention_head_shared(
                Q, K, V, topk_idx, 0.0, self.training, token_mask, bsz, num_heads,
                is_causal=is_causal,
            )
        return out

    def _causal_attention(self, Q, K, V, routed_idx, token_mask, bsz, num_heads):
        """Local window + routed keys; sparse flash kernel on CUDA fp16/bf16."""
        if (self.attn_backend == "flash" and bigger_bird_flash is not None
                and Q.is_cuda and Q.dtype in _FLASH_DTYPES):
            self.last_backend = "flash"
            return bigger_bird_flash(
                Q, K, V, routed_idx[:, None, :], front=0, window=LOCAL_WINDOW,
                token_mask=token_mask, num_heads=num_heads, scale=1.0,
            )
        self.last_backend = "torch"
        return causal_sparse_attention(
            Q, K, V, routed_idx, local_window=LOCAL_WINDOW,
            token_mask=token_mask, bsz=bsz, num_heads=num_heads,
            query_chunk=self.query_chunk,
        )


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    top_k: int = 128,
    low_rank_dim: int = 64,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    pooling: str = "last",
):
    """Build the patched R1-8B model with DeepSeek top-k attention + LoRA."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=DeepSeekTopKAttention,
        num_labels=num_labels,
        attn_kwargs={
            "top_k": top_k,
            "low_rank_dim": low_rank_dim,
            "use_triton": False,  # safer on MIG; PyTorch fallback works
        },
        pooling=pooling,
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
