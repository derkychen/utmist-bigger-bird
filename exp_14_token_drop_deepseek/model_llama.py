"""Exp 14 — Token Dropping + DeepSeek Top-K on R1-Distill-Llama-8B.

Combines two sparsity mechanisms:
  1. Token dropping: after a few early dense layers, low-importance tokens are
     dropped from the key set (importance estimated from key-vector L2 norm).
  2. DeepSeek top-k: on the remaining tokens, a low-rank proxy selects the
     top-k most relevant keys per head (shared across query positions).

This gives two sparsity wins: (1) shorter key sequence via token dropping,
(2) fewer keys per query via top-k routing.

This is the Llama-3 port of the original BART-based experiment.  Key changes:
  - Inherits from ``LlamaSparseAttention`` (handles GQA, RoPE, projections)
  - ``sparse_attention()`` receives already-projected, RoPE'd, GQA-expanded
    Q/K/V as [BH, T, d] -- Q is pre-scaled, so no extra scaling is applied
  - ``self.layer_idx`` determines whether this layer is before or after
    ``drop_after_layer``
  - Token dropping is implemented as attention masking (selecting a subset of
    keys) rather than physically removing tokens from the hidden-state sequence,
    since the HF LlamaModel controls the layer loop
  - Bidirectional attention (no causal mask) for sequence classification
  - LoRA training instead of full fine-tuning
"""

import os
import torch
import torch.nn as nn

from shared.llama_patched_model import (
    LlamaSparseAttention,
    LlamaPatchedModel,
    apply_lora,
)
from shared.sparse_attn_utils import (
    dense_self_attention,
    effective_top_k,
    head_shared_topk_indices,
    last_query_topk_indices,
    causal_sparse_attention,
    sdpa_head_shared_or_none,
    sparse_attention_head_shared,
)


class TokenDropDeepSeekAttention(LlamaSparseAttention):
    """Dense attention for early layers; token-drop + DeepSeek top-k later.

    Layers with ``layer_idx < drop_after_layer`` use full dense attention.
    Later layers first select the top ``(1 - drop_ratio)`` tokens by key-vector
    L2 norm (head-shared), then apply DeepSeek-style low-rank top-k routing on
    the remaining tokens.
    """

    def __init__(
        self,
        base_attn,
        drop_after_layer: int = 3,
        drop_ratio: float = 0.3,
        top_k: int = 64,
        low_rank_dim: int = 16,
        use_triton: bool = False,
    ):
        super().__init__(base_attn)
        self.drop_after_layer = drop_after_layer
        self.drop_ratio = drop_ratio
        self.top_k = top_k
        self.low_rank_dim = low_rank_dim
        self.use_triton = use_triton

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        BH, tgt_len, d = Q.shape
        src_len = K.size(1)

        # --- Early layers: full dense attention ---
        if self.layer_idx < self.drop_after_layer:
            return dense_self_attention(
                Q, K, V, token_mask, bsz, num_heads, 0.0, self.training,
                is_causal=is_causal,
            )

        # Causal mode: two-stage routing + local window (O(N) attention)
        if is_causal:
            k_eff = effective_top_k(self.top_k, src_len, min_k=64, ratio=2)
            if src_len <= k_eff + 256:
                return dense_self_attention(
                    Q, K, V, token_mask, bsz, num_heads, 0.0, self.training,
                    is_causal=True,
                )
            # Stage 1: norm-based drop (cap at 512 for O(N))
            keep_n = max(1, int(src_len * (1.0 - self.drop_ratio)))
            keep_n = min(keep_n, 512)
            if keep_n < src_len:
                K_reshaped = K.view(bsz, num_heads, src_len, d)
                norms = K_reshaped.norm(dim=-1).mean(dim=1)
                if token_mask is not None:
                    norms = norms.masked_fill(~token_mask, -1e9)
                _, top_idx = torch.topk(norms, k=keep_n, dim=-1)
                top_idx, _ = torch.sort(top_idx, dim=-1)
                indices_drop = top_idx.unsqueeze(1).expand(bsz, num_heads, keep_n).reshape(BH, keep_n)
                idx_exp = indices_drop.unsqueeze(-1).expand(-1, -1, d)
                K_kept = torch.gather(K, 1, idx_exp)
                V_kept = torch.gather(V, 1, idx_exp)
            else:
                K_kept = K
                V_kept = V
                keep_n = src_len

            # Stage 2: low-rank top-k on kept tokens (using last query)
            k_final = min(k_eff, keep_n)
            d_low = min(self.low_rank_dim, self.head_dim)
            Q_low = Q[:, :, :d_low]
            K_low = K_kept[:, :, :d_low]
            routed_idx = last_query_topk_indices(
                Q_low, K_low, k_final, None, bsz, num_heads,
            )
            # Map routed indices back to original K positions
            if keep_n < src_len:
                # routed_idx indexes into K_kept; map back to original indices
                idx_map = indices_drop  # [BH, keep_n]
                routed_idx = torch.gather(idx_map, 1, routed_idx)  # [BH, k_final]

            return causal_sparse_attention(
                Q, K, V, routed_idx, local_window=256,
                token_mask=token_mask, bsz=bsz, num_heads=num_heads,
            )

        # --- Later layers: token drop + DeepSeek top-k (bidirectional only) ---

        # Step 1: Select top (1 - drop_ratio) tokens by importance (K norms)
        keep_n = max(1, int(src_len * (1.0 - self.drop_ratio)))
        if keep_n < src_len:
            K_reshaped = K.view(bsz, num_heads, src_len, d)
            norms = K_reshaped.norm(dim=-1).mean(dim=1)  # [B, T]
            if token_mask is not None:
                norms = norms.masked_fill(~token_mask, -1e9)
            _, top_idx = torch.topk(norms, k=keep_n, dim=-1)  # [B, keep_n]
            top_idx, _ = torch.sort(top_idx, dim=-1)  # preserve relative order

            # Gather K, V at kept positions
            indices_drop = top_idx.unsqueeze(1).expand(bsz, num_heads, keep_n).reshape(BH, keep_n)
            idx_exp = indices_drop.unsqueeze(-1).expand(-1, -1, d)
            K_kept = torch.gather(K, 1, idx_exp)  # [BH, keep_n, d]
            V_kept = torch.gather(V, 1, idx_exp)  # [BH, keep_n, d]

            # Update token mask for kept positions
            if token_mask is not None:
                token_mask_kept = torch.gather(token_mask, 1, top_idx)  # [B, keep_n]
            else:
                token_mask_kept = None
        else:
            K_kept = K
            V_kept = V
            token_mask_kept = token_mask
            keep_n = src_len

        # Step 2: DeepSeek top-k on the kept tokens
        k_eff = effective_top_k(self.top_k, keep_n, min_k=64, ratio=2)
        if keep_n <= k_eff:
            return dense_self_attention(
                Q, K_kept, V_kept, token_mask_kept, bsz, num_heads, 0.0, self.training,
            )

        d_low = min(self.low_rank_dim, self.head_dim)
        Q_low = Q[:, :, :d_low]
        K_low = K_kept[:, :, :d_low]
        topk_idx = head_shared_topk_indices(
            Q_low, K_low, k_eff, token_mask_kept, bsz, num_heads
        )

        out = sdpa_head_shared_or_none(
            Q, K_kept, V_kept, topk_idx, None, bsz, num_heads,
            self.use_triton, self.training,
        )
        if out is None:
            out = sparse_attention_head_shared(
                Q, K_kept, V_kept, topk_idx, 0.0, self.training,
                token_mask_kept, bsz, num_heads,
            )
        return out


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    drop_after_layer: int = 3,
    drop_ratio: float = 0.3,
    top_k: int = 128,
    low_rank_dim: int = 64,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    pooling: str = "last",
):
    """Build the patched R1-8B model with token-drop + DeepSeek top-k attention + LoRA."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=TokenDropDeepSeekAttention,
        num_labels=num_labels,
        attn_kwargs={
            "drop_after_layer": drop_after_layer,
            "drop_ratio": drop_ratio,
            "top_k": top_k,
            "low_rank_dim": low_rank_dim,
            "use_triton": False,
        },
        pooling=pooling,
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
