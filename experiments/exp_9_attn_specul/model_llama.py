"""Exp 9 — Attention Speculation on R1-Distill-Llama-8B.

Inspired by speculative decoding: run a CHEAP attention path (local window +
anchor tokens) and a FULL attention path occasionally to teach the cheap path
via KL divergence.

  - Fast path: each query attends to its local window (``window_size`` tokens)
    plus a few evenly-spaced ``num_anchors`` anchor tokens.  Cost: O(n * (W + A)).
  - Verifier path: full O(n^2) attention, applied only every ``verify_every``
    layers to provide a KL signal during training.

At inference: only the fast path runs.

This is the Llama-3 port of the original BART-based experiment.  Key changes:
  - Inherits from ``LlamaSparseAttention`` (handles GQA, RoPE, projections)
  - ``sparse_attention()`` receives already-projected, RoPE'd, GQA-expanded
    Q/K/V as [BH, T, d] -- Q is pre-scaled, so no extra scaling is applied
  - ``self.layer_idx`` determines whether this is a verify layer
    (``layer_idx % verify_every == 0``)
  - Bidirectional attention (no causal mask) for sequence classification
  - LoRA training instead of full fine-tuning
  - KL loss is collected from attention modules and added to the classification
    loss via a ``LlamaPatchedModel`` subclass
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

from patches.llama.llama_patched_model import (
    LlamaSparseAttention,
    LlamaPatchedModel,
    apply_lora,
)
from sparse_attn_utils import gather_attention_triton_or_none


class AttnSpeculAttention(LlamaSparseAttention):
    """Attention speculation: local window + anchors (fast) with optional KL verify."""

    def __init__(
        self,
        base_attn,
        window_size: int = 64,
        num_anchors: int = 4,
        verify_every: int = 4,
        verify_kl_weight: float = 0.1,
        use_triton: bool = False,
    ):
        super().__init__(base_attn)
        self.window_size = window_size
        self.num_anchors = num_anchors
        self.verify_every = verify_every
        self.verify_kl_weight = verify_kl_weight
        self.use_triton = use_triton
        self.verify = (self.layer_idx % verify_every == 0)
        self.last_kl = None  # populated when verify=True during training

    def _anchor_indices(self, src_len, device):
        """Pick ``num_anchors`` evenly spaced anchors (first, last, and intermediate)."""
        if self.num_anchors <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if self.num_anchors >= src_len:
            return torch.arange(src_len, device=device)
        positions = torch.linspace(0, src_len - 1, steps=self.num_anchors, device=device).long()
        return positions

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads):
        BH, tgt_len, d = Q.shape
        src_len = K.size(1)

        # --- Build sparse index set: window around each query + anchors ---
        t = torch.arange(tgt_len, device=Q.device)
        win_start = torch.clamp(
            t - self.window_size // 2,
            min=0,
            max=max(0, src_len - self.window_size),
        )
        offs = torch.arange(self.window_size, device=Q.device)
        win_idx = win_start.unsqueeze(1) + offs.unsqueeze(0)  # [Tq, W]
        anchors = self._anchor_indices(src_len, Q.device)  # [A]
        anchors_exp = anchors.unsqueeze(0).expand(tgt_len, -1)  # [Tq, A]
        # NOTE: window + anchors may overlap; we keep duplicates for simplicity
        # (softmax over duplicates is mathematically equivalent to summing their weights).
        abs_idx = torch.cat([win_idx, anchors_exp], dim=-1)  # [Tq, W+A]
        abs_idx, _ = torch.sort(abs_idx, dim=-1)
        # Expand to BH
        abs_idx_bh = abs_idx.unsqueeze(0).expand(BH, tgt_len, abs_idx.size(-1))
        M = abs_idx_bh.size(-1)

        # --- Fast path: gather + softmax (fused Triton at inference) ---
        out_fast = gather_attention_triton_or_none(
            Q, K, V, abs_idx_bh, token_mask, bsz, num_heads,
            self.use_triton, self.training,
        )
        scores_fast = None
        if out_fast is None:
            idx_g = abs_idx_bh.unsqueeze(-1).expand(-1, -1, -1, d)
            K_sel = torch.gather(
                K.unsqueeze(1).expand(BH, tgt_len, src_len, d), 2, idx_g
            )
            V_sel = torch.gather(
                V.unsqueeze(1).expand(BH, tgt_len, src_len, d), 2, idx_g
            )
            scores_fast = torch.matmul(Q.unsqueeze(2), K_sel.transpose(-1, -2)).squeeze(2)

            if token_mask is not None:
                am_bh = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
                allowed = torch.gather(
                    am_bh.unsqueeze(1).expand(BH, tgt_len, src_len), 2, abs_idx_bh
                )
                scores_fast = scores_fast.masked_fill(~allowed, -1e9)

            attn_fast = F.softmax(scores_fast, dim=-1)
            attn_fast = F.dropout(attn_fast, p=0.0, training=self.training)
            out_fast = torch.bmm(
                attn_fast.reshape(BH * tgt_len, 1, M),
                V_sel.reshape(BH * tgt_len, M, d),
            ).reshape(BH, tgt_len, d)

        # --- Verifier path (training-only, on verify layers) ---
        if self.verify and self.training and scores_fast is not None:
            full_scores = torch.bmm(Q, K.transpose(1, 2))
            if token_mask is not None:
                me = token_mask.unsqueeze(1).unsqueeze(1).expand(
                    bsz, num_heads, tgt_len, src_len
                ).reshape(BH, tgt_len, src_len)
                full_scores = full_scores.masked_fill(~me, -1e9)
            full_log_probs = F.log_softmax(full_scores, dim=-1)
            # Compute the cheap path's probabilities over the full vocabulary by scattering
            fast_log_probs_sparse = F.log_softmax(scores_fast, dim=-1)
            full_probs_at_sparse = torch.gather(full_log_probs, 2, abs_idx_bh)
            # KL(fast || full) on the sparse support -- encourages fast distribution to align
            kl = (
                fast_log_probs_sparse.exp()
                * (fast_log_probs_sparse - full_probs_at_sparse)
            ).sum(dim=-1).mean()
            self.last_kl = kl * self.verify_kl_weight
        else:
            self.last_kl = None

        return out_fast


class AttnSpeculLlamaModel(LlamaPatchedModel):
    """LlamaPatchedModel with KL loss collection from attention speculation modules."""

    def _collect_kl(self):
        total = None
        for m in self.model.modules():
            if isinstance(m, AttnSpeculAttention) and m.last_kl is not None:
                total = m.last_kl if total is None else total + m.last_kl
        return total

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = super().forward(input_ids, attention_mask, labels, **kwargs)
        if out.loss is not None and self.training:
            kl_loss = self._collect_kl()
            if kl_loss is not None:
                out = SequenceClassifierOutput(
                    loss=out.loss + kl_loss, logits=out.logits
                )
        return out


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    window_size: int = 64,
    num_anchors: int = 4,
    verify_every: int = 4,
    verify_kl_weight: float = 0.1,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Build the patched R1-8B model with attention speculation + LoRA."""
    model = AttnSpeculLlamaModel.from_pretrained(
        model_path=model_path,
        attention_cls=AttnSpeculAttention,
        num_labels=num_labels,
        attn_kwargs={
            "window_size": window_size,
            "num_anchors": num_anchors,
            "verify_every": verify_every,
            "verify_kl_weight": verify_kl_weight,
            "use_triton": False,
        },
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
