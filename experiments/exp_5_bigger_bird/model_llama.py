"""Exp 5 — BiggerBird sparse attention on R1-Distill-Llama-8B.

Combines three sparse-selection mechanisms in a single attention module:
  (1) Diversity-aware LOCAL top-k from a sliding window (MMR-lite)
  (2) Submodular-style GLOBAL selection via a learned gate with coverage penalty
  (3) Biased random TELEPORTS (mix of high-gate tokens + uniform random)
Total selected per query: M = k + G + T -> softmax over M keys instead of N.

Key changes from the BART version:
  - Inherits from ``LlamaSparseAttention`` (handles GQA, RoPE, projections)
  - ``sparse_attention()`` receives already-projected, RoPE'd, GQA-expanded
    Q/K/V as [BH, T, d] — same interface the sparse_attn_utils expect
  - The learned ``global_gate`` operates on ``hidden_states`` [B, T, 4096];
    we capture ``hidden_states`` in a thin ``forward`` override so it is
    available inside ``sparse_attention``.
  - Bidirectional attention (no causal mask) for sequence classification
  - LoRA training instead of full fine-tuning; the gate is kept trainable
    alongside LoRA adapters.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from patches.llama.llama_patched_model import (
    LlamaSparseAttention,
    patch_llama,
    LlamaPatchedModel,
    apply_lora,
)
from sparse_attn_utils import (
    dense_self_attention,
    gather_attention_triton_or_none,
)


class BiggerBirdAttention(LlamaSparseAttention):
    def __init__(
        self,
        base_attn,
        window_size: int = 64,
        local_k: int = 32,
        num_globals: int = 16,
        num_teleports: int = 8,
        diversity_lambda: float = 0.3,
        teleport_bias: float = 0.5,
        use_triton: bool = True,
    ):
        super().__init__(base_attn)
        self.window_size = window_size
        self.local_k = local_k
        self.num_globals = num_globals
        self.num_teleports = num_teleports
        self.diversity_lambda = diversity_lambda
        self.teleport_bias = teleport_bias
        self.use_triton = use_triton

        # Learned global importance gate (submodular surrogate)
        self.global_gate = nn.Linear(self.config.hidden_size, 1)
        nn.init.zeros_(self.global_gate.bias)
        nn.init.normal_(self.global_gate.weight, std=0.02)

    def forward(self, hidden_states, **kwargs):
        # Capture hidden_states so sparse_attention can compute the gate
        self._hidden_states = hidden_states
        return super().forward(hidden_states, **kwargs)

    def _mmr_local_topk(self, Q, K, window_idx, k):
        """Diverse top-k inside a window using a 1-step MMR penalty.

        Q: [BH, Tq, D], K: [BH, Src, D], window_idx: [Tq, W] absolute indices
        Returns: [BH, Tq, k] absolute indices.
        """
        BH, Tq, D = Q.shape
        W = window_idx.size(-1)
        # Gather window keys: [BH, Tq, W, D]
        idx = window_idx.unsqueeze(0).unsqueeze(-1).expand(BH, Tq, W, D)
        K_exp = K.unsqueeze(1).expand(BH, Tq, K.size(1), D)
        K_w = torch.gather(K_exp, 2, idx)
        # Relevance score: q . k for each window position
        rel = torch.einsum("bqd,bqwd->bqw", Q, K_w)  # [BH, Tq, W]
        # Diversity penalty: subtract max similarity to the current top-1
        top1 = rel.argmax(dim=-1, keepdim=True)  # [BH, Tq, 1]
        top1_k = torch.gather(
            K_w, 2, top1.unsqueeze(-1).expand(-1, -1, 1, D)
        )  # [BH, Tq, 1, D]
        sim_to_top1 = torch.einsum(
            "bqwd,bqod->bqw", K_w, top1_k.squeeze(2).unsqueeze(2)
        )
        mmr_scores = rel - self.diversity_lambda * sim_to_top1
        # Take top-k in window (relative indices)
        kk = min(k, W)
        _, sel_rel = torch.topk(mmr_scores, k=kk, dim=-1)  # [BH, Tq, k]
        # Map to absolute indices
        win_idx_exp = window_idx.unsqueeze(0).expand(BH, Tq, W)
        sel_abs = torch.gather(win_idx_exp, 2, sel_rel)
        return sel_abs

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads):
        BH, tgt_len, _ = Q.shape
        src_len = K.size(1)
        M_total = self.local_k + self.num_globals + self.num_teleports

        # ---- Fallback to dense for very short sequences ----
        if src_len <= M_total:
            return dense_self_attention(
                Q, K, V, None, bsz, num_heads, 0.0, self.training
            )

        # ---- (1) LOCAL: diverse top-k within sliding window ----
        t = torch.arange(tgt_len, device=Q.device)
        ws = torch.clamp(
            t - self.window_size // 2,
            min=0,
            max=max(0, src_len - self.window_size),
        )
        offs = torch.arange(self.window_size, device=Q.device)
        window_idx = ws.unsqueeze(1) + offs.unsqueeze(0)  # [Tq, W]
        local_idx = self._mmr_local_topk(
            Q, K, window_idx, self.local_k
        )  # [BH, Tq, k]

        # ---- (2) GLOBAL: learned submodular-style gate ----
        hidden = self._hidden_states
        g_scores = self.global_gate(
            hidden.to(self.global_gate.weight.dtype)
        ).squeeze(-1)  # [B, T]
        if token_mask is not None:
            g_scores = g_scores.masked_fill(~token_mask, -1e9)
        _, g_idx = torch.topk(
            g_scores, k=self.num_globals, dim=-1
        )  # [B, G]
        g_idx_exp = g_idx.unsqueeze(1).expand(bsz, tgt_len, self.num_globals)
        g_idx_exp = g_idx_exp.unsqueeze(1).expand(
            bsz, num_heads, tgt_len, self.num_globals
        )
        g_idx_exp = g_idx_exp.reshape(BH, tgt_len, self.num_globals)

        # ---- (3) TELEPORTS: biased random ----
        n_biased = self.num_teleports // 2
        n_random = self.num_teleports - n_biased
        if n_biased > 0:
            top2g = max(2 * self.num_globals, self.num_teleports)
            _, cand_idx = torch.topk(
                g_scores, k=min(top2g, src_len), dim=-1
            )  # [B, C]
            rand_pick = torch.randint(
                0, cand_idx.size(-1), (bsz, tgt_len, n_biased), device=Q.device
            )
            biased_idx = torch.gather(
                cand_idx.unsqueeze(1).expand(
                    bsz, tgt_len, cand_idx.size(-1)
                ),
                2,
                rand_pick,
            )
            biased_idx = (
                biased_idx.unsqueeze(1)
                .expand(bsz, num_heads, tgt_len, n_biased)
                .reshape(BH, tgt_len, n_biased)
            )
        else:
            biased_idx = torch.empty(
                BH, tgt_len, 0, dtype=torch.long, device=Q.device
            )
        if n_random > 0:
            random_idx = torch.randint(
                0, src_len, (BH, tgt_len, n_random), device=Q.device
            )
        else:
            random_idx = torch.empty(
                BH, tgt_len, 0, dtype=torch.long, device=Q.device
            )
        teleport_idx = torch.cat([biased_idx, random_idx], dim=-1)

        # ---- Combine indices: M = k + G + T ----
        abs_idx = torch.cat(
            [local_idx, g_idx_exp, teleport_idx], dim=-1
        )  # [BH, Tq, M]

        out = gather_attention_triton_or_none(
            Q, K, V, abs_idx, token_mask, bsz, num_heads,
            self.use_triton, self.training,
        )
        if out is None:
            M = abs_idx.size(-1)
            idx_g = abs_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            K_sel = torch.gather(
                K.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim),
                2,
                idx_g,
            )
            V_sel = torch.gather(
                V.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim),
                2,
                idx_g,
            )
            scores_sel = torch.matmul(
                Q.unsqueeze(2), K_sel.transpose(-1, -2)
            ).squeeze(2)  # [BH, Tq, M]
            if token_mask is not None:
                am_bh = (
                    token_mask.unsqueeze(1)
                    .expand(bsz, num_heads, src_len)
                    .reshape(BH, src_len)
                )
                allowed = torch.gather(
                    am_bh.unsqueeze(1).expand(BH, tgt_len, src_len),
                    2,
                    abs_idx,
                )
                scores_sel = scores_sel.masked_fill(~allowed, -1e9)
            attn_probs = F.softmax(scores_sel, dim=-1)
            attn_probs = F.dropout(attn_probs, p=0.0, training=self.training)
            out = torch.bmm(
                attn_probs.reshape(BH * tgt_len, 1, M),
                V_sel.reshape(BH * tgt_len, M, self.head_dim),
            ).reshape(BH, tgt_len, self.head_dim)

        return out


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    window_size: int = 64,
    local_k: int = 32,
    num_globals: int = 16,
    num_teleports: int = 8,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Build the patched R1-8B model with BiggerBird attention + LoRA."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=BiggerBirdAttention,
        num_labels=num_labels,
        attn_kwargs={
            "window_size": window_size,
            "local_k": local_k,
            "num_globals": num_globals,
            "num_teleports": num_teleports,
            "use_triton": False,
        },
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)

    # Re-enable grad for global_gate params (frozen by PEFT)
    for module in model.modules():
        if hasattr(module, "global_gate"):
            module.global_gate.requires_grad_(True)

    return model
