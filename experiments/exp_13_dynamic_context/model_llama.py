"""Exp 13 — Dynamic Context Window on R1-Distill-Llama-8B.

Content-routed sparse attention whose key budget adapts to the input instead
of being a fixed constant.

Causal (generative) path:
  - Routing scores are the QK logits of the last ``num_route_queries`` query
    positions (max over those queries), so the question tokens — not just the
    final prompt token — choose which keys are retrieved.
  - ``budget_mode="length"``: k = clamp(src_len // tokens_per_key,
    min_budget, max_budget). With the defaults this is 512 keys up to 32K,
    1024 at 64K and 2048 at 128K, matching the budgets exp 1 needed at each
    length.
  - ``budget_mode="mass"``: each head keeps the smallest key set covering
    ``mass`` of its routing softmax, clamped to [min_budget, max_budget].
  - The first ``num_sinks`` tokens are always routed (attention sinks).
  - Routed keys inside a query's local window are deduplicated.

Bidirectional (classification) path: head-shared top-``target_budget`` keys.
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
    sdpa_head_shared_or_none,
    sparse_attention_head_shared,
    causal_sparse_attention,
    head_shared_topk_indices,
)


def length_scaled_budget(src_len: int, tokens_per_key: int, min_budget: int, max_budget: int) -> int:
    return min(src_len, max(min_budget, min(max_budget, src_len // tokens_per_key)))


def dynamic_route_indices(
    Q: torch.Tensor,
    K: torch.Tensor,
    budget_mode: str,
    min_budget: int,
    max_budget: int,
    tokens_per_key: int,
    mass: float,
    num_route_queries: int,
    num_sinks: int,
    token_mask=None,
    bsz: int = 1,
    num_heads: int = 32,
):
    """Select routed keys with an input-dependent budget.

    Args:
        Q, K: [BH, T, d] — Q is pre-scaled, so Q @ K^T are attention logits

    Returns:
        indices: [BH, k] routed key indices, best first
        routed_mask: [BH, k] bool, or None when every head uses all k keys
    """
    BH, tgt_len, _ = Q.shape
    src_len = K.size(1)
    neg = torch.finfo(Q.dtype).min

    n = min(num_route_queries, tgt_len)
    scores = torch.bmm(Q[:, -n:, :], K.transpose(1, 2)).amax(dim=1)  # [BH, S]

    am = None
    if token_mask is not None:
        am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
        scores = scores.masked_fill(~am, neg)

    n_sinks = min(num_sinks, src_len)
    if budget_mode == "length":
        need = length_scaled_budget(src_len, tokens_per_key, min_budget, max_budget)
        need = torch.full((BH,), need, device=Q.device, dtype=torch.long)
    elif budget_mode == "mass":
        probs = torch.softmax(scores.float(), dim=-1)
        sorted_probs, _ = probs.sort(dim=-1, descending=True)
        need = (sorted_probs.cumsum(dim=-1) < mass).sum(dim=-1) + 1
        need = need.clamp(min=min_budget, max=max_budget)
    else:
        raise ValueError(f"unknown budget_mode {budget_mode!r}")
    need = (need + n_sinks).clamp(max=src_len)

    if n_sinks > 0:
        scores = scores.clone()
        scores[:, :n_sinks] = torch.finfo(Q.dtype).max
        if am is not None:
            scores = scores.masked_fill(~am, neg)

    k = int(need.max())
    _, indices = torch.topk(scores, k=k, dim=-1)
    if bool((need == k).all()):
        return indices, None
    routed_mask = torch.arange(k, device=Q.device).unsqueeze(0) < need.unsqueeze(1)
    return indices, routed_mask


class DynamicContextAttention(LlamaSparseAttention):
    """Sparse attention with an input-adaptive routed key budget."""

    def __init__(
        self,
        base_attn,
        target_budget: int = 4096,
        budget_mode: str = "length",
        min_budget: int = 512,
        max_budget: int = 4096,
        tokens_per_key: int = 64,
        mass: float = 0.95,
        num_route_queries: int = 1,
        num_sinks: int = 4,
        local_window: int = 256,
        use_triton: bool = False,
    ):
        super().__init__(base_attn)
        self.target_budget = target_budget
        self.budget_mode = budget_mode
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.tokens_per_key = tokens_per_key
        self.mass = mass
        self.num_route_queries = num_route_queries
        self.num_sinks = num_sinks
        self.local_window = local_window
        self.use_triton = use_triton

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        BH, tgt_len, d = Q.shape
        src_len = K.size(1)

        if is_causal:
            routed_idx, routed_mask = dynamic_route_indices(
                Q, K,
                budget_mode=self.budget_mode,
                min_budget=self.min_budget,
                max_budget=self.max_budget,
                tokens_per_key=self.tokens_per_key,
                mass=self.mass,
                num_route_queries=self.num_route_queries,
                num_sinks=self.num_sinks,
                token_mask=token_mask, bsz=bsz, num_heads=num_heads,
            )
            return causal_sparse_attention(
                Q, K, V, routed_idx, local_window=self.local_window,
                token_mask=token_mask, bsz=bsz, num_heads=num_heads,
                routed_mask=routed_mask, dedupe_local=True,
                use_triton=self.use_triton,
            )

        # --- Bidirectional: content-based top-k selection ---
        budget = min(self.target_budget, src_len)

        # Content-based top-k selection using head-shared QK product
        d_low = min(d, 128)
        Q_low = Q[:, :, :d_low]
        K_low = K[:, :, :d_low]
        indices = head_shared_topk_indices(
            Q_low, K_low, budget, token_mask, bsz, num_heads,
        )

        out = sdpa_head_shared_or_none(
            Q, K, V, indices, None, bsz, num_heads,
            self.use_triton, self.training, is_causal=is_causal,
        )
        if out is None:
            out = sparse_attention_head_shared(
                Q, K, V, indices, 0.0, self.training, token_mask, bsz, num_heads,
                is_causal=is_causal,
            )
        return out


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    target_budget: int = 4096,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    pooling: str = "last",
):
    """Build the patched R1-8B model with dynamic-context attention + LoRA."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=DynamicContextAttention,
        num_labels=num_labels,
        attn_kwargs={
            "target_budget": target_budget,
            "use_triton": False,
        },
        pooling=pooling,
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
