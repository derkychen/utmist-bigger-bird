"""Exp 18: confidence-gated multi-resolution sparse attention on R1-Llama.

The method keeps exact local attention and a low-rank linear global branch for
all queries.  It computes an exact routed global branch only for attention
heads whose last-query remote score exceeds their local score by the configured
confidence margin.  The optimized causal path uses fused Triton window,
causal-linear, and routed-window kernels when available; all other paths remain
sparse reference implementations rather than dense attention.
"""

from __future__ import annotations

import os

from patches.llama.llama_patched_model import (
    LlamaPatchedModel,
    LlamaSparseAttention,
    apply_lora,
)

from .attention_core import ConfidenceGatedAttentionCore


class ConfidenceGatedAttention(LlamaSparseAttention):
    """Confidence-gated local + linear + routed sparse attention."""

    def __init__(
        self,
        base_attn,
        top_k: int = 512,
        low_rank_dim: int = 128,
        window_size: int = 256,
        gate_threshold: float = 0.5,
        peak_threshold: float = -1.0,
        linear_weight: float = 0.5,
        use_triton: bool = True,
        always_global: bool = True,
        num_route_queries: int = 1,
        adaptive_low_rank: bool = True,
    ):
        super().__init__(base_attn)
        self.core = ConfidenceGatedAttentionCore(
            top_k=top_k,
            low_rank_dim=low_rank_dim,
            window_size=window_size,
            gate_threshold=gate_threshold,
            peak_threshold=peak_threshold,
            linear_weight=linear_weight,
            use_triton=use_triton,
            always_global=always_global,
            num_route_queries=num_route_queries,
            adaptive_low_rank=adaptive_low_rank,
        )

    @property
    def last_stats(self):
        return self.core.last_stats

    def sparse_attention(
        self,
        Q,
        K,
        V,
        token_mask,
        bsz,
        num_heads,
        is_causal=False,
    ):
        return self.core(
            Q,
            K,
            V,
            token_mask,
            bsz,
            num_heads,
            is_causal=is_causal,
            training=self.training,
        )


def build_model(
    model_path: str = os.path.join(
        os.environ.get("SCRATCH", "/scratch/$USER"),
        "models",
        "DeepSeek-R1-Distill-Llama-8B",
    ),
    top_k: int = 512,
    low_rank_dim: int = 128,
    window_size: int = 256,
    gate_threshold: float = 0.5,
    peak_threshold: float = -1.0,
    linear_weight: float = 0.5,
    use_triton: bool = True,
    always_global: bool = True,
    num_route_queries: int = 1,
    adaptive_low_rank: bool = True,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    pooling: str = "last",
):
    """Build R1-Llama with Exp 18 attention and optional LoRA adapters."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=ConfidenceGatedAttention,
        num_labels=num_labels,
        attn_kwargs={
            "top_k": top_k,
            "low_rank_dim": low_rank_dim,
            "window_size": window_size,
            "gate_threshold": gate_threshold,
            "peak_threshold": peak_threshold,
            "linear_weight": linear_weight,
            "use_triton": use_triton,
            "always_global": always_global,
            "num_route_queries": num_route_queries,
            "adaptive_low_rank": adaptive_low_rank,
        },
        pooling=pooling,
    )
    return apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
