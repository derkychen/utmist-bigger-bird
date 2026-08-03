"""Shared infrastructure for patching Llama-3 (R1-Distill-Llama-8B) attention.

This module provides the base class and helpers that all experiments use when
running on R1-Distill-Llama-8B instead of BART-base. The key differences from
the BART setup:

  - **GQA**: Llama-3 has 32 query heads but only 8 KV heads (4:1 grouping).
    The base class projects Q to 32 heads and K/V to 8 heads, applies RoPE,
    then expands K/V to 32 heads via repeat_interleave before calling the
    sparse attention logic.

  - **RoPE**: Rotary position embeddings are applied to Q and K after
    projection, before sparse key selection. This is handled by the base
    class; subclasses receive already-position-encoded Q/K/V.

  - **Bidirectional**: For sequence classification we disable causal masking.
    The model was pretrained causally, but for encoding+pooling we want every
    token to attend to every other token (same as the BART encoder setup).
    The base class extracts only the padding mask from the HF attention_mask
    and ignores the causal component.

  - **LoRA**: The 8B model is too large for full fine-tuning on a 40GB MIG
    slice. LoRA adapters are applied to q_proj/k_proj/v_proj/o_proj after
    patching. Only the adapters + classification head are trained.

Usage in an experiment's model.py:

    from shared.llama_patched_model import LlamaSparseAttention, patch_llama

    class MySparseAttention(LlamaSparseAttention):
        def __init__(self, base_attn, my_param=64):
            super().__init__(base_attn)
            self.my_param = my_param

        def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads):
            # Q, K, V are [BH, T, d] — already projected, RoPE'd, GQA-expanded
            # token_mask is [B, T] bool or None
            # Return [BH, T, d]
            ...

    def patch_model(model, **kwargs):
        return patch_llama(model, MySparseAttention, **kwargs)
"""

import os

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from sparse_attn_utils import token_mask_1d


class LlamaSparseAttention(nn.Module):
    """Base class for sparse attention on Llama-3.

    Handles Q/K/V projection with GQA, RoPE application, KV-head expansion,
    and output projection. Subclasses override ``sparse_attention()`` to
    implement their specific sparse routing logic.

    The sparse_attention method receives Q, K, V as ``[BH, T, d]`` tensors
    (batch*heads merged) and a 1D token_mask ``[B, T]`` (or None). This
    matches the interface expected by the existing ``sparse_attn_utils``
    helpers (head_shared_topk_indices, sparse_attention_head_shared, etc.).
    """

    def __init__(self, base_attn: LlamaAttention):
        super().__init__()
        self.config = base_attn.config
        self.layer_idx = getattr(base_attn, "layer_idx", 0)
        self.head_dim = base_attn.head_dim
        self.num_heads = base_attn.config.num_attention_heads
        self.num_kv_heads = base_attn.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = base_attn.scaling
        self.is_causal = False  # default: bidirectional for classification

        # Reuse the same projection weights (LoRA adapters attach here)
        self.q_proj = base_attn.q_proj
        self.k_proj = base_attn.k_proj
        self.v_proj = base_attn.v_proj
        self.o_proj = base_attn.o_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        bsz, seq_len, _ = hidden_states.shape

        # --- Project Q/K/V (GQA: Q has num_heads, K/V have num_kv_heads) ---
        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, H, T, d]
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, Hkv, T, d]
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, Hkv, T, d]

        # --- Apply RoPE ---
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # --- Expand KV heads to match query heads (GQA) ---
        key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        # --- Reshape to [BH, T, d] for sparse attention utilities ---
        BH = bsz * self.num_heads
        Q = query_states.reshape(BH, seq_len, self.head_dim) * self.scaling
        K = key_states.reshape(BH, seq_len, self.head_dim)
        V = value_states.reshape(BH, seq_len, self.head_dim)

        # --- Extract padding mask (ignore causal component) ---
        token_mask = token_mask_1d(attention_mask, bsz, seq_len, Q.device)

        # --- Call subclass sparse attention ---
        out = self.sparse_attention(Q, K, V, token_mask, bsz, self.num_heads, is_causal=self.is_causal)

        # --- Reshape back and project ---
        attn_output = out.view(bsz, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        return (attn_output, None)

    def sparse_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_mask: torch.Tensor | None,
        bsz: int,
        num_heads: int,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Override with sparse attention logic.

        Args:
            Q, K, V: [BH, T, d] — already projected, RoPE'd, GQA-expanded, Q pre-scaled
            token_mask: [B, T] bool padding mask, or None
            bsz, num_heads: batch size and number of query heads
            is_causal: if True, apply causal mask (for generative evaluation)

        Returns:
            [BH, T, d] attention output
        """
        raise NotImplementedError


def patch_llama(
    model: nn.Module,
    attention_cls: type,
    **attn_kwargs,
) -> nn.Module:
    """Recursively replace every LlamaAttention with attention_cls.

    Args:
        model: a LlamaModel or LlamaForSequenceClassification
        attention_cls: subclass of LlamaSparseAttention
        **attn_kwargs: passed to attention_cls(base_attn, **attn_kwargs)

    Returns:
        the model (modified in-place)
    """
    inner = getattr(model, "model", model)
    for layer in inner.layers:
        base_attn = layer.self_attn
        layer.self_attn = attention_cls(base_attn, **attn_kwargs)
    return model


def llama_classification_forward(
    base_model: nn.Module,
    input_ids=None,
    attention_mask=None,
    labels=None,
    pooling: str = "last",
    **kwargs,
):
    """Llama encoder forward + pool + classification head.

    By default uses **last-token pooling**: the hidden state of the last
    non-pad token.  This is the natural pooling for a causal LM — the last
    token has attended to every token before it, so its representation is
    a summary of the entire context.  This is critical for retrieval tasks
    like RULER niah where the answer is a tiny needle buried in 4096 tokens:
    mean-pooling averages the needle signal to ~0.4% of the pooled vector,
    while last-token pooling preserves it.

    For tasks where the signal is distributed (e.g. LRA listops), set
    ``pooling="mean"`` to average over all tokens.
    """
    inner = base_model.model
    outputs = inner(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state

    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    elif pooling == "last":
        # Last non-pad token — the natural "summary" position for a causal LM.
        # attention_mask is [B, T] with 1 for real tokens, 0 for padding.
        # We assume left-padding (pad tokens at the start) so the last token
        # in the sequence is the last real token.
        batch_size = hidden.shape[0]
        seq_len = hidden.shape[1]
        # Find the index of the last real token for each sample
        lengths = attention_mask.sum(dim=1) - 1  # [B], 0-indexed last position
        pooled = hidden[torch.arange(batch_size, device=hidden.device), lengths]
    else:
        pooled = hidden[:, 0, :]  # first-token pooling

    # Cast to classification head dtype (head is float32, hidden may be bf16)
    head = base_model.classification_head
    pooled = pooled.to(head.weight.dtype)
    logits = head(pooled)

    loss = None
    if labels is not None:
        if labels.dtype != torch.long:
            labels = labels.long()
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, base_model.config.num_labels),
            labels.view(-1),
        )
    return SequenceClassifierOutput(loss=loss, logits=logits)


class LlamaPatchedModel(nn.Module):
    """Wrapper for Llama-3 with patched attention + classification head.

    Usage:
        model = LlamaPatchedModel.from_pretrained(
            model_path=os.path.join(
                os.environ.get("SCRATCH", "/scratch/$USER"),
                "models", "DeepSeek-R1-Distill-Llama-8B"
            ),
            attention_cls=MySparseAttention,
            num_labels=2,
            attn_kwargs={"my_param": 64},
        )
    """

    def __init__(self, base_model: nn.Module, classification_head: nn.Module, config, pooling: str = "last"):
        super().__init__()
        self.model = base_model
        self.classification_head = classification_head
        self.config = config
        self.pooling = pooling  # "last", "mean", or "first"

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        attention_cls: type,
        num_labels: int = 2,
        attn_kwargs: dict | None = None,
        torch_dtype=torch.bfloat16,
        pooling: str = "last",
        **kwargs,
    ) -> "LlamaPatchedModel":
        from transformers import AutoModel, AutoConfig

        attn_kwargs = attn_kwargs or {}
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = num_labels

        base_model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch_dtype, **kwargs
        )

        # Patch attention with sparse variant
        patch_llama(base_model, attention_cls, **attn_kwargs)

        # Classification head in float32 for stable training
        hidden_size = config.hidden_size
        classification_head = nn.Linear(hidden_size, num_labels).float()

        return cls(base_model, classification_head, config, pooling=pooling)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable()

    @property
    def supports_gradient_checkpointing(self):
        return getattr(self.model, "supports_gradient_checkpointing", True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return llama_classification_forward(
            self, input_ids, attention_mask, labels,
            pooling=kwargs.pop("pooling", self.pooling),
            **kwargs,
        )


def apply_lora(
    model: LlamaPatchedModel,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> LlamaPatchedModel:
    """Apply LoRA adapters to the patched Llama model.

    Targets q_proj, k_proj, v_proj, o_proj by default (the projections that
    our LlamaSparseAttention subclasses share with the original LlamaAttention).
    The classification head is always trained fully (not LoRA).

    Returns the model with LoRA adapters applied. Use model.print_trainable_parameters()
    to verify only adapters + head are trainable.
    """
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=None,  # we have our own classification head
        bias="none",
    )

    # PEFT wraps the model; our classification_head stays outside LoRA
    model.model = get_peft_model(model.model, lora_config)

    # Ensure classification head requires grad
    model.classification_head.requires_grad_(True)

    return model
