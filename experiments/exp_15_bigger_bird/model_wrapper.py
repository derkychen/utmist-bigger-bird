"""Wrapper to use proper BiggerBird (BigBird-based) with the shared runner."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BigBirdForSequenceClassification

from experiments.exp_15_bigger_bird.config import BiggerBirdConfig
from experiments.exp_15_bigger_bird.model import BiggerBirdAttention


def extend_bigbird_embeddings(model, context_len: int):
    """
    BigBird-RoBERTa was pretrained with max_position_embeddings around 4096.
    If we pass 8000 or larger token inputs, we must resize:
      1. position_embeddings
      2. position_ids buffer
      3. token_type_ids buffer

    Otherwise, we get:
    RuntimeError: expanded size X must match existing size 4096
    """
    emb = model.bert.embeddings

    old_pos_emb = emb.position_embeddings
    old_max, hidden_size = old_pos_emb.weight.shape

    padding_idx = old_pos_emb.padding_idx
    if padding_idx is None:
        padding_idx = 0

    # RoBERTa-style position ids can reach padding_idx + context_len.
    # Embedding size must therefore be at least context_len + padding_idx + 1.
    new_max = context_len + padding_idx + 1

    print(f"[pos-emb] old_max={old_max}, context_len={context_len}, new_max={new_max}")

    if new_max > old_max:
        new_pos_emb = nn.Embedding(
            new_max,
            hidden_size,
            padding_idx=padding_idx,
        ).to(
            device=old_pos_emb.weight.device,
            dtype=old_pos_emb.weight.dtype,
        )

        with torch.no_grad():
            # Copy pretrained positions.
            new_pos_emb.weight[:old_max] = old_pos_emb.weight

            # Initialize new positions by repeating the final pretrained position.
            # This is simple and avoids random initialization spikes.
            new_pos_emb.weight[old_max:] = old_pos_emb.weight[-1].unsqueeze(0)

        emb.position_embeddings = new_pos_emb

    # These buffers caused the expansion error
    emb.register_buffer(
        "position_ids",
        torch.arange(new_max, device=old_pos_emb.weight.device).expand((1, -1)),
        persistent=False,
    )

    emb.register_buffer(
        "token_type_ids",
        torch.zeros(
            (1, new_max),
            dtype=torch.long,
            device=old_pos_emb.weight.device,
        ),
        persistent=False,
    )

    model.config.max_position_embeddings = new_max
    model.bert.config.max_position_embeddings = new_max

    print("[pos-emb] position_embeddings:", emb.position_embeddings.weight.shape)
    print("[pos-emb] position_ids:", emb.position_ids.shape)
    print("[pos-emb] token_type_ids:", emb.token_type_ids.shape)

    return model


class PatchedModel(nn.Module):
    """
    Wraps a BigBird-RoBERTa model with BiggerBirdAttention in every encoder layer,
    exposing the interface expected by shared/runner.py.
    """

    def __init__(
        self,
        base_model,
        context_len: int = 4096,
        fragment_size: int = 128,
        max_k: int = 64,
        globals_per_head: int = 6,
        teleports_per_head: int = 4,
        teleport_bias_frac: float = 0.75,
        use_topk_mmr: bool = True,
        use_dynamic_globals: bool = True,
        use_random_attn: bool = True,
        use_teleports: bool = False,
        min_k: int = 56,
        r_target_softmax: float = 0.02,
        top_u: int = 32,
        proto_count: int = 48,
        mmr_prefilter_mult: int = 3,
        mmr_diversity_steps: int = 2,
        gamma_diversity: float = 0.16,
        alpha_pos_prior: float = 0.12,
        share_stride_layers: int = 2,
        dense_fallback_under: int = 512,
        random_selection: bool = False,
        debug_collect: bool = False,
        log_once_pairs: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Build the BiggerBird config
        bb_config = BiggerBirdConfig(
            fragment_size=fragment_size,
            r_target_softmax=r_target_softmax,
            min_k=min_k,
            max_k=max_k,
            globals_per_head=globals_per_head,
            teleports_per_head=teleports_per_head,
            teleport_bias_frac=teleport_bias_frac,
            top_u=top_u,
            proto_count=proto_count,
            mmr_prefilter_mult=mmr_prefilter_mult,
            mmr_diversity_steps=mmr_diversity_steps,
            gamma_diversity=gamma_diversity,
            alpha_pos_prior=alpha_pos_prior,
            share_stride_layers=share_stride_layers,
            dense_fallback_under=dense_fallback_under,
            random_selection=random_selection,
            debug_collect=debug_collect,
            log_once_pairs=log_once_pairs,
            use_topk_mmr=use_topk_mmr,
            use_dynamic_globals=use_dynamic_globals,
            use_random_attn=use_random_attn,
            use_teleports=use_teleports,
            # BigBird architecture params
            block_size=fragment_size,
            num_random_blocks=2,
            hidden_size=base_model.config.hidden_size,
            num_attention_heads=base_model.config.num_attention_heads,
            num_hidden_layers=base_model.config.num_hidden_layers,
            intermediate_size=base_model.config.intermediate_size,
            max_position_embeddings=max(context_len, base_model.config.max_position_embeddings),
        )

        # Extend embeddings if context_len exceeds pretrained max
        if context_len > base_model.config.max_position_embeddings:
            base_model = extend_bigbird_embeddings(base_model, context_len)

        # Sync the base model's block_size / num_random_blocks with the BiggerBird config.
        # The pretrained bigbird-roberta-base uses block_size=64, but BiggerBirdAttention
        # uses fragment_size (128) as its block_size. The BigBird encoder forward prepares
        # masks using self.block_size (set from config during __init__), so we must update
        # both the config AND the runtime attribute, otherwise mask shapes won't match.
        base_model.config.block_size = fragment_size
        base_model.config.num_random_blocks = 2
        if hasattr(base_model, 'bert'):
            base_model.bert.config.block_size = fragment_size
            base_model.bert.config.num_random_blocks = 2
            base_model.bert.block_size = fragment_size
            base_model.bert.num_random_blocks = 2

        # Replace attention in every encoder layer
        for layer in base_model.bert.encoder.layer:
            old_attn = layer.attention.self
            new_attn = BiggerBirdAttention(bb_config)
            new_attn.load_state_dict(old_attn.state_dict(), strict=False)
            layer.attention.self = new_attn

        self.model = base_model
        self.config = base_model.config
        self.bb_config = bb_config
        self.context_len = context_len

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
