import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartAttention
from transformers.modeling_outputs import SequenceClassifierOutput

from shared.patched_model import classification_forward
from shared.sparse_attn_utils import gather_attention_triton_or_none

# Idea E: Attention Speculation
# Inspired by speculative decoding: run a CHEAP attention path first, and a
# FULL attention path occasionally to teach the cheap path via KL.
#   - Fast path: each query attends to its local window + a few "anchor" tokens
#     (first, last, mid). Cost: O(n * (W + A)).
#   - Verifier path: full O(n^2) attention, but applied only every `verify_every`
#     layer to provide a KL signal during training.
# At inference: only the fast path runs.


class AttnSpeculAttention(BartAttention):
    def __init__(self, base_attn: BartAttention,
                 window_size: int = 64,
                 num_anchors: int = 4,
                 verify: bool = False,
                 verify_kl_weight: float = 0.1,
                 use_triton: bool = True):
        super().__init__(
            embed_dim=base_attn.embed_dim,
            num_heads=base_attn.num_heads,
            dropout=base_attn.dropout.p if isinstance(base_attn.dropout, nn.Dropout) else float(base_attn.dropout),
            is_decoder=base_attn.is_decoder,
            bias=base_attn.k_proj.bias is not None,
        )
        self.q_proj.load_state_dict(base_attn.q_proj.state_dict())
        self.k_proj.load_state_dict(base_attn.k_proj.state_dict())
        self.v_proj.load_state_dict(base_attn.v_proj.state_dict())
        self.out_proj.load_state_dict(base_attn.out_proj.state_dict())
        self.window_size = window_size
        self.num_anchors = num_anchors
        self.verify = verify
        self.verify_kl_weight = verify_kl_weight
        self.use_triton = use_triton
        self.last_kl = None  # populated when verify=True

    def _shape(self, tensor, seq_len, bsz):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _anchor_indices(self, src_len, device):
        """Pick `num_anchors` evenly spaced anchors (first, last, and intermediate)."""
        if self.num_anchors <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if self.num_anchors >= src_len:
            return torch.arange(src_len, device=device)
        positions = torch.linspace(0, src_len - 1, steps=self.num_anchors, device=device).long()
        return positions

    def forward(self, hidden_states, key_value_states=None, past_key_value=None,
                attention_mask=None, layer_head_mask=None, output_attentions=False,
                use_cache=False, **kwargs):
        for k in ("cache_position", "position_bias", "alibi_bias"):
            kwargs.pop(k, None)

        is_cross = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        Qs = self.q_proj(hidden_states) * (self.head_dim ** -0.5)
        if is_cross:
            Ks = self._shape(self.k_proj(key_value_states), -1, bsz)
            Vs = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            Ks = self._shape(self.k_proj(hidden_states), -1, bsz)
            Vs = self._shape(self.v_proj(hidden_states), -1, bsz)
        BH = bsz * self.num_heads
        Q = self._shape(Qs, tgt_len, bsz).reshape(BH, tgt_len, self.head_dim)
        K = Ks.reshape(BH, -1, self.head_dim)
        V = Vs.reshape(BH, -1, self.head_dim)
        src_len = K.size(1)

        # --- Build sparse index set: window around each query + anchors ---
        t = torch.arange(tgt_len, device=Q.device)
        win_start = torch.clamp(t - self.window_size // 2, min=0, max=max(0, src_len - self.window_size))
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
            Q, K, V, abs_idx_bh, attention_mask, bsz, self.num_heads,
            self.use_triton, self.training,
        )
        if out_fast is None:
            idx_g = abs_idx_bh.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            K_sel = torch.gather(K.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_g)
            V_sel = torch.gather(V.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_g)
            scores_fast = torch.matmul(Q.unsqueeze(2), K_sel.transpose(-1, -2)).squeeze(2)

            if attention_mask is not None:
                am = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                if am.dim() == 4: am = am[:, 0, 0, :]
                am_bh = am.unsqueeze(1).expand(bsz, self.num_heads, src_len).reshape(BH, src_len)
                allowed = torch.gather(am_bh.unsqueeze(1).expand(BH, tgt_len, src_len), 2, abs_idx_bh)
                scores_fast = scores_fast.masked_fill(~allowed, -1e9)

            attn_fast = F.softmax(scores_fast, dim=-1)
            attn_fast = F.dropout(attn_fast, p=self.dropout, training=self.training)
            out_fast = torch.bmm(
                attn_fast.reshape(BH * tgt_len, 1, M),
                V_sel.reshape(BH * tgt_len, M, self.head_dim)
            ).reshape(BH, tgt_len, self.head_dim)

        # --- Verifier path (training-only, on verify layers) ---
        if self.verify and self.training:
            full_scores = torch.bmm(Q, K.transpose(1, 2))
            if attention_mask is not None:
                me = am.unsqueeze(1).unsqueeze(1).expand(bsz, self.num_heads, tgt_len, src_len).reshape(BH, tgt_len, src_len)
                full_scores = full_scores.masked_fill(~me, -1e9)
            full_log_probs = F.log_softmax(full_scores, dim=-1)
            # Compute the cheap path's probabilities over the full vocabulary by scattering
            fast_log_probs_sparse = F.log_softmax(scores_fast, dim=-1)
            full_probs_at_sparse = torch.gather(full_log_probs, 2, abs_idx_bh)
            # KL(fast || full) on the sparse support — encourages fast distribution to align
            kl = (fast_log_probs_sparse.exp() * (fast_log_probs_sparse - full_probs_at_sparse)).sum(dim=-1).mean()
            self.last_kl = kl * self.verify_kl_weight
        else:
            self.last_kl = None

        attn_output = out_fast.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        return (self.out_proj(attn_output), None)


def patch_bart(model: nn.Module, window_size=64, num_anchors=4, verify_every=4, verify_kl_weight=0.1, use_triton=True):
    encoder = getattr(model, "model", model)
    if hasattr(encoder, "encoder"):
        encoder = encoder.encoder
    layers = getattr(encoder, "layers", None)
    if layers is None:
        return 0
    for i, layer in enumerate(layers):
        sa = layer.self_attn
        verify = (i % verify_every == 0)
        layer.self_attn = AttnSpeculAttention(sa, window_size=window_size, num_anchors=num_anchors,
                                              verify=verify, verify_kl_weight=verify_kl_weight,
                                              use_triton=use_triton)
    return len(layers)


class PatchedModel(nn.Module):
    def __init__(self, base_model, window_size=64, num_anchors=4, verify_every=4, verify_kl_weight=0.1, use_triton=True):
        super().__init__()
        self.model = base_model
        self.use_triton = use_triton
        patch_bart(self.model, window_size=window_size, num_anchors=num_anchors,
                   verify_every=verify_every, verify_kl_weight=verify_kl_weight,
                   use_triton=use_triton)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(**kwargs)
    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable()
    @property
    def supports_gradient_checkpointing(self):
        return getattr(self.model, "supports_gradient_checkpointing", True)

    def _collect_kl(self):
        total = None
        for m in self.model.modules():
            if isinstance(m, AttnSpeculAttention) and m.last_kl is not None:
                total = m.last_kl if total is None else total + m.last_kl
        return total

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = classification_forward(self.model, input_ids, attention_mask, labels, **kwargs)
        if out.loss is not None:
            kl_loss = self._collect_kl()
            if kl_loss is not None:
                out = SequenceClassifierOutput(loss=out.loss + kl_loss, logits=out.logits)
        return out

    @property
    def config(self):
        return self.model.config
