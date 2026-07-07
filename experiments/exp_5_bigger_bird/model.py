import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartAttention

from shared.patched_model import classification_forward
from shared.sparse_attn_utils import gather_attention_triton_or_none

# Idea A: Unified Bigger Bird
# Combines the THREE components from the original proposal in a SINGLE attention module:
#   (1) Diversity-aware LOCAL top-k from a sliding window (MMR-lite)
#   (2) Submodular-style GLOBAL selection via a learned gate with coverage penalty
#   (3) Biased random TELEPORTS (mix of high-gate tokens + uniform random)
# Total selected per query: M = k + G + T  -> softmax over M keys instead of N.


class BiggerBirdAttention(BartAttention):
    def __init__(self, base_attn: BartAttention,
                 window_size: int = 64,
                 local_k: int = 32,
                 num_globals: int = 16,
                 num_teleports: int = 8,
                 diversity_lambda: float = 0.3,
                 teleport_bias: float = 0.5,
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
        self.local_k = local_k
        self.num_globals = num_globals
        self.num_teleports = num_teleports
        self.diversity_lambda = diversity_lambda
        self.teleport_bias = teleport_bias
        self.use_triton = use_triton

        # Learned global importance gate (submodular surrogate)
        self.global_gate = nn.Linear(base_attn.embed_dim, 1)
        nn.init.zeros_(self.global_gate.bias)
        nn.init.normal_(self.global_gate.weight, std=0.02)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

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
        # Diversity penalty: subtract max similarity to the current top-1 within window
        top1 = rel.argmax(dim=-1, keepdim=True)  # [BH, Tq, 1]
        top1_k = torch.gather(K_w, 2, top1.unsqueeze(-1).expand(-1, -1, 1, D))  # [BH, Tq, 1, D]
        sim_to_top1 = torch.einsum("bqwd,bqod->bqw", K_w, top1_k.squeeze(2).unsqueeze(2))
        # Note: this is a coarse MMR-lite (one anchor). Full MMR is sequential.
        mmr_scores = rel - self.diversity_lambda * sim_to_top1
        # Take top-k in window (relative indices)
        kk = min(k, W)
        _, sel_rel = torch.topk(mmr_scores, k=kk, dim=-1)  # [BH, Tq, k]
        # Map to absolute indices
        win_idx_exp = window_idx.unsqueeze(0).expand(BH, Tq, W)
        sel_abs = torch.gather(win_idx_exp, 2, sel_rel)
        return sel_abs

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        for k in ("cache_position", "position_bias", "alibi_bias"):
            kwargs.pop(k, None)

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * (self.head_dim ** -0.5)
        if is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        BH = bsz * self.num_heads
        Q = self._shape(query_states, tgt_len, bsz).reshape(BH, tgt_len, self.head_dim)
        K = key_states.reshape(BH, -1, self.head_dim)
        V = value_states.reshape(BH, -1, self.head_dim)
        src_len = K.size(1)

        M_total = self.local_k + self.num_globals + self.num_teleports

        # ---- Fallback to dense for very short sequences ----
        if src_len <= M_total:
            scores = torch.bmm(Q, K.transpose(1, 2))
            if attention_mask is not None:
                am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                if am_bool.dim() == 2: am_bool = am_bool[:, None, None, :]
                mask_expanded = am_bool.expand(bsz, self.num_heads, tgt_len, src_len).reshape(BH, tgt_len, src_len)
                scores = scores.masked_fill(~mask_expanded, -1e9)
            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            out = torch.bmm(attn_probs, V).reshape(BH, tgt_len, self.head_dim)
        else:
            # ---- (1) LOCAL: diverse top-k within sliding window ----
            t = torch.arange(tgt_len, device=Q.device)
            ws = torch.clamp(t - self.window_size // 2, min=0, max=max(0, src_len - self.window_size))
            offs = torch.arange(self.window_size, device=Q.device)
            window_idx = ws.unsqueeze(1) + offs.unsqueeze(0)  # [Tq, W]
            local_idx = self._mmr_local_topk(Q, K, window_idx, self.local_k)  # [BH, Tq, k]

            # ---- (2) GLOBAL: learned submodular-style gate ----
            g_scores = self.global_gate(hidden_states).squeeze(-1)  # [B, T]
            if attention_mask is not None:
                gm = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                if gm.dim() == 4: gm = gm[:, 0, 0, :]
                g_scores = g_scores.masked_fill(~gm, -1e9)
            _, g_idx = torch.topk(g_scores, k=self.num_globals, dim=-1)  # [B, G]
            g_idx_exp = g_idx.unsqueeze(1).expand(bsz, tgt_len, self.num_globals)
            g_idx_exp = g_idx_exp.unsqueeze(1).expand(bsz, self.num_heads, tgt_len, self.num_globals)
            g_idx_exp = g_idx_exp.reshape(BH, tgt_len, self.num_globals)

            # ---- (3) TELEPORTS: biased random (half from gate top-2G, half uniform) ----
            n_biased = self.num_teleports // 2
            n_random = self.num_teleports - n_biased
            # Biased: top-2G candidates from gate, sample n_biased per query
            if n_biased > 0:
                top2g = max(2 * self.num_globals, self.num_teleports)
                _, cand_idx = torch.topk(g_scores, k=min(top2g, src_len), dim=-1)  # [B, C]
                # Random pick from candidates (per-query independent)
                rand_pick = torch.randint(0, cand_idx.size(-1), (bsz, tgt_len, n_biased), device=Q.device)
                biased_idx = torch.gather(cand_idx.unsqueeze(1).expand(bsz, tgt_len, cand_idx.size(-1)), 2, rand_pick)
                biased_idx = biased_idx.unsqueeze(1).expand(bsz, self.num_heads, tgt_len, n_biased).reshape(BH, tgt_len, n_biased)
            else:
                biased_idx = torch.empty(BH, tgt_len, 0, dtype=torch.long, device=Q.device)
            if n_random > 0:
                random_idx = torch.randint(0, src_len, (BH, tgt_len, n_random), device=Q.device)
            else:
                random_idx = torch.empty(BH, tgt_len, 0, dtype=torch.long, device=Q.device)
            teleport_idx = torch.cat([biased_idx, random_idx], dim=-1)

            # ---- Combine indices: M = k + G + T ----
            abs_idx = torch.cat([local_idx, g_idx_exp, teleport_idx], dim=-1)  # [BH, Tq, M]

            out = gather_attention_triton_or_none(
                Q, K, V, abs_idx, attention_mask, bsz, self.num_heads,
                self.use_triton, self.training,
            )
            if out is None:
                M = abs_idx.size(-1)

                # Gather K, V
                idx_g = abs_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
                K_sel = torch.gather(K.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_g)
                V_sel = torch.gather(V.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_g)

                # Scores
                scores_sel = torch.matmul(Q.unsqueeze(2), K_sel.transpose(-1, -2)).squeeze(2)  # [BH, Tq, M]

                # Mask gathered padding positions
                if attention_mask is not None:
                    am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                    if am_bool.dim() == 4: am_bool = am_bool[:, 0, 0, :]
                    # Expand to [BH, src_len] and gather along abs_idx
                    am_bh = am_bool.unsqueeze(1).expand(bsz, self.num_heads, src_len).reshape(BH, src_len)
                    allowed = torch.gather(am_bh.unsqueeze(1).expand(BH, tgt_len, src_len), 2, abs_idx)
                    scores_sel = scores_sel.masked_fill(~allowed, -1e9)

                attn_probs = F.softmax(scores_sel, dim=-1)
                attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
                out = torch.bmm(
                    attn_probs.reshape(BH * tgt_len, 1, M),
                    V_sel.reshape(BH * tgt_len, M, self.head_dim)
                ).reshape(BH, tgt_len, self.head_dim)

        attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim) \
                         .transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, None)


def patch_bart(model: nn.Module, **kw):
    def _recurse(m):
        for name, child in list(m.named_children()):
            if isinstance(child, BartAttention):
                if getattr(child, "is_decoder", False): continue
                setattr(m, name, BiggerBirdAttention(child, **kw))
            else:
                _recurse(child)
    _recurse(model)


class PatchedModel(nn.Module):
    def __init__(self, base_model, window_size=64, local_k=32, num_globals=16, num_teleports=8,
                 diversity_lambda=0.3, teleport_bias=0.5, use_triton=True):
        super().__init__()
        self.model = base_model
        self.use_triton = use_triton
        patch_bart(self.model, window_size=window_size, local_k=local_k,
                   num_globals=num_globals, num_teleports=num_teleports,
                   diversity_lambda=diversity_lambda, teleport_bias=teleport_bias,
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

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return classification_forward(self.model, input_ids, attention_mask, labels, **kwargs)

    @property
    def config(self):
        return self.model.config
