import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartAttention
from transformers.modeling_outputs import SequenceClassifierOutput

from shared.kernels import (
    build_gather_key_mask,
    build_key_mask,
    should_use_triton,
    sliding_window_attention,
    sparse_gather_attention,
)

try:
    from .kernels import build_block_ok, compressed_causal_attention
except ImportError:
    from kernels import build_block_ok, compressed_causal_attention


class NSAAttention(BartAttention):
    def __init__(
        self,
        base_attn: BartAttention,
        block_size: int = 32,
        stride: int = 32,
        topk_blocks: int = 4,
        window_size: int = 128,
        use_triton: bool = True,
    ):
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

        self.block_size = block_size
        self.stride = stride
        self.topk_blocks = topk_blocks
        self.window_size = window_size
        self.use_triton = use_triton

        # Learnable block compression φ (maps a block of keys/values → one vector)
        flat = block_size * self.head_dim
        self.compress_k = nn.Linear(flat, self.head_dim, bias=False)
        self.compress_v = nn.Linear(flat, self.head_dim, bias=False)

        # Separate sliding-window K/V projections (paper §3.3.3)
        has_bias = base_attn.k_proj.bias is not None
        self.k_win_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=has_bias)
        self.v_win_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=has_bias)
        self.k_win_proj.load_state_dict(base_attn.k_proj.state_dict())
        self.v_win_proj.load_state_dict(base_attn.v_proj.state_dict())

        # Per-token branch gates from hidden states
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 4),
            nn.GELU(),
            nn.Linear(self.embed_dim // 4, 3),
        )
        # Favor local window at init (reference impl gate_init ≈ (2, -2, -2))
        with torch.no_grad():
            self.gate_mlp[-1].bias.copy_(torch.tensor([2.0, -2.0, -2.0]))

    def _use_triton_kernels(self, q: torch.Tensor) -> bool:
        return should_use_triton(self.use_triton, q, training=self.training)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _compress_blocks(self, tensor: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        """tensor: [BH, L, D] → compressed [BH, n_blocks, D] via learnable φ."""
        bh, length, dim = tensor.shape
        blk = self.block_size
        pad = (blk - length % blk) % blk
        if pad:
            tensor = F.pad(tensor, (0, 0, 0, pad))
        n_blocks = tensor.size(1) // blk
        blocks = tensor.view(bh, n_blocks, blk, dim).reshape(bh, n_blocks, blk * dim)
        return proj(blocks)

    def _block_means(self, k: torch.Tensor) -> torch.Tensor:
        """Mean-pool keys into block representatives for selection routing."""
        bh, length, dim = k.shape
        blk = self.block_size
        pad = (blk - length % blk) % blk
        if pad:
            k = F.pad(k, (0, 0, 0, pad))
            length = k.size(1)
        n_blocks = length // blk
        return k.view(bh, n_blocks, blk, dim).mean(dim=2)

    def _apply_padding_mask(
        self,
        scores: torch.Tensor,
        attention_mask,
        bsz: int,
        tgt_len: int,
        key_len: int,
        key_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return scores
        am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
        if am_bool.dim() == 2:
            am_bool = am_bool[:, None, None, :]
        if key_indices is None:
            mask_expanded = am_bool.expand(bsz, self.num_heads, tgt_len, key_len).reshape(scores.size(0), tgt_len, key_len)
            return scores.masked_fill(~mask_expanded, torch.finfo(scores.dtype).min)
        # Gathered keys: key_indices [BH, tgt_len, M]
        am_small = am_bool.expand(bsz, 1, tgt_len, key_len)
        bh = scores.size(0)
        abs_idx_hb = key_indices.view(self.num_heads, bsz, tgt_len, -1)
        allowed = []
        for h in range(self.num_heads):
            allowed.append(torch.gather(am_small, -1, abs_idx_hb[h].unsqueeze(1)).squeeze(1))
        allowed = torch.cat(allowed, dim=0)
        return scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

    def _window_branch(self, q, hidden_states, bsz, tgt_len, attention_mask):
        k_win = self._shape(self.k_win_proj(hidden_states), -1, bsz).reshape(-1, tgt_len, self.head_dim)
        v_win = self._shape(self.v_win_proj(hidden_states), -1, bsz).reshape(-1, tgt_len, self.head_dim)
        bh = q.size(0)
        src_len = k_win.size(1)
        w = min(self.window_size, src_len)

        if self._use_triton_kernels(q):
            try:
                key_mask = build_key_mask(attention_mask, bsz, self.num_heads, src_len, q.device)
                return sliding_window_attention(q, k_win, v_win, w, key_mask, scale=1.0)
            except Exception:
                pass

        t = torch.arange(tgt_len, device=q.device)
        starts = torch.clamp(t - w + 1, min=0)
        offsets = torch.arange(w, device=q.device)
        local_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)  # [Tq, W]
        local_idx = local_idx.clamp(max=src_len - 1)
        local_idx_exp = local_idx.unsqueeze(0).expand(bh, -1, -1)

        idx_gather = local_idx_exp.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        k_sel = torch.gather(k_win.unsqueeze(1).expand(bh, tgt_len, src_len, self.head_dim), 2, idx_gather)
        v_sel = torch.gather(v_win.unsqueeze(1).expand(bh, tgt_len, src_len, self.head_dim), 2, idx_gather)

        scores = torch.matmul(q.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2)
        scores = self._apply_padding_mask(scores, attention_mask, bsz, tgt_len, src_len, local_idx_exp)
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        return torch.bmm(attn.reshape(bh * tgt_len, 1, w), v_sel.reshape(bh * tgt_len, w, self.head_dim)).reshape(
            bh, tgt_len, self.head_dim
        )

    def _compressed_branch(self, q, k, v, bsz, tgt_len, attention_mask):
        bh = q.size(0)
        k_cmp = self._compress_blocks(k, self.compress_k)
        v_cmp = self._compress_blocks(v, self.compress_v)
        n_cmp = k_cmp.size(1)

        if self._use_triton_kernels(q):
            try:
                block_ok = build_block_ok(
                    attention_mask, bsz, self.num_heads, n_cmp, self.stride, q.device
                )
                return compressed_causal_attention(
                    q, k_cmp, v_cmp, self.block_size, self.stride, block_ok, scale=1.0
                )
            except Exception:
                pass

        scores = torch.bmm(q, k_cmp.transpose(1, 2))
        # Causal: query t may attend to compressed block c only if block end ≤ t
        block_ends = torch.arange(n_cmp, device=q.device) * self.stride + self.block_size
        causal = block_ends.unsqueeze(0) <= torch.arange(tgt_len, device=q.device).unsqueeze(1)
        scores = scores.masked_fill(~causal.unsqueeze(0), torch.finfo(scores.dtype).min)

        if attention_mask is not None:
            am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
            if am_bool.dim() == 2:
                am_bool = am_bool[:, None, None, :]
            # Block is valid if block-start token is unmasked; broadcast to [BH, T, n_cmp]
            block_starts = torch.arange(n_cmp, device=q.device) * self.stride
            block_starts = block_starts.clamp(max=am_bool.size(-1) - 1)
            block_ok = am_bool[:, 0, 0, block_starts]  # [B, n_cmp]
            block_ok = (
                block_ok.unsqueeze(1)
                .expand(bsz, self.num_heads, n_cmp)
                .reshape(bh, 1, n_cmp)
                .expand(-1, tgt_len, -1)
            )
            scores = scores.masked_fill(~block_ok, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        return torch.bmm(attn, v_cmp)

    def _selected_branch(self, q, k, v, bsz, tgt_len, attention_mask):
        bh, src_len, dim = k.shape
        blk = self.block_size
        n_blocks = math.ceil(src_len / blk)
        k_blocks = self._block_means(k)

        block_scores = torch.bmm(q, k_blocks.transpose(1, 2))
        tok_block = torch.div(torch.arange(tgt_len, device=q.device), blk, rounding_mode="floor")
        blk_id = torch.arange(n_blocks, device=q.device)
        causal = tok_block.view(1, -1, 1) >= blk_id.view(1, 1, -1)
        diag = tok_block.view(1, -1, 1) != blk_id.view(1, 1, -1)
        block_scores = block_scores.masked_fill(~(causal & diag), torch.finfo(block_scores.dtype).min)

        m = min(self.topk_blocks, n_blocks)
        _, top_blocks = torch.topk(block_scores, k=m, dim=-1)

        if self._use_triton_kernels(q):
            try:
                block_offset = torch.arange(blk, device=q.device).view(1, 1, 1, blk)
                base_idx = (top_blocks.unsqueeze(2) * blk).unsqueeze(-1)
                token_idx = (base_idx + block_offset).reshape(bh, tgt_len, m * blk).clamp(max=src_len - 1)
                key_mask = build_gather_key_mask(
                    attention_mask, bsz, self.num_heads, tgt_len, token_idx
                )
                return sparse_gather_attention(q, k, v, token_idx, key_mask, scale=1.0)
            except Exception:
                pass

        top_blocks_exp = top_blocks.unsqueeze(2).expand(-1, -1, blk, -1).reshape(bh, tgt_len, m * blk)
        block_offset = torch.arange(blk, device=q.device).view(1, 1, 1, blk)
        base_idx = (top_blocks.unsqueeze(2) * blk).unsqueeze(-1)
        token_idx = (base_idx + block_offset).reshape(bh, tgt_len, m * blk).clamp(max=src_len - 1)

        m_tokens = m * blk
        idx_gather = token_idx.unsqueeze(-1).expand(-1, -1, -1, dim)
        k_sel = torch.gather(k.unsqueeze(1).expand(bh, tgt_len, src_len, dim), 2, idx_gather)
        v_sel = torch.gather(v.unsqueeze(1).expand(bh, tgt_len, src_len, dim), 2, idx_gather)

        scores_sel = torch.matmul(q.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2)
        scores_sel = self._apply_padding_mask(scores_sel, attention_mask, bsz, tgt_len, src_len, token_idx)

        attn = F.softmax(scores_sel, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        return torch.bmm(
            attn.reshape(bh * tgt_len, 1, m_tokens),
            v_sel.reshape(bh * tgt_len, m_tokens, dim),
        ).reshape(bh, tgt_len, dim)

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
        _ = kwargs.pop("cache_position", None)
        _ = kwargs.pop("position_bias", None)
        _ = kwargs.pop("alibi_bias", None)

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * (self.head_dim ** -0.5)
        if is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        bh = bsz * self.num_heads
        q = self._shape(query_states, tgt_len, bsz).reshape(bh, tgt_len, self.head_dim)
        k = key_states.reshape(bh, -1, self.head_dim)
        v = value_states.reshape(bh, -1, self.head_dim)
        src_len = k.size(1)

        sparse_budget = self.window_size + self.topk_blocks * self.block_size
        if src_len <= sparse_budget:
            scores = torch.bmm(q, k.transpose(1, 2))
            scores = self._apply_padding_mask(scores, attention_mask, bsz, tgt_len, src_len)
            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            out = torch.bmm(attn_probs, v)
        else:
            out_win = self._window_branch(q, hidden_states, bsz, tgt_len, attention_mask)
            out_cmp = self._compressed_branch(q, k, v, bsz, tgt_len, attention_mask)
            out_slc = self._selected_branch(q, k, v, bsz, tgt_len, attention_mask)

            gates = F.softmax(self.gate_mlp(hidden_states), dim=-1)  # [B, T, 3]
            g = gates.unsqueeze(1).expand(bsz, self.num_heads, tgt_len, 3).reshape(bh, tgt_len, 3)
            out = (
                g[..., 0:1] * out_win
                + g[..., 1:2] * out_cmp
                + g[..., 2:3] * out_slc
            )

        attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(
            bsz, tgt_len, self.embed_dim
        )
        attn_output = self.out_proj(attn_output)
        return (attn_output, None)


def patch_bart(
    model: nn.Module,
    block_size: int = 32,
    stride: int = 32,
    topk_blocks: int = 4,
    window_size: int = 128,
    use_triton: bool = True,
):
    def _recurse(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, BartAttention):
                if getattr(child, "is_decoder", False):
                    continue
                setattr(
                    module,
                    name,
                    NSAAttention(
                        child, block_size, stride, topk_blocks, window_size, use_triton=use_triton
                    ),
                )
            else:
                _recurse(child)

    _recurse(model)


class AttnPool(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.proj(x))
        s = self.score(h).squeeze(-1)
        s = s.masked_fill(~mask.bool(), torch.finfo(s.dtype).min)
        a = torch.softmax(s, dim=-1)
        return torch.bmm(a.unsqueeze(1), x).squeeze(1)


class PatchedModel(nn.Module):
    def __init__(
        self,
        base_model,
        block_size: int = 32,
        stride: int = 32,
        topk_blocks: int = 4,
        window_size: int = 128,
        use_triton: bool = True,
    ):
        super().__init__()
        self.model = base_model
        self.use_triton = use_triton
        patch_bart(self.model, block_size, stride, topk_blocks, window_size, use_triton=use_triton)
        hidden_size = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model")
        self.attn_pool = AttnPool(hidden_size)

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
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state
        if attention_mask is None:
            if input_ids is not None and self.model.config.pad_token_id is not None:
                attention_mask = (input_ids != self.model.config.pad_token_id).long()
            else:
                attention_mask = torch.ones(last_hidden.size()[:2], device=last_hidden.device, dtype=torch.long)

        pooled = self.attn_pool(last_hidden, attention_mask)
        logits = self.model.classification_head(pooled)

        loss = None
        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

    @property
    def config(self):
        return self.model.config
