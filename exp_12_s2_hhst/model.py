import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartAttention
from transformers.modeling_outputs import SequenceClassifierOutput

from shared.kernels import build_gather_key_mask, should_use_triton, sparse_gather_attention


class S2HHSTAttention(BartAttention):

    def __init__(
        self,
        base_attn: BartAttention,
        shard_size: int = 32,
        local_blocks: int = 2,
        stride_blocks: int | None = None,
        use_sink: bool = True,
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

        self.shard_size = shard_size
        self.local_blocks = local_blocks
        # Vertical stride in blocks; default = num_heads for heterogeneous complete union
        self.stride_blocks = stride_blocks if stride_blocks is not None else self.num_heads
        self.use_sink = use_sink
        self.use_triton = use_triton
        self._index_cache: dict[tuple[int, str], torch.Tensor] = {}

    def _use_triton_kernels(self, q: torch.Tensor) -> bool:
        return should_use_triton(self.use_triton, q, training=self.training)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _strided_blocks(self, n_blocks: int, head: int) -> list[int]:
        """Key blocks visible to `head` via the HHST strided shard (block-level mask)."""
        blocks = []
        for b in range(n_blocks):
            if b >= head and (b - head) % self.stride_blocks == 0:
                blocks.append(b)
        return blocks

    def _build_gather_indices(self, src_len: int, device: torch.device) -> torch.Tensor:
        """[num_heads, src_len, M_max] key indices; padded with 0 and masked in forward."""
        cache_key = (src_len, str(device))
        if cache_key in self._index_cache:
            return self._index_cache[cache_key]

        s = self.shard_size
        bl = self.local_blocks
        n_blocks = (src_len + s - 1) // s
        per_head_q: list[list[list[int]]] = []

        for h in range(self.num_heads):
            strided_blocks = set(self._strided_blocks(n_blocks, h))
            rows: list[list[int]] = []
            for q in range(src_len):
                q_block = q // s
                toks: set[int] = set()
                if self.use_sink and src_len > 0:
                    toks.add(0)
                lo = max(0, q - bl * s)
                hi = min(src_len, q + bl * s + 1)
                toks.update(range(lo, hi))
                for b in strided_blocks:
                    if abs(b - q_block) >= bl:
                        start, end = b * s, min((b + 1) * s, src_len)
                        toks.update(range(start, end))
                rows.append(sorted(toks))
            per_head_q.append(rows)

        m_max = max(len(idx) for h_rows in per_head_q for idx in h_rows)
        m_max = max(m_max, 1)
        out = torch.zeros(self.num_heads, src_len, m_max, dtype=torch.long, device=device)
        for h, rows in enumerate(per_head_q):
            for q, idx in enumerate(rows):
                out[h, q, : len(idx)] = torch.tensor(idx, dtype=torch.long, device=device)
        self._index_cache[cache_key] = out
        return out

    def _sparse_budget(self, src_len: int) -> int:
        local = 2 * self.local_blocks * self.shard_size + (1 if self.use_sink else 0)
        n_blocks = (src_len + self.shard_size - 1) // self.shard_size
        strided_per_head = sum(
            min(self.shard_size, src_len - b * self.shard_size)
            for b in self._strided_blocks(n_blocks, 0)
        )
        return local + strided_per_head

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

        if src_len <= self._sparse_budget(src_len):
            scores = torch.bmm(q, k.transpose(1, 2))
            if attention_mask is not None:
                am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                if am_bool.dim() == 2:
                    am_bool = am_bool[:, None, None, :]
                mask_expanded = am_bool.expand(bsz, self.num_heads, tgt_len, src_len).reshape(bh, tgt_len, src_len)
                scores = scores.masked_fill(~mask_expanded, -1e9)
            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            out = torch.bmm(attn_probs, v)
        else:
            idx_hqt = self._build_gather_indices(src_len, q.device)  # [H, Tq, M]
            m = idx_hqt.size(-1)
            abs_idx = idx_hqt.unsqueeze(1).expand(self.num_heads, bsz, tgt_len, m)
            abs_idx = abs_idx.reshape(bh, tgt_len, m)

            out = None
            if self._use_triton_kernels(q):
                try:
                    key_mask = build_gather_key_mask(attention_mask, bsz, self.num_heads, tgt_len, abs_idx)
                    out = sparse_gather_attention(q, k, v, abs_idx, key_mask, scale=1.0)
                except Exception:
                    out = None

            if out is None:
                idx_gather = abs_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
                k_sel = torch.gather(k.unsqueeze(1).expand(bh, tgt_len, src_len, self.head_dim), 2, idx_gather)
                v_sel = torch.gather(v.unsqueeze(1).expand(bh, tgt_len, src_len, self.head_dim), 2, idx_gather)

                scores_sel = torch.matmul(q.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2)

                if attention_mask is not None:
                    am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                    if am_bool.dim() == 2:
                        am_bool = am_bool[:, None, None, :]
                    am_small = am_bool.expand(bsz, 1, tgt_len, src_len)
                    abs_idx_hb = abs_idx.view(self.num_heads, bsz, tgt_len, m)
                    allowed = []
                    for h in range(self.num_heads):
                        allowed.append(torch.gather(am_small, -1, abs_idx_hb[h].unsqueeze(1)).squeeze(1))
                    allowed = torch.cat(allowed, dim=0)
                    scores_sel = scores_sel.masked_fill(~allowed, -1e9)

                attn_probs = F.softmax(scores_sel, dim=-1)
                attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
                out = torch.bmm(
                    attn_probs.reshape(bh * tgt_len, 1, m),
                    v_sel.reshape(bh * tgt_len, m, self.head_dim),
                ).reshape(bh, tgt_len, self.head_dim)

        attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(
            bsz, tgt_len, self.embed_dim
        )
        attn_output = self.out_proj(attn_output)
        return (attn_output, None)


def patch_bart(
    model: nn.Module,
    shard_size: int = 32,
    local_blocks: int = 2,
    stride_blocks: int | None = None,
    use_sink: bool = True,
    dense_layers: set[int] | list[int] | None = None,
    use_triton: bool = True,
):
    """Replace encoder self-attention with HHST; keep listed layer indices dense (hybrid)."""
    if dense_layers is None:
        dense_layers = {0}
    elif isinstance(dense_layers, list):
        dense_layers = set(dense_layers)

    layer_idx = 0

    def _recurse(module: nn.Module):
        nonlocal layer_idx
        for name, child in list(module.named_children()):
            if isinstance(child, BartAttention):
                if getattr(child, "is_decoder", False):
                    continue
                if layer_idx not in dense_layers:
                    setattr(
                        module,
                        name,
                        S2HHSTAttention(
                            child,
                            shard_size=shard_size,
                            local_blocks=local_blocks,
                            stride_blocks=stride_blocks,
                            use_sink=use_sink,
                            use_triton=use_triton,
                        ),
                    )
                layer_idx += 1
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
        shard_size: int = 32,
        local_blocks: int = 2,
        stride_blocks: int | None = None,
        use_sink: bool = True,
        dense_layers: set[int] | list[int] | None = None,
        use_triton: bool = True,
    ):
        super().__init__()
        self.model = base_model
        self.use_triton = use_triton
        if isinstance(dense_layers, list):
            dense_layers = set(dense_layers)
        patch_bart(
            self.model,
            shard_size=shard_size,
            local_blocks=local_blocks,
            stride_blocks=stride_blocks,
            use_sink=use_sink,
            dense_layers=dense_layers,
            use_triton=use_triton,
        )
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
