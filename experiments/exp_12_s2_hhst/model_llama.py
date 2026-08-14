"""Exp 12 — S2-HHST (Strided Shard Heterogeneous Sparse Attention) on R1-Distill-Llama-8B.

Block-sparse attention with per-head strided sharding:
  - Local blocks: each query attends to nearby blocks (local_blocks * shard_size tokens)
  - Strided blocks: each head h attends to blocks {h, h+stride, h+2*stride, ...}
  - Sink token: position 0 is always attended to
  - Dense layers: specified layer indices keep full dense attention (hybrid)

The strided pattern ensures heterogeneous coverage across heads — different
heads see different blocks, so the union of all heads covers the full sequence.

Bidirectional adaptation: no causal mask (classification mode).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from patches.llama.llama_patched_model import (
    LlamaSparseAttention,
    LlamaPatchedModel,
    apply_lora,
)
from sparse_attn_utils import dense_self_attention


class S2HHSTAttention(LlamaSparseAttention):
    def __init__(
        self,
        base_attn,
        shard_size: int = 32,
        local_blocks: int = 2,
        stride_blocks: int | None = None,
        use_sink: bool = True,
        use_triton: bool = False,
    ):
        super().__init__(base_attn)
        self.shard_size = shard_size
        self.local_blocks = local_blocks
        self.stride_blocks = stride_blocks if stride_blocks is not None else self.num_heads
        self.use_sink = use_sink
        self.use_triton = use_triton
        self._index_cache: dict[tuple[int, str], torch.Tensor] = {}

    def _strided_blocks(self, n_blocks: int, head: int) -> list[int]:
        blocks = []
        for b in range(n_blocks):
            if b >= head and (b - head) % self.stride_blocks == 0:
                blocks.append(b)
        return blocks

    def _build_gather_indices(self, src_len: int, device: torch.device) -> torch.Tensor:
        """[num_heads, src_len, M_max] key indices; padded with 0."""
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

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        if is_causal:
            return dense_self_attention(Q, K, V, token_mask, bsz, num_heads, 0.0, self.training, is_causal=True)
        BH, tgt_len, _ = Q.shape
        src_len = K.size(1)

        if src_len <= self._sparse_budget(src_len):
            return dense_self_attention(Q, K, V, None, bsz, num_heads, 0.0, self.training)

        idx_hqt = self._build_gather_indices(src_len, Q.device)  # [H, Tq, M]
        m = idx_hqt.size(-1)
        abs_idx = idx_hqt.unsqueeze(1).expand(num_heads, bsz, tgt_len, m)
        abs_idx = abs_idx.reshape(BH, tgt_len, m)

        idx_gather = abs_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        k_sel = torch.gather(
            K.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_gather
        )
        v_sel = torch.gather(
            V.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_gather
        )

        scores_sel = torch.matmul(Q.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2)

        if token_mask is not None:
            am = token_mask.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(BH, src_len)
            allowed = torch.gather(
                am.unsqueeze(1).expand(BH, tgt_len, src_len), 2, abs_idx
            )
            scores_sel = scores_sel.masked_fill(~allowed, torch.finfo(scores_sel.dtype).min)

        attn_probs = F.softmax(scores_sel, dim=-1)
        return torch.bmm(
            attn_probs.reshape(BH * tgt_len, 1, m),
            v_sel.reshape(BH * tgt_len, m, self.head_dim),
        ).reshape(BH, tgt_len, self.head_dim)


class DenseLlamaAttention(LlamaSparseAttention):
    """Full dense attention for dense_layers in the hybrid scheme."""

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        if is_causal:
            return dense_self_attention(Q, K, V, token_mask, bsz, num_heads, 0.0, self.training, is_causal=True)
        return dense_self_attention(Q, K, V, None, bsz, num_heads, 0.0, self.training)


def patch_llama_hybrid(model, sparse_cls, dense_layers, **attn_kwargs):
    """Patch Llama with sparse attention, keeping specified layers dense."""
    from patches.llama.llama_patched_model import patch_llama
    inner = getattr(model, "model", model)
    if isinstance(dense_layers, list):
        dense_layers = set(dense_layers)
    for i, layer in enumerate(inner.layers):
        base_attn = layer.self_attn
        if i in dense_layers:
            layer.self_attn = DenseLlamaAttention(base_attn)
        else:
            layer.self_attn = sparse_cls(base_attn, **attn_kwargs)
    return model


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    shard_size: int = 32,
    local_blocks: int = 2,
    stride_blocks: int = 16,
    use_sink: bool = True,
    dense_layers: list[int] | None = None,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    if dense_layers is None:
        dense_layers = [0]

    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = num_labels
    base_model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    patch_llama_hybrid(
        base_model,
        S2HHSTAttention,
        dense_layers,
        shard_size=shard_size,
        local_blocks=local_blocks,
        stride_blocks=stride_blocks,
        use_sink=use_sink,
        use_triton=False,
    )

    hidden_size = config.hidden_size
    classification_head = nn.Linear(hidden_size, num_labels).float()

    from patches.llama.llama_patched_model import LlamaPatchedModel
    model = LlamaPatchedModel(base_model, classification_head, config)
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
