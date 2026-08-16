"""Exp 12 — S2-HHST (Strided Shard Heterogeneous Sparse Attention) on R1-Distill-Llama-8B.

Block-sparse attention with per-head strided sharding:
  - Local blocks: each query attends to nearby blocks (local_blocks * shard_size tokens)
  - Strided blocks: each head h attends to blocks {h, h+stride, h+2*stride, ...}
  - Sink token: position 0 is always attended to
  - All layers use the strided sparse pattern; dense layers are not supported

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
from sparse_attn_utils import causal_sparse_attention, causal_sparse_attention_with_indices


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

    def _strided_block_indices_tensor(self, n_blocks: int, head: int, device) -> torch.Tensor:
        """Vectorized strided block indices for a single head."""
        # Blocks: head, head+stride, head+2*stride, ...
        max_b = n_blocks - 1
        if head > max_b:
            return torch.empty(0, dtype=torch.long, device=device)
        n_strided = (max_b - head) // self.stride_blocks + 1
        return torch.arange(head, head + n_strided * self.stride_blocks, self.stride_blocks, device=device, dtype=torch.long)

    def _build_gather_indices(self, src_len: int, device: torch.device) -> torch.Tensor:
        """[num_heads, src_len, M_max] key indices; padded with 0.

        Vectorized version: builds local + strided indices without Python loops
        over src_len.
        """
        cache_key = (src_len, str(device))
        if cache_key in self._index_cache:
            return self._index_cache[cache_key]

        s = self.shard_size
        bl = self.local_blocks
        n_blocks = (src_len + s - 1) // s
        H = self.num_heads

        # Precompute strided block indices per head
        strided_per_head = []
        max_strided = 0
        for h in range(H):
            idx = self._strided_blocks(n_blocks, h)
            strided_per_head.append(idx)
            max_strided = max(max_strided, len(idx))

        # Local window size: 2 * bl * s + 1
        local_size = 2 * bl * s + 1
        # Total M per query: local + max_strided * s + 1 (sink)
        M = local_size + max_strided * s + (1 if self.use_sink else 0)

        # Build index tensor: [H, src_len, M]
        out = torch.zeros(H, src_len, M, dtype=torch.long, device=device)

        positions = torch.arange(src_len, device=device)

        for h in range(H):
            # Local window: [src_len, local_size]
            lo = (positions - bl * s).clamp(min=0)
            hi = (positions + bl * s).clamp(max=src_len - 1)
            # Build local indices: for each q, [lo, lo+1, ..., hi]
            # Use a padded approach: max local_size entries
            local_offsets = torch.arange(-bl * s, bl * s + 1, device=device)  # [local_size]
            local_idx = (positions.unsqueeze(1) + local_offsets.unsqueeze(0)).clamp(0, src_len - 1)  # [src_len, local_size]

            # Strided block token indices (same for all queries in this head)
            strided_blocks = strided_per_head[h]
            strided_toks = []
            for b in strided_blocks:
                start = b * s
                end = min((b + 1) * s, src_len)
                strided_toks.extend(range(start, end))
            strided_toks_t = torch.tensor(strided_toks, dtype=torch.long, device=device) if strided_toks else torch.empty(0, dtype=torch.long, device=device)

            # Build full index: [src_len, M]
            full_idx = torch.zeros(src_len, M, dtype=torch.long, device=device)
            col = 0
            # Sink
            if self.use_sink and src_len > 0:
                full_idx[:, col] = 0
                col += 1
            # Local
            full_idx[:, col:col + local_size] = local_idx
            col += local_size
            # Strided (same for all queries)
            if len(strided_toks_t) > 0:
                full_idx[:, col:col + len(strided_toks_t)] = strided_toks_t.unsqueeze(0).expand(src_len, -1)
                col += len(strided_toks_t)

            out[h] = full_idx

        self._index_cache[cache_key] = out
        return out

    def _build_strided_routed_indices(self, src_len: int, device) -> torch.Tensor:
        """Build head-shared strided block indices for causal_sparse_attention.

        Returns: [BH, k] token indices (sorted), where k = n_strided_blocks * shard_size.
        Each head h gets blocks {h, h+stride, h+2*stride, ...} expanded to tokens.
        """
        s = self.shard_size
        n_blocks = (src_len + s - 1) // s
        H = self.num_heads

        # Build per-head strided token indices
        all_indices = []
        max_k = 0
        for h in range(H):
            strided_blocks = self._strided_blocks(n_blocks, h)
            toks = []
            for b in strided_blocks:
                start = b * s
                end = min((b + 1) * s, src_len)
                toks.extend(range(start, end))
            all_indices.append(toks)
            max_k = max(max_k, len(toks))

        # Pad to max_k and stack
        k = max(max_k, 1)
        out = torch.zeros(H, k, dtype=torch.long, device=device)
        for h, toks in enumerate(all_indices):
            if toks:
                out[h, :len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
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
        BH, tgt_len, _ = Q.shape
        src_len = K.size(1)

        if is_causal:
            # Use causal_sparse_attention with strided routed indices + local window
            routed_idx = self._build_strided_routed_indices(src_len, Q.device)  # [H, k]
            # Expand to BH: [BH, k]
            routed_bh = routed_idx.unsqueeze(1).expand(-1, bsz, -1).reshape(BH, -1)
            local_w = 2 * self.local_blocks * self.shard_size + 1
            return causal_sparse_attention(
                Q, K, V, routed_bh, local_window=local_w,
                token_mask=token_mask, bsz=bsz, num_heads=num_heads,
            )

        # Bidirectional mode: use pre-built per-query indices
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


def patch_llama_hybrid(model, sparse_cls, dense_layers=None, **attn_kwargs):
    """Patch every Llama layer with the requested sparse attention class."""
    if dense_layers:
        raise ValueError("S2-HHST is sparse-only; dense_layers must be empty")
    inner = getattr(model, "model", model)
    for layer in inner.layers:
        base_attn = layer.self_attn
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
