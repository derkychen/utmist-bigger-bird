"""Exp 11 — NSA (Native Sparse Attention) on R1-Distill-Llama-8B.

Three-branch sparse attention (arXiv:2502.11089):
  (1) Compressed: learnable block compression → attend to block summaries
  (2) Selected: top-k block selection via block-mean routing → attend to tokens in selected blocks
  (3) Sliding window: local window attention with separate K/V projections

Branch outputs are combined via a per-token learned gate (3-way softmax).

Bidirectional adaptation: causal constraints in compressed/selected branches
are removed for sequence classification (every token attends to all blocks).
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from patches.llama.llama_patched_model import (
    LlamaSparseAttention,
    LlamaPatchedModel,
    apply_lora,
)
from sparse_attn_utils import (
    dense_self_attention,
    causal_sparse_attention,
    last_query_block_topk_indices,
    causal_sparse_attention_with_indices,
)


class NSAAttention(LlamaSparseAttention):
    def __init__(
        self,
        base_attn,
        block_size: int = 32,
        stride: int = 32,
        topk_blocks: int = 4,
        window_size: int = 128,
        use_triton: bool = False,
    ):
        super().__init__(base_attn)
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
        hidden = self.config.hidden_size
        self.k_win_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.v_win_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        # Initialize from base k/v proj
        self.k_win_proj.weight.data.copy_(base_attn.k_proj.weight.data)
        self.v_win_proj.weight.data.copy_(base_attn.v_proj.weight.data)

        # Per-token branch gates from hidden states
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, 3),
        )
        with torch.no_grad():
            self.gate_mlp[-1].bias.copy_(torch.tensor([2.0, -2.0, -2.0]))

    def forward(self, hidden_states, **kwargs):
        self._hidden_states = hidden_states
        return super().forward(hidden_states, **kwargs)

    def _compress_blocks(self, tensor, proj):
        """[BH, L, D] → compressed [BH, n_blocks, D] via learnable φ."""
        bh, length, dim = tensor.shape
        blk = self.block_size
        pad = (blk - length % blk) % blk
        if pad:
            tensor = F.pad(tensor, (0, 0, 0, pad))
        n_blocks = tensor.size(1) // blk
        blocks = tensor.view(bh, n_blocks, blk, dim).reshape(bh, n_blocks, blk * dim)
        return proj(blocks)

    def _block_means(self, k):
        """Mean-pool keys into block representatives [BH, n_blocks, D]."""
        bh, length, dim = k.shape
        blk = self.block_size
        pad = (blk - length % blk) % blk
        if pad:
            k = F.pad(k, (0, 0, 0, pad))
        n_blocks = k.size(1) // blk
        return k.view(bh, n_blocks, blk, dim).mean(dim=2)

    def _window_branch(self, Q, bsz, tgt_len, token_mask):
        """Sliding-window attention with separate K/V projections."""
        hidden = self._hidden_states
        bh = Q.size(0)
        k_win = (
            self.k_win_proj(hidden)
            .view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(bh, tgt_len, self.head_dim)
        )
        v_win = (
            self.v_win_proj(hidden)
            .view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(bh, tgt_len, self.head_dim)
        )
        src_len = k_win.size(1)
        w = min(self.window_size, src_len)

        t = torch.arange(tgt_len, device=Q.device)
        starts = torch.clamp(t - w + 1, min=0)
        offsets = torch.arange(w, device=Q.device)
        local_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)  # [Tq, W]
        local_idx = local_idx.clamp(max=src_len - 1)
        local_idx_exp = local_idx.unsqueeze(0).expand(bh, -1, -1)  # [BH, Tq, W]

        idx_gather = local_idx_exp.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        k_sel = torch.gather(
            k_win.unsqueeze(1).expand(bh, tgt_len, src_len, self.head_dim), 2, idx_gather
        )
        v_sel = torch.gather(
            v_win.unsqueeze(1).expand(bh, tgt_len, src_len, self.head_dim), 2, idx_gather
        )

        scores = torch.matmul(Q.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2)
        if token_mask is not None:
            am = token_mask.unsqueeze(1).expand(bsz, self.num_heads, src_len).reshape(bh, src_len)
            allowed = torch.gather(
                am.unsqueeze(1).expand(bh, tgt_len, src_len), 2, local_idx_exp
            )
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        return torch.bmm(
            attn.reshape(bh * tgt_len, 1, w),
            v_sel.reshape(bh * tgt_len, w, self.head_dim),
        ).reshape(bh, tgt_len, self.head_dim)

    def _compressed_branch(self, Q, K, V, bsz, tgt_len, token_mask):
        """Compressed attention: attend to learnable block summaries."""
        bh = Q.size(0)
        k_cmp = self._compress_blocks(K, self.compress_k)
        v_cmp = self._compress_blocks(V, self.compress_v)
        n_cmp = k_cmp.size(1)

        scores = torch.bmm(Q, k_cmp.transpose(1, 2))  # [BH, Tq, n_cmp]

        if token_mask is not None:
            blk = self.block_size
            n_blocks = n_cmp
            block_starts = torch.arange(n_blocks, device=Q.device) * self.stride
            block_starts = block_starts.clamp(max=token_mask.size(-1) - 1)
            block_ok = token_mask[:, block_starts]  # [B, n_cmp]
            block_ok = (
                block_ok.unsqueeze(1)
                .expand(bsz, self.num_heads, n_cmp)
                .reshape(bh, 1, n_cmp)
                .expand(-1, tgt_len, -1)
            )
            scores = scores.masked_fill(~block_ok, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn, v_cmp)

    def _selected_branch(self, Q, K, V, bsz, tgt_len, token_mask):
        """Selected attention: top-k blocks via block-mean routing."""
        bh, src_len, dim = K.shape
        blk = self.block_size
        n_blocks = math.ceil(src_len / blk)
        k_blocks = self._block_means(K)  # [BH, n_blocks, D]

        block_scores = torch.bmm(Q, k_blocks.transpose(1, 2))  # [BH, Tq, n_blocks]

        # Bidirectional: no causal mask, just padding
        if token_mask is not None:
            block_starts = torch.arange(n_blocks, device=Q.device) * blk
            block_starts = block_starts.clamp(max=token_mask.size(-1) - 1)
            block_ok = token_mask[:, block_starts]  # [B, n_blocks]
            block_ok = (
                block_ok.unsqueeze(1)
                .expand(bsz, self.num_heads, n_blocks)
                .reshape(bh, 1, n_blocks)
                .expand(-1, tgt_len, -1)
            )
            block_scores = block_scores.masked_fill(~block_ok, torch.finfo(block_scores.dtype).min)

        m = min(self.topk_blocks, n_blocks)
        _, top_blocks = torch.topk(block_scores, k=m, dim=-1)  # [BH, Tq, m]

        # Expand block indices to token indices
        block_offset = torch.arange(blk, device=Q.device).view(1, 1, 1, blk)
        base_idx = (top_blocks.unsqueeze(2) * blk).unsqueeze(-1)
        token_idx = (base_idx + block_offset).reshape(bh, tgt_len, m * blk).clamp(max=src_len - 1)

        m_tokens = m * blk
        idx_gather = token_idx.unsqueeze(-1).expand(-1, -1, -1, dim)
        k_sel = torch.gather(K.unsqueeze(1).expand(bh, tgt_len, src_len, dim), 2, idx_gather)
        v_sel = torch.gather(V.unsqueeze(1).expand(bh, tgt_len, src_len, dim), 2, idx_gather)

        scores_sel = torch.matmul(Q.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2)

        if token_mask is not None:
            am = token_mask.unsqueeze(1).expand(bsz, self.num_heads, src_len).reshape(bh, src_len)
            allowed = torch.gather(
                am.unsqueeze(1).expand(bh, tgt_len, src_len), 2, token_idx
            )
            scores_sel = scores_sel.masked_fill(~allowed, torch.finfo(scores_sel.dtype).min)

        attn = F.softmax(scores_sel, dim=-1)
        return torch.bmm(
            attn.reshape(bh * tgt_len, 1, m_tokens),
            v_sel.reshape(bh * tgt_len, m_tokens, dim),
        ).reshape(bh, tgt_len, dim)

    def _causal_window_branch(self, Q, K, V, bsz, tgt_len, token_mask):
        """Causal sliding-window attention: query t attends to keys [max(0, t-W+1), t]."""
        bh = Q.size(0)
        src_len = K.size(1)
        w = min(self.window_size, src_len)
        d = self.head_dim

        t = torch.arange(tgt_len, device=Q.device)
        starts = torch.clamp(t - w + 1, min=0)
        offsets = torch.arange(w, device=Q.device)
        local_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)  # [Tq, W]
        local_idx = local_idx.clamp(max=src_len - 1)
        local_idx_exp = local_idx.unsqueeze(0).expand(bh, -1, -1)  # [BH, Tq, W]

        idx_gather = local_idx_exp.unsqueeze(-1).expand(-1, -1, -1, d)
        k_sel = torch.gather(K.unsqueeze(1).expand(bh, tgt_len, src_len, d), 2, idx_gather)
        v_sel = torch.gather(V.unsqueeze(1).expand(bh, tgt_len, src_len, d), 2, idx_gather)

        scores = torch.matmul(Q.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2)

        # Causal mask
        q_pos = t.view(1, -1, 1)
        causal_allowed = local_idx.unsqueeze(0) <= q_pos
        scores = scores.masked_fill(~causal_allowed, torch.finfo(scores.dtype).min)

        if token_mask is not None:
            am = token_mask.unsqueeze(1).expand(bsz, self.num_heads, src_len).reshape(bh, src_len)
            allowed = torch.gather(am.unsqueeze(1).expand(-1, tgt_len, -1), 2, local_idx_exp)
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        return torch.bmm(
            attn.reshape(bh * tgt_len, 1, w),
            v_sel.reshape(bh * tgt_len, w, d),
        ).reshape(bh, tgt_len, d)

    def _causal_compressed_branch(self, Q, K, V, bsz, tgt_len, token_mask):
        """Causal compressed attention: attend to block summaries with causal masking."""
        bh = Q.size(0)
        d = self.head_dim
        k_cmp = self._compress_blocks(K, self.compress_k)
        v_cmp = self._compress_blocks(V, self.compress_v)
        n_cmp = k_cmp.size(1)

        scores = torch.bmm(Q, k_cmp.transpose(1, 2))  # [BH, Tq, n_cmp]

        # Causal mask: block b is allowed for query t if block_start(b) <= t
        blk = self.block_size
        block_starts = torch.arange(n_cmp, device=Q.device) * blk  # [n_cmp]
        q_pos = torch.arange(tgt_len, device=Q.device).unsqueeze(-1)  # [Tq, 1]
        causal_allowed = block_starts.unsqueeze(0) <= q_pos  # [Tq, n_cmp]
        scores = scores.masked_fill(~causal_allowed.unsqueeze(0), torch.finfo(scores.dtype).min)

        if token_mask is not None:
            block_ok_starts = block_starts.clamp(max=token_mask.size(-1) - 1)
            block_ok = token_mask[:, block_ok_starts]  # [B, n_cmp]
            block_ok = block_ok.unsqueeze(1).expand(bsz, self.num_heads, n_cmp).reshape(bh, 1, n_cmp).expand(-1, tgt_len, -1)
            scores = scores.masked_fill(~block_ok, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn, v_cmp)

    def _causal_selected_branch(self, Q, K, V, bsz, tgt_len, token_mask):
        """Causal selected attention: top-k blocks via last-query routing + causal masking."""
        bh, src_len, dim = K.shape
        blk = self.block_size
        d_low = min(self.head_dim, self.head_dim)

        # Use last query for block routing (causal-safe)
        Q_low = Q[:, :, :d_low]
        K_low = K[:, :, :d_low]
        routed_idx = last_query_block_topk_indices(
            Q_low, K_low, blk, self.topk_blocks,
            token_mask, bsz, self.num_heads,
        )  # [BH, k*blk]

        # Build per-query indices: routed indices are head-shared, expand to all queries
        M = routed_idx.size(-1)
        all_idx = routed_idx.unsqueeze(1).expand(-1, tgt_len, -1)  # [BH, Tq, M]

        return causal_sparse_attention_with_indices(
            Q, K, V, all_idx, token_mask, bsz, self.num_heads,
        )

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        BH, tgt_len, _ = Q.shape
        src_len = K.size(1)

        sparse_budget = self.window_size + self.topk_blocks * self.block_size
        if src_len <= sparse_budget:
            return dense_self_attention(
                Q, K, V, token_mask if not is_causal else None,
                bsz, num_heads, 0.0, self.training, is_causal=is_causal,
            )

        if is_causal:
            out_win = self._causal_window_branch(Q, K, V, bsz, tgt_len, token_mask)
            out_cmp = self._causal_compressed_branch(Q, K, V, bsz, tgt_len, token_mask)
            out_slc = self._causal_selected_branch(Q, K, V, bsz, tgt_len, token_mask)
        else:
            out_win = self._window_branch(Q, bsz, tgt_len, token_mask)
            out_cmp = self._compressed_branch(Q, K, V, bsz, tgt_len, token_mask)
            out_slc = self._selected_branch(Q, K, V, bsz, tgt_len, token_mask)

        hidden = self._hidden_states
        gates = F.softmax(self.gate_mlp(hidden), dim=-1)  # [B, T, 3]
        g = gates.unsqueeze(1).expand(bsz, num_heads, tgt_len, 3).reshape(BH, tgt_len, 3)
        return g[..., 0:1] * out_win + g[..., 1:2] * out_cmp + g[..., 2:3] * out_slc


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    block_size: int = 32,
    stride: int = 32,
    topk_blocks: int = 4,
    window_size: int = 128,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=NSAAttention,
        num_labels=num_labels,
        attn_kwargs={
            "block_size": block_size,
            "stride": stride,
            "topk_blocks": topk_blocks,
            "window_size": window_size,
            "use_triton": False,
        },
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
