"""Exp 15 — BiggerBird (block-sparse with diverse selection) on R1-Distill-Llama-8B.

Block-sparse attention combining:
  (1) Local window: nearby blocks (sliding window at block granularity)
  (2) Global anchors: first and last blocks (always attended to)
  (3) Top-k block selection: block-mean routing picks most relevant blocks
  (4) Optional teleports: biased random blocks toward high-gate tokens

This is a Llama-3 port of the original BigBird-based experiment. The original
patched BigBirdBlockSparseAttention (890 lines); this port reimplements the
sparse pattern as a LlamaSparseAttention subclass using gather-indices attention.

Key simplification: the original's MMR diversity, prototype selection, and
share-stride layers are collapsed into block-mean top-k routing + local window
+ globals. The core sparse pattern (O(N * M) instead of O(N²)) is preserved.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.llama_patched_model import (
    LlamaSparseAttention,
    LlamaPatchedModel,
    apply_lora,
)
from shared.sparse_attn_utils import dense_self_attention, causal_sparse_attention


class BiggerBirdAttention(LlamaSparseAttention):
    def __init__(
        self,
        base_attn,
        fragment_size: int = 128,
        max_k: int = 64,
        min_k: int = 56,
        globals_per_head: int = 6,
        teleports_per_head: int = 4,
        teleport_bias_frac: float = 0.75,
        use_teleports: bool = False,
        use_triton: bool = False,
    ):
        super().__init__(base_attn)
        self.fragment_size = fragment_size
        self.max_k = max_k
        self.min_k = min_k
        self.globals_per_head = globals_per_head
        self.teleports_per_head = teleports_per_head
        self.teleport_bias_frac = teleport_bias_frac
        self.use_teleports = use_teleports

        # Learned global importance gate
        self.global_gate = nn.Linear(self.config.hidden_size, 1)
        nn.init.zeros_(self.global_gate.bias)
        nn.init.normal_(self.global_gate.weight, std=0.02)

    def forward(self, hidden_states, **kwargs):
        self._hidden_states = hidden_states
        return super().forward(hidden_states, **kwargs)

    def _block_means(self, K, n_blocks, blk):
        """Mean-pool keys into block representatives [BH, n_blocks, D]."""
        bh, src_len, dim = K.shape
        pad = (blk - src_len % blk) % blk
        if pad:
            K = F.pad(K, (0, 0, 0, pad))
        n_actual = K.size(1) // blk
        means = K.view(bh, n_actual, blk, dim).mean(dim=2)
        if n_actual < n_blocks:
            # Pad with zeros
            extra = torch.zeros(bh, n_blocks - n_actual, dim, device=K.device, dtype=K.dtype)
            means = torch.cat([means, extra], dim=1)
        return means

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        BH, tgt_len, _ = Q.shape
        src_len = K.size(1)
        blk = self.fragment_size
        n_blocks = (src_len + blk - 1) // blk

        # Effective k scaled by sequence length
        k_eff = max(self.min_k, min(self.max_k, src_len))
        total_budget = k_eff + self.globals_per_head * blk
        if self.use_teleports:
            total_budget += self.teleports_per_head * blk

        if src_len <= total_budget:
            return dense_self_attention(
                Q, K, V, None, bsz, num_heads, 0.0, self.training,
                is_causal=is_causal,
            )

        # Causal mode: use block routing to select candidate blocks, then
        # use causal_sparse_attention with local window for actual attention.
        # This fixes the 53% accuracy issue by adding a token-level local
        # window that catches the needle even when block routing misses it.
        if is_causal:
            # Block routing: use last query to select top-k blocks (head-shared)
            k_blocks = max(1, k_eff // blk)
            k_blocks = min(k_blocks, n_blocks)
            block_means = self._block_means(K, n_blocks, blk)  # [BH, n_blocks, D]
            # Use last query position for routing (causal-safe)
            q_last = Q[:, -1:, :]  # [BH, 1, D]
            block_scores = torch.bmm(q_last, block_means.transpose(1, 2)).squeeze(1)  # [BH, n_blocks]

            if token_mask is not None:
                block_starts = torch.arange(n_blocks, device=Q.device) * blk
                block_starts = block_starts.clamp(max=token_mask.size(-1) - 1)
                block_ok = token_mask[:, block_starts]  # [B, n_blocks]
                block_ok = block_ok.unsqueeze(1).expand(bsz, num_heads, n_blocks).reshape(BH, n_blocks)
                block_scores = block_scores.masked_fill(~block_ok, torch.finfo(block_scores.dtype).min)

            _, top_blocks = torch.topk(block_scores, k=k_blocks, dim=-1)  # [BH, k_blocks]

            # Convert block indices to token indices
            block_offset = torch.arange(blk, device=Q.device).view(1, 1, blk)
            base_idx = (top_blocks.unsqueeze(-1) * blk).unsqueeze(-1)  # [BH, k_blocks, 1, blk]
            selected_idx = (base_idx + block_offset).reshape(BH, k_blocks * blk)  # [BH, k_blocks*blk]
            selected_idx = selected_idx.clamp(max=src_len - 1)

            # Add global tokens (first and last blocks)
            global_tokens = list(range(min(self.globals_per_head * blk // 2, src_len)))
            global_tokens += list(range(max(0, src_len - self.globals_per_head * blk // 2), src_len))
            global_tokens = sorted(set(global_tokens))
            global_idx = torch.tensor(global_tokens, device=Q.device, dtype=torch.long)
            global_idx = global_idx.unsqueeze(0).expand(BH, -1)

            routed_idx = torch.cat([selected_idx, global_idx], dim=-1)  # [BH, k_blocks*blk + globals]

            # Use shared causal_sparse_attention with local window=256
            # This adds a token-level local window that catches the needle
            # even when block-mean routing misses its block
            return causal_sparse_attention(
                Q, K, V, routed_idx, local_window=256,
                token_mask=token_mask, bsz=bsz, num_heads=num_heads,
            )

        # --- Bidirectional mode: original per-query block routing ---
        # --- Block-mean routing: pick top-k blocks per query ---
        k_blocks = max(1, k_eff // blk)
        k_blocks = min(k_blocks, n_blocks)
        block_means = self._block_means(K, n_blocks, blk)  # [BH, n_blocks, D]
        block_scores = torch.bmm(Q, block_means.transpose(1, 2))  # [BH, Tq, n_blocks]

        if token_mask is not None:
            block_starts = torch.arange(n_blocks, device=Q.device) * blk
            block_starts = block_starts.clamp(max=token_mask.size(-1) - 1)
            block_ok = token_mask[:, block_starts]  # [B, n_blocks]
            block_ok = (
                block_ok.unsqueeze(1)
                .expand(bsz, num_heads, n_blocks)
                .reshape(BH, 1, n_blocks)
                .expand(-1, tgt_len, -1)
            )
            block_scores = block_scores.masked_fill(~block_ok, torch.finfo(block_scores.dtype).min)

        _, top_blocks = torch.topk(block_scores, k=k_blocks, dim=-1)  # [BH, Tq, k_blocks]

        # --- Build token indices from selected blocks ---
        block_offset = torch.arange(blk, device=Q.device).view(1, 1, 1, blk)
        base_idx = (top_blocks.unsqueeze(2) * blk).unsqueeze(-1)
        selected_idx = (base_idx + block_offset).reshape(BH, tgt_len, k_blocks * blk)
        selected_idx = selected_idx.clamp(max=src_len - 1)

        # --- Add global tokens (first and last blocks) ---
        global_tokens = list(range(min(self.globals_per_head * blk // 2, src_len)))
        global_tokens += list(range(max(0, src_len - self.globals_per_head * blk // 2), src_len))
        global_tokens = sorted(set(global_tokens))
        global_idx = torch.tensor(global_tokens, device=Q.device, dtype=torch.long)
        global_idx = global_idx.unsqueeze(0).unsqueeze(0).expand(BH, tgt_len, -1)

        all_idx = torch.cat([selected_idx, global_idx], dim=-1)

        # --- Optional teleports: biased random blocks ---
        if self.use_teleports and self.teleports_per_head > 0:
            hidden = self._hidden_states
            g_scores = self.global_gate(
                hidden.to(self.global_gate.weight.dtype)
            ).squeeze(-1)  # [B, T]
            if token_mask is not None:
                g_scores = g_scores.masked_fill(~token_mask, -1e9)
            n_teleport_blocks = self.teleports_per_head
            _, top_gate_blocks = torch.topk(
                g_scores, k=min(n_teleport_blocks * 2, src_len), dim=-1
            )  # [B, C]
            # Sample teleport blocks from top-gate candidates
            n_biased = int(n_teleport_blocks * self.teleport_bias_frac)
            n_random = n_teleport_blocks - n_biased
            teleport_indices = []
            if n_biased > 0:
                rand_pick = torch.randint(
                    0, top_gate_blocks.size(-1), (bsz, n_biased), device=Q.device
                )
                biased_idx = torch.gather(top_gate_blocks, 1, rand_pick)  # [B, n_biased]
                biased_idx = biased_idx.unsqueeze(1).expand(bsz, tgt_len, n_biased)
                biased_idx = biased_idx.unsqueeze(1).expand(bsz, num_heads, tgt_len, n_biased)
                biased_idx = biased_idx.reshape(BH, tgt_len, n_biased)
                teleport_indices.append(biased_idx)
            if n_random > 0:
                random_idx = torch.randint(0, src_len, (BH, tgt_len, n_random), device=Q.device)
                teleport_indices.append(random_idx)
            teleport_idx = torch.cat(teleport_indices, dim=-1)
            all_idx = torch.cat([all_idx, teleport_idx], dim=-1)

        # --- Gather attention over selected tokens (memory-efficient chunked) ---
        M = all_idx.size(-1)
        dim = self.head_dim

        # Precompute token_mask expanded for gather (avoid recompute per chunk)
        if token_mask is not None:
            am_expanded = token_mask.unsqueeze(1).expand(
                bsz, num_heads, src_len
            ).reshape(BH, src_len)  # [BH, src_len]
        else:
            am_expanded = None

        # Process queries in chunks to limit peak memory
        # Peak per chunk: BH * chunk * M * dim * 2 (K_sel + V_sel) * 2 bytes
        # For BH=32, M=1024, dim=128: chunk=256 -> 32*256*1024*128*4 = 4GB (manageable)
        QUERY_CHUNK = 256
        out_chunks = []
        for q_start in range(0, tgt_len, QUERY_CHUNK):
            q_end = min(q_start + QUERY_CHUNK, tgt_len)
            Q_chunk = Q[:, q_start:q_end, :]  # [BH, chunk, d]
            idx_chunk = all_idx[:, q_start:q_end, :]  # [BH, chunk, M]

            # Gather K, V for this chunk: [BH, chunk, M, d]
            from shared.sparse_attn_utils import _gather_kv
            k_sel, v_sel = _gather_kv(K, V, idx_chunk)

            # Scores: [BH, chunk, M]
            scores = torch.matmul(Q_chunk.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2)

            if am_expanded is not None:
                allowed = torch.gather(
                    am_expanded.unsqueeze(1).expand(-1, q_end - q_start, -1), 2, idx_chunk
                )
                scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

            if is_causal:
                q_pos = torch.arange(q_start, q_end, device=Q.device).unsqueeze(0).unsqueeze(-1)
                causal_allowed = idx_chunk <= q_pos  # [BH, chunk, M]
                scores = scores.masked_fill(~causal_allowed, torch.finfo(scores.dtype).min)

            attn = F.softmax(scores, dim=-1)
            chunk_out = torch.bmm(
                attn.reshape(BH * (q_end - q_start), 1, M),
                v_sel.reshape(BH * (q_end - q_start), M, dim),
            ).reshape(BH, q_end - q_start, dim)
            out_chunks.append(chunk_out)

        return torch.cat(out_chunks, dim=1)


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    fragment_size: int = 128,
    max_k: int = 64,
    min_k: int = 56,
    globals_per_head: int = 6,
    teleports_per_head: int = 4,
    use_teleports: bool = False,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    pooling: str = "last",
):
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=BiggerBirdAttention,
        num_labels=num_labels,
        attn_kwargs={
            "fragment_size": fragment_size,
            "max_k": max_k,
            "min_k": min_k,
            "globals_per_head": globals_per_head,
            "teleports_per_head": teleports_per_head,
            "use_teleports": use_teleports,
            "use_triton": False,
        },
        pooling=pooling,
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
