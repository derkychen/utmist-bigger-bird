"""Block summarization and candidate generation for graph routing.

Query summaries: first token of each query chunk (pre-RoPE).
Key summaries: mean-pooled key blocks (pre-RoPE).
Semantic projection: optional low-rank projection to routing_dim.
"""
import torch
import torch.nn.functional as F


def build_block_summaries(hidden: torch.Tensor, block_size: int, mode: str = "mean") -> torch.Tensor:
    """Summarize contiguous blocks of hidden states.

    Args:
        hidden: [N, D] token hidden states (pre-RoPE).
        block_size: tokens per block.
        mode: "mean" or "first".

    Returns:
        [num_blocks, D] block summaries. The last block is shorter if N % block_size != 0.
    """
    n, d = hidden.shape
    num_full = n // block_size
    parts = []
    if num_full > 0:
        full = hidden[:num_full * block_size].view(num_full, block_size, d)
        parts.append(full.mean(1) if mode == "mean" else full[:, 0])
    if n % block_size:
        tail = hidden[num_full * block_size:]
        parts.append((tail.mean(0) if mode == "mean" else tail[0]).unsqueeze(0))
    return torch.cat(parts, 0)


def build_query_summaries(hidden: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """First token of each query chunk as the chunk's routing representative.

    Args:
        hidden: [N, D] token hidden states (pre-RoPE).
        chunk_size: tokens per query chunk.

    Returns:
        [num_chunks, D] query summaries.
    """
    n = hidden.shape[0]
    indices = torch.arange(0, n, chunk_size, device=hidden.device)
    return hidden[indices]


def build_question_summaries(hidden: torch.Tensor, chunk_size: int,
                             question_tokens: int = 64) -> torch.Tensor:
    """Mean of the last ``question_tokens`` hidden states, broadcast to all chunks.

    Every query chunk uses the same question-derived representation so the
    routing decision for every chunk is conditioned on the final question
    rather than on the chunk's own first token.

    Args:
        hidden: [N, D] token hidden states (pre-RoPE).
        chunk_size: tokens per query chunk.
        question_tokens: number of trailing tokens to average.

    Returns:
        [num_chunks, D] query summaries (all rows identical).
    """
    n = hidden.shape[0]
    num_chunks = -(-n // chunk_size)
    q_start = max(0, n - question_tokens)
    question_repr = hidden[q_start:].mean(0, keepdim=True)
    return question_repr.expand(num_chunks, -1)


def project_for_routing(summaries: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """Low-rank projection: [M, D] @ [D, routing_dim] -> [M, routing_dim], L2-normalized."""
    out = summaries.float() @ proj.float()
    return F.normalize(out, dim=-1)


def compute_similarity(query_summaries: torch.Tensor, key_summaries: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between query chunks and key blocks.

    Args:
        query_summaries: [Q, routing_dim] L2-normalized.
        key_summaries: [B, routing_dim] L2-normalized.

    Returns:
        [Q, B] similarity scores in [-1, 1].
    """
    return query_summaries @ key_summaries.T


def causal_block_mask(num_query_chunks: int, num_key_blocks: int,
                      chunk_size: int, block_size: int,
                      seq_len: int, device: torch.device) -> torch.Tensor:
    """Boolean mask: True where a key block is fully completed before the query chunk starts.

    For strict causal routing, query chunk q (starting at q*chunk_size) can only attend to
    key block b if the block's last token (b*block_size + block_size - 1) < q*chunk_size.

    Args:
        num_query_chunks: Q
        num_key_blocks: B
        chunk_size: tokens per query chunk
        block_size: tokens per key block
        seq_len: total sequence length

    Returns:
        [Q, B] boolean mask.
    """
    q_starts = torch.arange(num_query_chunks, device=device) * chunk_size
    b_ends = torch.arange(num_key_blocks, device=device) * block_size + block_size - 1
    b_ends = b_ends.clamp(max=seq_len - 1)
    return b_ends[None, :] < q_starts[:, None]


def find_candidates(hidden: torch.Tensor, config, proj: torch.Tensor = None, causal: bool = True):
    """Full candidate pipeline: summarize, project, score, mask.

    Complexity: O(Q * B * routing_dim) for the similarity matmul, where Q = ceil(N / chunk)
    and B = ceil(N / block). This is an all-pairs block-level computation, NOT linear in N.
    For N=128K, chunk=512, block=64: Q=256, B=2048, routing_dim=64 → ~33M FLOPs — negligible
    versus attention itself.

    When ``config.semantic.question_conditioned`` is True, all query chunks use the
    mean of the last ``question_tokens`` hidden states as their representative,
    routing every chunk toward blocks relevant to the final question.

    Args:
        hidden: [N, D] pre-RoPE hidden states for one sequence.
        config: GraphConfig.
        proj: optional [D, routing_dim] projection matrix.
        causal: if True (strict_causal mode), only blocks fully before each query chunk
            are eligible. If False (offline mode), all blocks are eligible and the kernel
            enforces per-token causality.

    Returns:
        similarity: [Q, B] causal-masked similarity (ineligible = -inf).
            If config.semantic.candidate_count > 0, only the top candidates per row
            remain unmasked; the rest are set to -inf.
        causal_mask: [Q, B] boolean (causal eligibility, before candidate_count pruning).
        num_query_chunks: int.
        num_key_blocks: int.
    """
    n, d = hidden.shape
    num_query_chunks = -(-n // config.query_chunk_size)
    num_key_blocks = -(-n // config.key_block_size)

    sem = config.semantic
    if getattr(sem, "question_conditioned", False):
        q_tokens = getattr(sem, "question_tokens", 64)
        q_sum = build_question_summaries(hidden, config.query_chunk_size, q_tokens)
    else:
        q_sum = build_query_summaries(hidden, config.query_chunk_size)
    k_sum = build_block_summaries(hidden, config.key_block_size, mode="mean")

    if proj is not None:
        q_sum = project_for_routing(q_sum, proj)
        k_sum = project_for_routing(k_sum, proj)
    else:
        q_sum = F.normalize(q_sum.float(), dim=-1)
        k_sum = F.normalize(k_sum.float(), dim=-1)

    sim = compute_similarity(q_sum, k_sum)
    if causal:
        mask = causal_block_mask(num_query_chunks, num_key_blocks,
                                 config.query_chunk_size, config.key_block_size,
                                 n, hidden.device)
    else:
        mask = torch.ones(num_query_chunks, num_key_blocks, dtype=torch.bool, device=hidden.device)
    sim = sim.masked_fill(~mask, float("-inf"))

    cand_count = getattr(config.semantic, "candidate_count", 0)
    if cand_count > 0 and cand_count < num_key_blocks:
        _, top_idx = sim.topk(min(cand_count, mask.sum(1).max().item()), dim=1)
        pruned = torch.full_like(sim, float("-inf"))
        pruned.scatter_(1, top_idx, sim.gather(1, top_idx))
        sim = pruned

    return sim, mask, num_query_chunks, num_key_blocks
