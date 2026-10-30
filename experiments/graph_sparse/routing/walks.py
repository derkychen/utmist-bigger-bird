"""Traveling paths: multi-layer information propagation via reserved walk edges.

A walk follows causally valid block-to-block hops across layers:
  layer L: query chunk A attends to key block X
  layer L+1: query chunk covering X attends to key block Y
  => A's information can reach Y through X in two hops.

Walk algorithms:
  random_walk   -- choose the next valid destination uniformly at random.
  semantic_walk -- choose destinations based on semantic similarity (cosine).
  novelty_walk  -- prefer destinations not already visited by this path.

Walks reserve slots from the normal edge budget so total attention cost is controlled.
"""
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class Walk:
    source_chunk: int
    current_block: int
    start_layer: int
    hops_completed: int
    walk_type: str = "random"
    visited: List[int] = field(default_factory=list)
    alive: bool = True
    completed: bool = False


def random_walk(forward_blocks: torch.Tensor, walk: Walk,
                generator: torch.Generator = None) -> Optional[int]:
    """Pick the next block uniformly at random from causally-forward candidates."""
    if forward_blocks.numel() == 0:
        return None
    idx = torch.randint(forward_blocks.numel(), (1,), generator=generator).item()
    return forward_blocks[idx].item()


def semantic_walk(forward_blocks: torch.Tensor, walk: Walk,
                  similarity: Optional[torch.Tensor], covering_chunk: int,
                  generator: torch.Generator = None) -> Optional[int]:
    """Pick the next block weighted by semantic similarity to the covering query chunk.

    Falls back to random if no similarity is available.
    """
    if forward_blocks.numel() == 0:
        return None
    if similarity is None:
        return random_walk(forward_blocks, walk, generator)
    scores = similarity[covering_chunk, forward_blocks]
    valid = scores > float("-inf")
    if not valid.any():
        return random_walk(forward_blocks, walk, generator)
    probs = torch.softmax(scores[valid].float(), dim=0)
    sel = torch.multinomial(probs, 1, generator=generator).item()
    return forward_blocks[valid][sel].item()


def novelty_walk(forward_blocks: torch.Tensor, walk: Walk,
                 generator: torch.Generator = None) -> Optional[int]:
    """Prefer blocks not already visited by this walk path.

    Falls back to any forward block if all candidates are visited.
    """
    if forward_blocks.numel() == 0:
        return None
    visited_set = set(walk.visited)
    unvisited_mask = torch.tensor(
        [b.item() not in visited_set for b in forward_blocks],
        dtype=torch.bool, device=forward_blocks.device)
    candidates = forward_blocks[unvisited_mask] if unvisited_mask.any() else forward_blocks
    idx = torch.randint(candidates.numel(), (1,), generator=generator).item()
    return candidates[idx].item()


WALK_DISPATCH = {
    "random": random_walk,
    "semantic": semantic_walk,
    "novelty": novelty_walk,
}


def _get_forward_blocks(edges: torch.Tensor, covering_chunk: int,
                        current_block: int, num_blocks: int) -> torch.Tensor:
    """Get causally-forward valid blocks from the covering chunk's edges."""
    chunk_edges = edges[covering_chunk]
    valid = chunk_edges[(chunk_edges >= 0) & (chunk_edges < num_blocks)]
    return valid[valid > current_block]


def advance_walks(active_walks: list, edges: torch.Tensor, layer_idx: int,
                  chunk_size: int, block_size: int, seq_len: int,
                  walk_config=None, similarity: Optional[torch.Tensor] = None,
                  generator: torch.Generator = None) -> list:
    """Advance each active walk by one hop using the current layer's edges.

    For each walk at block B, find the query chunk that covers B's tokens, then
    pick a causally-forward routed block as the next hop using the walk's algorithm.
    Walks terminate when: they can't advance, they exceed walk_length, or a
    probabilistic continuation/restart check fails.

    Args:
        active_walks: list of Walk objects.
        edges: [Q, total_blocks] block indices for this layer.
        layer_idx: current layer.
        chunk_size: query chunk size.
        block_size: key block size.
        seq_len: total sequence length.
        walk_config: optional WalkConfig for continuation/restart control.
        similarity: optional [Q, B] similarity scores for semantic_walk.
        generator: optional torch.Generator for reproducibility.

    Returns:
        Updated list of Walk objects (dead walks removed, restarted walks included).
    """
    Q = edges.shape[0]
    num_blocks = -(-seq_len // block_size)
    advanced = []

    max_length = walk_config.walk_length if walk_config else 3
    cont_prob = walk_config.continuation_probability if walk_config else 1.0
    restart_prob = walk_config.restart_probability if walk_config else 0.0
    avoid = walk_config.avoid_revisits if walk_config else True

    for w in active_walks:
        if not w.alive:
            continue
        if w.hops_completed >= max_length:
            w.alive = False
            w.completed = True
            continue

        if cont_prob < 1.0:
            r = torch.rand(1, generator=generator).item()
            if r > cont_prob:
                w.alive = False
                continue

        token_pos = w.current_block * block_size
        covering_chunk = min(token_pos // chunk_size, Q - 1)
        forward = _get_forward_blocks(edges, covering_chunk, w.current_block, num_blocks)

        if avoid and forward.numel() > 0:
            visited_set = set(w.visited)
            unvisited = torch.tensor(
                [b.item() not in visited_set for b in forward],
                dtype=torch.bool, device=forward.device)
            if unvisited.any():
                filtered = forward[unvisited]
            else:
                filtered = forward
        else:
            filtered = forward

        next_block = None
        if w.walk_type == "semantic":
            next_block = semantic_walk(filtered, w, similarity, covering_chunk, generator)
        elif w.walk_type == "novelty":
            next_block = novelty_walk(filtered, w, generator)
        else:
            next_block = random_walk(filtered, w, generator)

        if next_block is None:
            if restart_prob > 0:
                r = torch.rand(1, generator=generator).item()
                if r < restart_prob:
                    chunk_edges = edges[w.source_chunk]
                    start_cands = chunk_edges[(chunk_edges >= 0) & (chunk_edges < num_blocks)]
                    if start_cands.numel() > 0:
                        idx = torch.randint(start_cands.numel(), (1,), generator=generator).item()
                        restart_block = start_cands[idx].item()
                        advanced.append(Walk(
                            source_chunk=w.source_chunk,
                            current_block=restart_block,
                            start_layer=layer_idx,
                            hops_completed=0,
                            walk_type=w.walk_type,
                            visited=[restart_block],
                            alive=True,
                            completed=False,
                        ))
                    continue
            w.alive = False
            continue

        new_visited = w.visited + [next_block]
        advanced.append(Walk(
            source_chunk=w.source_chunk,
            current_block=next_block,
            start_layer=w.start_layer,
            hops_completed=w.hops_completed + 1,
            walk_type=w.walk_type,
            visited=new_visited,
            alive=True,
            completed=False,
        ))

    return advanced


def spawn_walks(edges: torch.Tensor, num_query_chunks: int,
                walk_config, layer_idx: int, num_blocks: int,
                generator: torch.Generator = None) -> list:
    """Create initial walks from each query chunk's routed edges.

    For each query chunk, spawns walks_per_start walks starting from distinct
    valid routed blocks.
    """
    walks = []
    walks_per_start = walk_config.walks_per_start
    walk_type = walk_config.walk_type

    for q in range(num_query_chunks):
        chunk_edges = edges[q]
        valid = chunk_edges[(chunk_edges >= 0) & (chunk_edges < num_blocks)]
        if valid.numel() == 0:
            continue
        n_spawn = min(walks_per_start, valid.numel())
        if n_spawn == valid.numel():
            selected = valid
        else:
            perm = torch.randperm(valid.numel(), generator=generator,
                                  device=edges.device)[:n_spawn]
            selected = valid[perm]
        for i in range(n_spawn):
            block = selected[i].item()
            walks.append(Walk(
                source_chunk=q,
                current_block=block,
                start_layer=layer_idx,
                hops_completed=0,
                walk_type=walk_type,
                visited=[block],
                alive=True,
                completed=False,
            ))
    return walks


def inject_walk_edges(edges: torch.Tensor, active_walks: list,
                      reserved_budget: int) -> torch.Tensor:
    """Replace the last ``reserved_budget`` edge columns with walk destinations.

    Each walk injects its current_block into the source_chunk's reserved slot.
    If more walks target the same source_chunk than slots available, the excess
    is silently dropped (longest path wins by list order).

    Args:
        edges: [Q, total_blocks] block indices.
        active_walks: list of Walk objects.
        reserved_budget: how many edge slots per query chunk to reserve for walks.

    Returns:
        [Q, total_blocks] with walk edges injected.
    """
    if reserved_budget == 0 or not active_walks:
        return edges
    result = edges.clone()
    Q, total = result.shape
    slots_used = {}
    for w in active_walks:
        if not w.alive:
            continue
        q = w.source_chunk
        if q >= Q:
            continue
        used = slots_used.get(q, 0)
        if used >= reserved_budget:
            continue
        slot = total - reserved_budget + used
        if 0 <= slot < total:
            result[q, slot] = w.current_block
            slots_used[q] = used + 1
    return result


def walk_diagnostics(active_walks: list, completed_walks: list = None) -> dict:
    """Compute walk statistics for monitoring.

    Returns:
        Dict with: active_count, completed_count, mean_path_length,
        max_path_length, unique_blocks_visited, walks_by_type.
    """
    all_walks = list(active_walks)
    if completed_walks:
        all_walks.extend(completed_walks)
    if not all_walks:
        return {
            "active_count": 0,
            "completed_count": 0,
            "mean_path_length": 0.0,
            "max_path_length": 0,
            "unique_blocks_visited": 0,
            "walks_by_type": {},
        }
    active = [w for w in all_walks if w.alive]
    completed = [w for w in all_walks if w.completed]
    lengths = [w.hops_completed for w in all_walks]
    all_visited: Set[int] = set()
    by_type: dict = {}
    for w in all_walks:
        all_visited.update(w.visited)
        by_type[w.walk_type] = by_type.get(w.walk_type, 0) + 1
    return {
        "active_count": len(active),
        "completed_count": len(completed),
        "mean_path_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "max_path_length": max(lengths) if lengths else 0,
        "unique_blocks_visited": len(all_visited),
        "walks_by_type": by_type,
    }
