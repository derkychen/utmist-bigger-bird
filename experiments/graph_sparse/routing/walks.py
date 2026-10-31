"""Traveling paths: multi-layer information propagation via reserved walk edges.

Information flows through attention: at layer L, query chunk Q attends to key
block B, so B's information reaches Q.  A walk traces this chain across layers:

  Layer L:   query chunk Q0 attends to key block B0  (B0's info reaches Q0)
  Layer L+1: query chunk covering B0 attends to key block B1  (B0->B1 hop)
  Layer L+2: query chunk covering B1 attends to key block B2  (B1->B2 hop)

The walk reserves an edge slot in the *receiving* query chunk at each layer so
the kernel actually computes the attention that carries the information.

Walk algorithms:
  random_walk   -- choose the next destination uniformly at random.
  semantic_walk -- choose destinations weighted by semantic similarity.
  novelty_walk  -- prefer destinations not already visited by this path.

State tracking:
  spawned   -- walk created at layer 0 from an existing routed edge.
  advanced  -- walk successfully hopped to a new block at the current layer.
  completed -- walk reached walk_length hops.
  failed    -- walk could not advance (no eligible blocks) and was not restarted.
"""
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class Walk:
    source_chunk: int      # original query chunk that benefits from this path
    current_block: int     # key block where information currently resides
    start_layer: int       # layer where this walk was spawned
    hops_completed: int
    walk_type: str = "random"
    visited: List[int] = field(default_factory=list)
    alive: bool = True
    completed: bool = False
    status: str = "spawned"  # spawned, advanced, completed, failed


def random_walk(candidates: torch.Tensor, walk: Walk,
                generator: torch.Generator = None) -> Optional[int]:
    """Pick the next block uniformly at random from eligible candidates."""
    if candidates.numel() == 0:
        return None
    idx = torch.randint(candidates.numel(), (1,), generator=generator).item()
    return candidates[idx].item()


def semantic_walk(candidates: torch.Tensor, walk: Walk,
                  similarity: Optional[torch.Tensor], covering_chunk: int,
                  generator: torch.Generator = None) -> Optional[int]:
    """Pick the next block weighted by semantic similarity to the covering chunk.

    Falls back to random if no similarity is available.
    """
    if candidates.numel() == 0:
        return None
    if similarity is None:
        return random_walk(candidates, walk, generator)
    scores = similarity[covering_chunk, candidates]
    valid = scores > float("-inf")
    if not valid.any():
        return random_walk(candidates, walk, generator)
    probs = torch.softmax(scores[valid].float(), dim=0)
    sel = torch.multinomial(probs, 1, generator=generator).item()
    return candidates[valid][sel].item()


def novelty_walk(candidates: torch.Tensor, walk: Walk,
                 generator: torch.Generator = None) -> Optional[int]:
    """Prefer blocks not already visited by this walk path.

    Falls back to any candidate if all are visited.
    """
    if candidates.numel() == 0:
        return None
    visited_set = set(walk.visited)
    unvisited_mask = torch.tensor(
        [b.item() not in visited_set for b in candidates],
        dtype=torch.bool, device=candidates.device)
    pool = candidates[unvisited_mask] if unvisited_mask.any() else candidates
    idx = torch.randint(pool.numel(), (1,), generator=generator).item()
    return pool[idx].item()


WALK_DISPATCH = {
    "random": random_walk,
    "semantic": semantic_walk,
    "novelty": novelty_walk,
}


def _get_eligible_blocks(edges: torch.Tensor, covering_chunk: int,
                         current_block: int, num_blocks: int) -> torch.Tensor:
    """Get causally valid key blocks that the covering chunk can attend to.

    Returns blocks from the covering chunk's edge list that are:
      - valid (>= 0 and < num_blocks)
      - not the current block (the walk should move, not stay)
      - causally before the covering chunk's start position
    The last constraint is already enforced by the routing pipeline, so we
    only filter on validity and non-self.
    """
    chunk_edges = edges[covering_chunk]
    valid = chunk_edges[(chunk_edges >= 0) & (chunk_edges < num_blocks)]
    return valid[valid != current_block]


def advance_walks(active_walks: list, edges: torch.Tensor, layer_idx: int,
                  chunk_size: int, block_size: int, seq_len: int,
                  walk_config=None, similarity: Optional[torch.Tensor] = None,
                  generator: torch.Generator = None) -> list:
    """Advance each active walk by one hop using the current layer's edges.

    Information-flow model:
      A walk at key block B means information is "at" B.  At the next layer,
      the query chunk that *covers* B's tokens can read B, so information
      flows to whatever new key block that covering chunk attends to.  The
      walk picks one such block as the next hop and reserves an edge slot
      in the covering chunk so the kernel actually computes this attention.

    Args:
        active_walks: list of Walk objects.
        edges: [Q, routed_blocks] block indices for this layer.
        layer_idx: current layer index.
        chunk_size: query chunk size in tokens.
        block_size: key block size in tokens.
        seq_len: total sequence length.
        walk_config: optional WalkConfig for length/probability control.
        similarity: optional [Q, B] similarity scores for semantic_walk.
        generator: optional torch.Generator for reproducibility.

    Returns:
        list of Walk objects: advanced walks (dead/completed removed from active).
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
            w.status = "completed"
            continue

        if cont_prob < 1.0:
            r = torch.rand(1, generator=generator).item()
            if r > cont_prob:
                w.alive = False
                w.status = "failed"
                continue

        # The covering chunk is the query chunk whose token range includes
        # the current block's tokens — this is where information arrives.
        token_pos = w.current_block * block_size
        covering_chunk = min(token_pos // chunk_size, Q - 1)

        eligible = _get_eligible_blocks(edges, covering_chunk,
                                        w.current_block, num_blocks)

        if avoid and eligible.numel() > 0:
            visited_set = set(w.visited)
            unvisited = torch.tensor(
                [b.item() not in visited_set for b in eligible],
                dtype=torch.bool, device=eligible.device)
            if unvisited.any():
                filtered = eligible[unvisited]
            else:
                filtered = eligible
        else:
            filtered = eligible

        next_block = None
        if w.walk_type == "semantic":
            next_block = semantic_walk(filtered, w, similarity,
                                       covering_chunk, generator)
        elif w.walk_type == "novelty":
            next_block = novelty_walk(filtered, w, generator)
        else:
            next_block = random_walk(filtered, w, generator)

        if next_block is None:
            if restart_prob > 0:
                r = torch.rand(1, generator=generator).item()
                if r < restart_prob:
                    chunk_edges = edges[w.source_chunk]
                    start_cands = chunk_edges[
                        (chunk_edges >= 0) & (chunk_edges < num_blocks)]
                    if start_cands.numel() > 0:
                        idx = torch.randint(start_cands.numel(), (1,),
                                            generator=generator).item()
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
                            status="spawned",
                        ))
                    continue
            w.alive = False
            w.status = "failed"
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
            status="advanced",
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
                status="spawned",
            ))
    return walks


def inject_walk_edges(edges: torch.Tensor, active_walks: list,
                      reserved_budget: int, chunk_size: int,
                      block_size: int, seq_len: int) -> torch.Tensor:
    """Reserve edge slots for walk destinations in the correct query chunks.

    For each active walk at key block B, the *covering* query chunk (the one
    whose token range includes B) is where the edge must be injected — that's
    the chunk that needs to attend to B for information to flow.

    Args:
        edges: [Q, total_blocks] block indices.
        active_walks: list of Walk objects.
        reserved_budget: how many edge slots per query chunk to reserve.
        chunk_size: query chunk size in tokens.
        block_size: key block size in tokens.
        seq_len: total sequence length.

    Returns:
        [Q, total_blocks] with walk edges injected in the covering chunks.
    """
    if reserved_budget == 0 or not active_walks:
        return edges
    result = edges.clone()
    Q, total = result.shape
    num_blocks = -(-seq_len // block_size)
    slots_used = {}
    for w in sorted(active_walks, key=lambda w: -w.hops_completed):
        if not w.alive:
            continue
        if w.current_block < 0 or w.current_block >= num_blocks:
            continue
        block_last_token = (w.current_block + 1) * block_size - 1
        # target chunk = first chunk whose start > block_last_token
        target_chunk = block_last_token // chunk_size + 1
        if target_chunk >= Q:
            continue
        used = slots_used.get(target_chunk, 0)
        if used >= reserved_budget:
            continue
        slot = total - reserved_budget + used
        if 0 <= slot < total:
            result[target_chunk, slot] = w.current_block
            slots_used[target_chunk] = used + 1
    return result


def walk_diagnostics(active_walks: list, completed_walks: list = None) -> dict:
    """Compute walk statistics for monitoring.

    Returns:
        Dict with: active_count, completed_count, failed_count,
        spawned_count, advanced_count, mean_path_length, max_path_length,
        unique_blocks_visited, walks_by_type.
    """
    all_walks = list(active_walks)
    if completed_walks:
        all_walks.extend(completed_walks)
    if not all_walks:
        return {
            "active_count": 0,
            "completed_count": 0,
            "failed_count": 0,
            "spawned_count": 0,
            "advanced_count": 0,
            "mean_path_length": 0.0,
            "max_path_length": 0,
            "unique_blocks_visited": 0,
            "walks_by_type": {},
        }
    active = [w for w in all_walks if w.alive]
    completed = [w for w in all_walks if w.completed]
    failed = [w for w in all_walks if w.status == "failed"]
    spawned = [w for w in all_walks if w.status == "spawned"]
    adv = [w for w in all_walks if w.status == "advanced"]
    lengths = [w.hops_completed for w in all_walks]
    all_visited: Set[int] = set()
    by_type: dict = {}
    for w in all_walks:
        all_visited.update(w.visited)
        by_type[w.walk_type] = by_type.get(w.walk_type, 0) + 1
    return {
        "active_count": len(active),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "spawned_count": len(spawned),
        "advanced_count": len(adv),
        "mean_path_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "max_path_length": max(lengths) if lengths else 0,
        "unique_blocks_visited": len(all_visited),
        "walks_by_type": by_type,
    }
