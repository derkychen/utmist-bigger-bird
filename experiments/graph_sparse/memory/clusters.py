"""Token clustering strategies for memory-layout optimization.

Strategies:
  semantic_cluster  -- group tokens by cosine similarity of hidden states.
  graph_cluster     -- group tokens by co-access frequency across routing layers.
  cluster_tokens    -- dispatcher that calls the appropriate strategy.
"""
import torch
import torch.nn.functional as F


def cluster_tokens(hidden: torch.Tensor, cluster_size: int,
                   method: str = "semantic",
                   co_access: torch.Tensor = None,
                   graph_history_decay: float = 0.9,
                   seed: int = 42) -> torch.Tensor:
    """Assign each token to a cluster.

    Args:
        hidden: [N, D] token representations.
        cluster_size: target tokens per cluster.
        method: "semantic" or "graph".
        co_access: [N, N] co-access count matrix (required for "graph").
        graph_history_decay: exponential decay for older co-access counts.
        seed: for deterministic initialization.

    Returns:
        [N] int64 cluster assignments, values in [0, num_clusters).
    """
    if method == "semantic":
        return semantic_cluster(hidden, cluster_size, seed=seed)
    if method == "graph":
        if co_access is None:
            raise ValueError("graph clustering requires a co_access matrix")
        return graph_cluster(co_access, cluster_size,
                             decay=graph_history_decay, seed=seed)
    raise ValueError(f"Unknown clustering method {method!r}")


def semantic_cluster(hidden: torch.Tensor, cluster_size: int,
                     max_iters: int = 20, seed: int = 42) -> torch.Tensor:
    """K-means clustering on L2-normalized hidden states.

    Simple iterative k-means: initialize centroids from evenly-spaced tokens,
    assign each token to its nearest centroid (cosine similarity), recompute
    centroids as the mean of assigned tokens.

    Args:
        hidden: [N, D] token representations.
        cluster_size: target tokens per cluster.
        max_iters: k-means iterations.
        seed: for centroid initialization.

    Returns:
        [N] int64 cluster assignments.
    """
    n, d = hidden.shape
    num_clusters = max(1, -(-n // cluster_size))
    normed = F.normalize(hidden.float(), dim=-1)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    stride = max(1, n // num_clusters)
    init_idx = torch.arange(0, min(n, num_clusters * stride), stride,
                            device=hidden.device)[:num_clusters]
    if init_idx.numel() < num_clusters:
        extra = torch.randperm(n, generator=gen, device="cpu")[:num_clusters - init_idx.numel()]
        init_idx = torch.cat([init_idx, extra.to(hidden.device)])
    centroids = normed[init_idx].clone()

    assignments = torch.zeros(n, dtype=torch.long, device=hidden.device)
    for _ in range(max_iters):
        sim = normed @ centroids.T
        new_assignments = sim.argmax(dim=1)
        if torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for c in range(num_clusters):
            mask = assignments == c
            if mask.any():
                centroids[c] = F.normalize(normed[mask].mean(0, keepdim=True), dim=-1).squeeze(0)

    return assignments


def graph_cluster(co_access: torch.Tensor, cluster_size: int,
                  decay: float = 0.9, max_iters: int = 20,
                  seed: int = 42) -> torch.Tensor:
    """Cluster tokens by graph co-access affinity.

    Treats the co-access matrix as an affinity graph and runs k-means on the
    row-normalized affinity (each token's co-access profile is its feature vector).

    Args:
        co_access: [N, N] float co-access counts. co_access[i, j] = how often
            tokens i and j were routed to the same query chunk.
        cluster_size: target tokens per cluster.
        decay: exponential decay applied to the co-access matrix before clustering
            (co_access *= decay). Set to 1.0 to disable decay.
        max_iters: k-means iterations.
        seed: for centroid initialization.

    Returns:
        [N] int64 cluster assignments.
    """
    n = co_access.shape[0]
    num_clusters = max(1, -(-n // cluster_size))
    aff = co_access.float()
    if decay < 1.0:
        aff = aff * decay
    row_norm = aff.norm(dim=1, keepdim=True).clamp(min=1e-8)
    features = aff / row_norm

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    stride = max(1, n // num_clusters)
    init_idx = torch.arange(0, min(n, num_clusters * stride), stride,
                            device=co_access.device)[:num_clusters]
    if init_idx.numel() < num_clusters:
        extra = torch.randperm(n, generator=gen, device="cpu")[:num_clusters - init_idx.numel()]
        init_idx = torch.cat([init_idx, extra.to(co_access.device)])
    centroids = features[init_idx].clone()

    assignments = torch.zeros(n, dtype=torch.long, device=co_access.device)
    for _ in range(max_iters):
        sim = features @ centroids.T
        new_assignments = sim.argmax(dim=1)
        if torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for c in range(num_clusters):
            mask = assignments == c
            if mask.any():
                cent = features[mask].mean(0, keepdim=True)
                cent_norm = cent.norm(dim=1, keepdim=True).clamp(min=1e-8)
                centroids[c] = (cent / cent_norm).squeeze(0)

    return assignments
