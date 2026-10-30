"""Reversible logical-to-physical token permutations.

A Permutation maps every logical token position (the original chronological
order used for RoPE and causal masking) to a physical position (the order
tokens are stored in the K/V cache for memory locality).

Invariants:
  - logical_to_physical and physical_to_logical are inverses.
  - Every token appears exactly once (full permutation).
  - Logical positions are preserved externally for causal masking and RoPE;
    only the K/V cache storage order changes.
"""
import torch
from dataclasses import dataclass


@dataclass
class Permutation:
    """Reversible mapping between logical and physical token positions.

    Attributes:
        logical_to_physical: [N] int64. physical_pos = logical_to_physical[logical_pos].
        physical_to_logical: [N] int64. logical_pos = physical_to_logical[physical_pos].
    """
    logical_to_physical: torch.Tensor
    physical_to_logical: torch.Tensor

    def __post_init__(self):
        n = self.logical_to_physical.shape[0]
        assert self.physical_to_logical.shape[0] == n
        assert self.logical_to_physical.dtype == torch.long
        assert self.physical_to_logical.dtype == torch.long

    @property
    def n(self) -> int:
        return self.logical_to_physical.shape[0]

    def is_identity(self) -> bool:
        ref = torch.arange(self.n, device=self.logical_to_physical.device)
        return torch.equal(self.logical_to_physical, ref)

    def verify(self) -> bool:
        """Check that the two mappings are mutual inverses and full permutations."""
        n = self.n
        l2p = self.logical_to_physical
        p2l = self.physical_to_logical
        if l2p.min() < 0 or l2p.max() >= n:
            return False
        if p2l.min() < 0 or p2l.max() >= n:
            return False
        if l2p.unique().numel() != n:
            return False
        if p2l.unique().numel() != n:
            return False
        roundtrip_l = p2l[l2p]
        roundtrip_p = l2p[p2l]
        ref = torch.arange(n, device=l2p.device)
        return torch.equal(roundtrip_l, ref) and torch.equal(roundtrip_p, ref)


def identity_permutation(n: int, device: torch.device = None) -> Permutation:
    """Create an identity permutation (no reordering)."""
    idx = torch.arange(n, dtype=torch.long, device=device)
    return Permutation(logical_to_physical=idx, physical_to_logical=idx.clone())


def build_permutation(cluster_assignments: torch.Tensor,
                      cluster_size: int) -> Permutation:
    """Build a permutation that groups co-clustered tokens contiguously.

    Within each cluster, tokens are ordered by their original logical position
    (stable sort). This ensures deterministic, reproducible permutations for
    identical cluster assignments.

    Args:
        cluster_assignments: [N] int64 cluster id per token.
        cluster_size: used for documentation; the actual cluster sizes come
            from the assignments.

    Returns:
        A Permutation mapping logical positions to cluster-grouped physical positions.
    """
    n = cluster_assignments.shape[0]
    device = cluster_assignments.device
    logical_positions = torch.arange(n, dtype=torch.long, device=device)
    sorted_order = torch.argsort(cluster_assignments, stable=True)
    l2p = torch.empty(n, dtype=torch.long, device=device)
    l2p[sorted_order] = logical_positions
    p2l = sorted_order
    return Permutation(logical_to_physical=l2p, physical_to_logical=p2l)
