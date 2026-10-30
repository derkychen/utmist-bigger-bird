"""Physical K/V cache packing and unpacking.

pack_kv:   reorder K/V from logical to physical order using a Permutation.
unpack_kv: restore K/V from physical back to logical order.
translate_indices: convert routing indices from logical to physical positions.

These operate on actual tensor data (not views) to ensure physical memory
locality. A tensor view or index_select does not change the underlying
memory layout — only a contiguous copy does.
"""
import torch
from .permutation import Permutation


def pack_kv(k: torch.Tensor, v: torch.Tensor,
            perm: Permutation) -> tuple:
    """Reorder K and V from logical to physical token order.

    Args:
        k: [num_kv_heads, N, head_dim] key cache.
        v: [num_kv_heads, N, head_dim] value cache.
        perm: the permutation mapping.

    Returns:
        (k_packed, v_packed): both [num_kv_heads, N, head_dim], contiguous,
            with tokens in physical order.
    """
    p2l = perm.physical_to_logical
    k_packed = k[:, p2l, :].contiguous()
    v_packed = v[:, p2l, :].contiguous()
    return k_packed, v_packed


def unpack_kv(k_packed: torch.Tensor, v_packed: torch.Tensor,
              perm: Permutation) -> tuple:
    """Restore K and V from physical order back to logical (chronological) order.

    Args:
        k_packed: [num_kv_heads, N, head_dim] in physical order.
        v_packed: [num_kv_heads, N, head_dim] in physical order.
        perm: the permutation mapping.

    Returns:
        (k, v): both [num_kv_heads, N, head_dim], contiguous, in logical order.
    """
    l2p = perm.logical_to_physical
    k = k_packed[:, l2p, :].contiguous()
    v = v_packed[:, l2p, :].contiguous()
    return k, v


def translate_indices(logical_indices: torch.Tensor,
                      perm: Permutation) -> torch.Tensor:
    """Convert routing indices from logical positions to physical positions.

    Args:
        logical_indices: [H, G, K] int32 token positions in logical order.
            -1 entries (padding) are preserved.
        perm: the permutation mapping.

    Returns:
        [H, G, K] int32 physical positions, with -1 padding preserved.
    """
    mask = logical_indices >= 0
    result = logical_indices.clone()
    if mask.any():
        l2p = perm.logical_to_physical
        valid_logical = logical_indices[mask].long()
        result[mask] = l2p[valid_logical].to(logical_indices.dtype)
    return result
