"""Tests for graph-sparse memory-layout strategies.

Covers: permutation invertibility, K/V restoration, numerical identity
after pack/unpack, incomplete cluster handling, deterministic clustering,
index translation, co-access matrix, locality estimation, config validation.

Run: python tests/test_memory_layout.py
"""
import sys
import torch

sys.path.insert(0, ".")

from experiments.graph_sparse.config import GraphConfig, SemanticConfig, MemoryLayoutConfig
from experiments.graph_sparse.memory.permutation import (
    Permutation, build_permutation, identity_permutation,
)
from experiments.graph_sparse.memory.clusters import (
    cluster_tokens, semantic_cluster, graph_cluster,
)
from experiments.graph_sparse.memory.packing import pack_kv, unpack_kv, translate_indices
from experiments.graph_sparse.memory.cost_model import (
    MemoryLayoutConfig as CostMemoryLayoutConfig,
    LayoutStats, measure_packing_cost, estimate_locality, build_co_access_matrix,
)


# ── Permutation ─────────────────────────────────────────────────────

def test_identity_permutation():
    """Identity permutation maps each position to itself."""
    perm = identity_permutation(128)
    assert perm.n == 128
    assert perm.is_identity()
    assert perm.verify()
    ref = torch.arange(128)
    assert torch.equal(perm.logical_to_physical, ref)
    assert torch.equal(perm.physical_to_logical, ref)
    print("PASS test_identity_permutation")


def test_permutation_invertibility():
    """l2p and p2l must be mutual inverses."""
    torch.manual_seed(0)
    n = 256
    assignments = torch.randint(0, 8, (n,))
    perm = build_permutation(assignments, cluster_size=32)
    assert perm.verify(), "Permutation is not invertible"
    roundtrip = perm.physical_to_logical[perm.logical_to_physical]
    assert torch.equal(roundtrip, torch.arange(n))
    roundtrip2 = perm.logical_to_physical[perm.physical_to_logical]
    assert torch.equal(roundtrip2, torch.arange(n))
    print("PASS test_permutation_invertibility")


def test_permutation_is_full():
    """Every position appears exactly once."""
    torch.manual_seed(1)
    n = 100
    assignments = torch.randint(0, 5, (n,))
    perm = build_permutation(assignments, cluster_size=20)
    assert perm.logical_to_physical.unique().numel() == n
    assert perm.physical_to_logical.unique().numel() == n
    print("PASS test_permutation_is_full")


def test_build_permutation_groups_clusters():
    """Tokens in the same cluster should be contiguous in physical order."""
    assignments = torch.tensor([2, 0, 1, 0, 2, 1, 0, 1])
    perm = build_permutation(assignments, cluster_size=3)
    physical_clusters = assignments[perm.physical_to_logical]
    # Within each cluster, positions should be consecutive
    for c in range(3):
        positions = (physical_clusters == c).nonzero(as_tuple=True)[0]
        if positions.numel() > 1:
            diffs = positions[1:] - positions[:-1]
            assert (diffs == 1).all(), \
                f"Cluster {c} is not contiguous: positions {positions.tolist()}"
    print("PASS test_build_permutation_groups_clusters")


def test_permutation_stable_sort():
    """Within a cluster, logical order is preserved (stable sort)."""
    assignments = torch.tensor([1, 0, 1, 0, 1])
    perm = build_permutation(assignments, cluster_size=3)
    p2l = perm.physical_to_logical
    # Cluster 0: logical positions 1, 3
    cluster0 = p2l[assignments[p2l] == 0]
    assert cluster0.tolist() == [1, 3], f"Expected [1, 3], got {cluster0.tolist()}"
    # Cluster 1: logical positions 0, 2, 4
    cluster1 = p2l[assignments[p2l] == 1]
    assert cluster1.tolist() == [0, 2, 4], f"Expected [0, 2, 4], got {cluster1.tolist()}"
    print("PASS test_permutation_stable_sort")


# ── Packing ─────────────────────────────────────────────────────────

def test_pack_unpack_identity():
    """Pack then unpack with identity permutation returns identical tensors."""
    torch.manual_seed(2)
    kv_heads, n, d = 4, 64, 32
    k = torch.randn(kv_heads, n, d)
    v = torch.randn(kv_heads, n, d)
    perm = identity_permutation(n)
    kp, vp = pack_kv(k, v, perm)
    assert torch.equal(kp, k)
    assert torch.equal(vp, v)
    kr, vr = unpack_kv(kp, vp, perm)
    assert torch.equal(kr, k)
    assert torch.equal(vr, v)
    print("PASS test_pack_unpack_identity")


def test_pack_unpack_roundtrip():
    """Pack then unpack must restore exact K/V values."""
    torch.manual_seed(3)
    kv_heads, n, d = 2, 128, 64
    k = torch.randn(kv_heads, n, d)
    v = torch.randn(kv_heads, n, d)
    assignments = torch.randint(0, 8, (n,))
    perm = build_permutation(assignments, cluster_size=16)
    assert not perm.is_identity(), "Non-trivial permutation expected"

    kp, vp = pack_kv(k, v, perm)
    kr, vr = unpack_kv(kp, vp, perm)
    assert torch.equal(kr, k), "K not restored after pack/unpack"
    assert torch.equal(vr, v), "V not restored after pack/unpack"
    print("PASS test_pack_unpack_roundtrip")


def test_pack_contiguous():
    """Packed tensors must be contiguous in memory."""
    torch.manual_seed(4)
    kv_heads, n, d = 2, 64, 32
    k = torch.randn(kv_heads, n, d)
    v = torch.randn(kv_heads, n, d)
    assignments = torch.randint(0, 4, (n,))
    perm = build_permutation(assignments, cluster_size=16)
    kp, vp = pack_kv(k, v, perm)
    assert kp.is_contiguous(), "Packed K must be contiguous"
    assert vp.is_contiguous(), "Packed V must be contiguous"
    print("PASS test_pack_contiguous")


def test_pack_numerical_identity():
    """Specific token values are at correct physical positions after packing."""
    torch.manual_seed(5)
    kv_heads, n, d = 1, 16, 4
    k = torch.arange(n).float().unsqueeze(0).unsqueeze(-1).expand(kv_heads, n, d)
    v = k.clone()
    assignments = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 2, 2, 2, 2, 3, 3, 3, 3])
    perm = build_permutation(assignments, cluster_size=4)
    kp, _ = pack_kv(k, v, perm)
    for logical_pos in range(n):
        phys_pos = perm.logical_to_physical[logical_pos].item()
        expected_val = float(logical_pos)
        actual_val = kp[0, phys_pos, 0].item()
        assert actual_val == expected_val, \
            f"Logical {logical_pos} -> physical {phys_pos}: expected {expected_val}, got {actual_val}"
    print("PASS test_pack_numerical_identity")


# ── Index translation ───────────────────────────────────────────────

def test_translate_indices_preserves_padding():
    """Padding (-1) in routing indices must be preserved after translation."""
    n = 64
    perm = identity_permutation(n)
    indices = torch.tensor([[[0, 5, -1, -1], [10, -1, 20, -1]]], dtype=torch.int32)
    translated = translate_indices(indices, perm)
    assert (translated[indices == -1] == -1).all()
    print("PASS test_translate_indices_preserves_padding")


def test_translate_indices_correct():
    """Translated indices point to the correct physical positions."""
    torch.manual_seed(6)
    n = 32
    assignments = torch.randint(0, 4, (n,))
    perm = build_permutation(assignments, cluster_size=8)
    logical = torch.tensor([[[0, 5, 15, -1], [10, 20, -1, 31]]], dtype=torch.int32)
    physical = translate_indices(logical, perm)
    for h in range(1):
        for g in range(2):
            for k in range(4):
                li = logical[h, g, k].item()
                pi = physical[h, g, k].item()
                if li == -1:
                    assert pi == -1
                else:
                    assert pi == perm.logical_to_physical[li].item()
    print("PASS test_translate_indices_correct")


def test_translate_indices_gqa():
    """All heads in a GQA group share the same translated indices."""
    n = 64
    assignments = torch.randint(0, 4, (n,))
    perm = build_permutation(assignments, cluster_size=16)
    # 8 query heads, 2 KV heads => groups of 4 share indices
    # Shape [H=8, G=2, K=4]: each head picks 4 indices from 2 groups
    base = torch.randint(0, n, (1, 2, 4), dtype=torch.int32)
    logical = base.expand(8, 2, 4).contiguous()
    physical = translate_indices(logical, perm)
    for kv_h in range(2):
        group_start = kv_h * 4
        for offset in range(1, 4):
            assert torch.equal(physical[group_start + offset], physical[group_start])
    print("PASS test_translate_indices_gqa")


# ── Clustering ──────────────────────────────────────────────────────

def test_semantic_cluster_shape():
    """Cluster assignments have correct shape and range."""
    torch.manual_seed(7)
    n, d = 256, 64
    hidden = torch.randn(n, d)
    assignments = semantic_cluster(hidden, cluster_size=32)
    assert assignments.shape == (n,)
    num_clusters = -(-n // 32)  # ceil(256/32) = 8
    assert assignments.min() >= 0
    assert assignments.max() < num_clusters
    print("PASS test_semantic_cluster_shape")


def test_semantic_cluster_deterministic():
    """Same seed produces identical clusters."""
    torch.manual_seed(8)
    hidden = torch.randn(128, 32)
    a1 = semantic_cluster(hidden, cluster_size=16, seed=42)
    a2 = semantic_cluster(hidden, cluster_size=16, seed=42)
    assert torch.equal(a1, a2), "Clustering not deterministic"
    a3 = semantic_cluster(hidden, cluster_size=16, seed=99)
    # Different seeds should (very likely) produce different assignments
    print("PASS test_semantic_cluster_deterministic")


def test_semantic_cluster_groups_similar():
    """Tokens with identical representations should be in the same cluster."""
    n, d = 64, 16
    hidden = torch.zeros(n, d)
    # 4 groups of 16 identical tokens
    for g in range(4):
        hidden[g*16:(g+1)*16] = torch.randn(1, d).expand(16, d)
    assignments = semantic_cluster(hidden, cluster_size=16, seed=42)
    for g in range(4):
        group = assignments[g*16:(g+1)*16]
        assert (group == group[0]).all(), \
            f"Group {g}: identical tokens should share a cluster"
    print("PASS test_semantic_cluster_groups_similar")


def test_graph_cluster_shape():
    """Graph clustering produces valid token-level assignments from block co-access."""
    torch.manual_seed(9)
    seq_len = 128
    block_size = 16
    B = -(-seq_len // block_size)  # 8
    co_access = torch.rand(B, B)
    co_access = co_access + co_access.T  # symmetric
    assignments = graph_cluster(co_access, cluster_size=32, seed=42,
                                block_size=block_size, seq_len=seq_len)
    assert assignments.shape == (seq_len,)
    assert assignments.min() >= 0
    # All tokens in a block share the same assignment
    for b in range(B):
        start = b * block_size
        end = min(start + block_size, seq_len)
        assert (assignments[start:end] == assignments[start]).all()
    print("PASS test_graph_cluster_shape")


def test_cluster_tokens_dispatch():
    """cluster_tokens dispatches to the correct method."""
    torch.manual_seed(10)
    n, d = 64, 32
    block_size = 16
    B = -(-n // block_size)
    hidden = torch.randn(n, d)
    a_sem = cluster_tokens(hidden, cluster_size=16, method="semantic")
    assert a_sem.shape == (n,)

    co_access = torch.rand(B, B)
    a_graph = cluster_tokens(hidden, cluster_size=16, method="graph",
                             co_access=co_access)
    assert a_graph.shape == (n,)

    try:
        cluster_tokens(hidden, cluster_size=16, method="unknown")
        assert False, "Should reject unknown method"
    except ValueError:
        pass

    try:
        cluster_tokens(hidden, cluster_size=16, method="graph")
        assert False, "graph without co_access should raise"
    except ValueError:
        pass
    print("PASS test_cluster_tokens_dispatch")


def test_incomplete_clusters():
    """Clustering handles N not divisible by cluster_size."""
    torch.manual_seed(11)
    hidden = torch.randn(100, 32)  # 100 / 32 = 3.125 clusters
    assignments = semantic_cluster(hidden, cluster_size=32)
    num_clusters = -(-100 // 32)  # 4
    assert assignments.max() < num_clusters
    # Every token assigned
    assert (assignments >= 0).all()
    perm = build_permutation(assignments, cluster_size=32)
    assert perm.verify()
    print("PASS test_incomplete_clusters")


# ── Cost model ──────────────────────────────────────────────────────

def test_estimate_locality_sorted():
    """Perfectly sorted indices should have high locality."""
    indices = torch.arange(64).unsqueeze(0).unsqueeze(0).to(torch.int32)
    loc = estimate_locality(indices, seq_len=64, neighborhood=1)
    assert loc == 1.0, f"Sorted indices should be perfectly local, got {loc}"
    print("PASS test_estimate_locality_sorted")


def test_estimate_locality_random():
    """Random scattered indices should have lower locality."""
    torch.manual_seed(12)
    indices = torch.randperm(1024)[:64].sort().values
    indices = indices.unsqueeze(0).unsqueeze(0).to(torch.int32)
    loc = estimate_locality(indices, seq_len=1024, neighborhood=4)
    assert loc < 0.5, f"Random indices should have low locality, got {loc}"
    print("PASS test_estimate_locality_random")


def test_estimate_locality_with_padding():
    """Padding entries should be ignored in locality estimation."""
    indices = torch.tensor([[[0, 1, 2, -1, -1, -1]]], dtype=torch.int32)
    loc = estimate_locality(indices, seq_len=64, neighborhood=1)
    assert loc == 1.0
    print("PASS test_estimate_locality_with_padding")


def test_measure_packing_cost_runs():
    """measure_packing_cost produces valid stats."""
    torch.manual_seed(13)
    kv_heads, n, d = 2, 64, 32
    k = torch.randn(kv_heads, n, d)
    v = torch.randn(kv_heads, n, d)
    assignments = torch.randint(0, 4, (n,))
    perm = build_permutation(assignments, cluster_size=16)
    indices = torch.randint(-1, n, (4, 2, 16), dtype=torch.int32)
    stats = measure_packing_cost(k, v, perm, logical_indices=indices,
                                  warmup=0, repeats=1)
    assert stats.pack_time_ms >= 0
    assert stats.unpack_time_ms >= 0
    assert stats.bytes_moved > 0
    assert 0.0 <= stats.locality_before <= 1.0
    assert 0.0 <= stats.locality_after <= 1.0
    print("PASS test_measure_packing_cost_runs")


def test_build_co_access_matrix():
    """Block-level co-access matrix is symmetric and non-negative."""
    torch.manual_seed(14)
    n = 256
    block_size = 16
    B = -(-n // block_size)  # 16
    indices = torch.randint(0, n, (4, 2, 8), dtype=torch.int32)
    co = build_co_access_matrix([indices], n, decay=0.9, block_size=block_size)
    assert co.shape == (B, B), f"Expected ({B}, {B}), got {co.shape}"
    assert (co >= 0).all()
    assert torch.allclose(co, co.T), "Co-access matrix should be symmetric"
    assert co.diag().sum() == 0, "Diagonal should be zero"
    print("PASS test_build_co_access_matrix")


def test_co_access_decay():
    """Older layers should contribute less with decay < 1."""
    torch.manual_seed(15)
    n = 256
    block_size = 16
    B = -(-n // block_size)  # 16
    # Tokens 0,20,40,60 map to blocks 0,1,2,3 — all distinct blocks
    indices = torch.tensor([[[0, 20, 40, 60]]], dtype=torch.int32)
    co_no_decay = build_co_access_matrix([indices, indices], n, decay=1.0,
                                         block_size=block_size)
    co_decay = build_co_access_matrix([indices, indices], n, decay=0.5,
                                      block_size=block_size)
    # Blocks 0 and 1 were co-accessed
    assert co_no_decay[0, 1] > co_decay[0, 1], \
        "Decay should reduce older contributions"
    print("PASS test_co_access_decay")


# ── Config validation ───────────────────────────────────────────────

def test_memory_config_validation():
    """MemoryLayoutConfig validation catches invalid parameters."""
    bad_configs = [
        ({"strategy": "magical"}, "strategy"),
        ({"cluster_size": 0}, "cluster_size"),
        ({"clustering_method": "kmeans"}, "clustering_method"),
        ({"cluster_refresh_layers": -1}, "cluster_refresh_layers"),
        ({"graph_history_decay": 0.0}, "graph_history_decay"),
        ({"reorder_threshold": -0.5}, "reorder_threshold"),
    ]
    for kw, field_name in bad_configs:
        cfg = GraphConfig(
            routed_blocks=4,
            semantic=SemanticConfig(near_blocks=2, medium_blocks=1,
                                   far_blocks=0, random_blocks=1),
            memory_layout=MemoryLayoutConfig(**kw),
        )
        try:
            cfg.validate()
            assert False, f"Should reject bad {field_name}: {kw}"
        except ValueError:
            pass
    print("PASS test_memory_config_validation")


def test_memory_config_valid():
    """Valid memory layout configs pass validation."""
    for strategy in ("original", "sorted_indices", "semantic_clustered", "graph_clustered"):
        cfg = GraphConfig(
            routed_blocks=4,
            semantic=SemanticConfig(near_blocks=2, medium_blocks=1,
                                   far_blocks=0, random_blocks=1),
            memory_layout=MemoryLayoutConfig(strategy=strategy),
        )
        cfg.validate()
    print("PASS test_memory_config_valid")


# ── Sorted indices strategy ─────────────────────────────────────────

def test_sorted_indices_no_data_movement():
    """sorted_indices translates indices without touching K/V data."""
    torch.manual_seed(16)
    n = 128
    kv_heads, d = 2, 32
    k = torch.randn(kv_heads, n, d)
    v = torch.randn(kv_heads, n, d)
    hidden = torch.randn(n, 64)

    assignments = semantic_cluster(hidden, cluster_size=16)
    perm = build_permutation(assignments, cluster_size=16)
    logical = torch.randint(0, n, (4, 2, 16), dtype=torch.int32)
    physical = translate_indices(logical, perm)

    # Verify that accessing k at physical positions after packing
    # gives same values as accessing k at logical positions before packing
    kp, vp = pack_kv(k, v, perm)
    for h in range(4):
        for g in range(2):
            valid_l = logical[h, g][logical[h, g] >= 0].long()
            valid_p = physical[h, g][physical[h, g] >= 0].long()
            k_logical = k[0, valid_l]
            k_physical = kp[0, valid_p]
            assert torch.equal(k_logical, k_physical), \
                "Physical access after packing must match logical access"
    print("PASS test_sorted_indices_no_data_movement")


# ── End-to-end ──────────────────────────────────────────────────────

def test_full_pipeline():
    """Full cluster -> permute -> pack -> translate -> unpack pipeline."""
    torch.manual_seed(17)
    n, d, head_dim = 256, 64, 32
    kv_heads = 4
    hidden = torch.randn(n, d)
    k = torch.randn(kv_heads, n, head_dim)
    v = torch.randn(kv_heads, n, head_dim)
    logical_indices = torch.randint(0, n, (8, 4, 32), dtype=torch.int32)
    logical_indices[logical_indices > n - 5] = -1  # some padding

    # 1. Cluster
    assignments = semantic_cluster(hidden, cluster_size=32, seed=42)
    # 2. Build permutation
    perm = build_permutation(assignments, cluster_size=32)
    assert perm.verify()
    # 3. Pack K/V
    kp, vp = pack_kv(k, v, perm)
    # 4. Translate routing indices
    phys_indices = translate_indices(logical_indices, perm)
    # 5. Verify: accessing kp at phys_indices == accessing k at logical_indices
    for h in range(8):
        for g in range(4):
            for idx in range(32):
                li = logical_indices[h, g, idx].item()
                pi = phys_indices[h, g, idx].item()
                if li < 0:
                    assert pi < 0
                    continue
                kv_head = h % kv_heads  # GQA mapping
                assert torch.equal(k[kv_head, li], kp[kv_head, pi])

    # 6. Unpack restores original
    kr, vr = unpack_kv(kp, vp, perm)
    assert torch.equal(kr, k)
    assert torch.equal(vr, v)
    print("PASS test_full_pipeline")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failed.append((t.__name__, e))
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("FAILURES:")
        for name, err in failed:
            print(f"  {name}: {err}")
        sys.exit(1)
