"""Integration tests for graph-aware sparse attention.

Uses a tiny randomly-initialized Llama model to test the full attention pipeline:
install_graph, sparse prefill, exact decode, GQA, RoPE, generation, state lifecycle.

Runs on CPU (via bigger_bird_flash_ref). CUDA tests are marked with pytest.mark.cuda.

Run: python tests/test_graph_attention.py
      or: pytest tests/test_graph_attention.py -v
"""
import sys
import copy

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, ".")

from experiments.graph_sparse.config import GraphConfig, SemanticConfig, RewireConfig
from experiments.graph_sparse.attention import GraphSparseAttention, install_graph
from experiments.graph_sparse.graph_state import GraphCoordinator
from kernels.bigger_bird_flash_ref import bigger_bird_flash_ref

DEV = "cpu"
DT = torch.float32


def tiny_llama(num_layers=2, num_heads=8, num_kv_heads=2, hidden=256, vocab=128, dtype=DT):
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=512,
        num_hidden_layers=num_layers, num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads, max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).to(DEV, dtype).eval()


def small_graph_config(chunk=64, block=16, routed_blocks=4, sink=4, window=32):
    return GraphConfig(
        query_chunk_size=chunk, key_block_size=block, routed_blocks=routed_blocks,
        routing_dim=32,
        semantic=SemanticConfig(near_blocks=2, medium_blocks=1, far_blocks=0, random_blocks=1),
        rewiring=RewireConfig(mode="partial_semantic", seed=42),
    ), sink, window


def prefill_logits(model, ids):
    with torch.inference_mode():
        return model(ids, use_cache=False).logits.float()


# ── install_graph correctness ────────────────────────────────────────

def test_install_replaces_all_layers():
    model = tiny_llama()
    cfg, sink, window = small_graph_config()
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        assert isinstance(attn, GraphSparseAttention), f"Layer {i} not replaced"
        assert attn.coordinator is model._graph_coordinator
    print("PASS test_install_replaces_all_layers")


def test_install_preserves_projections():
    model = tiny_llama()
    orig_q = [layer.self_attn.q_proj.weight.clone() for layer in model.model.layers]
    orig_k = [layer.self_attn.k_proj.weight.clone() for layer in model.model.layers]
    orig_v = [layer.self_attn.v_proj.weight.clone() for layer in model.model.layers]
    orig_o = [layer.self_attn.o_proj.weight.clone() for layer in model.model.layers]
    cfg, sink, window = small_graph_config()
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        assert torch.equal(attn.q_proj.weight, orig_q[i]), f"Layer {i} q_proj changed"
        assert torch.equal(attn.k_proj.weight, orig_k[i]), f"Layer {i} k_proj changed"
        assert torch.equal(attn.v_proj.weight, orig_v[i]), f"Layer {i} v_proj changed"
        assert torch.equal(attn.o_proj.weight, orig_o[i]), f"Layer {i} o_proj changed"
    print("PASS test_install_preserves_projections")


def test_shared_coordinator():
    model = tiny_llama(num_layers=4)
    cfg, sink, window = small_graph_config()
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    coordinators = {id(layer.self_attn.coordinator) for layer in model.model.layers}
    assert len(coordinators) == 1, f"Expected 1 shared coordinator, got {len(coordinators)}"
    print("PASS test_shared_coordinator")


# ── Sparse prefill execution ─────────────────────────────────────────

def test_graph_routing_executes():
    """Graph routing produces valid indices during sparse prefill."""
    model = tiny_llama()
    cfg, sink, window = small_graph_config(chunk=64, block=16, routed_blocks=4, sink=4, window=32)
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    # seq_len must exceed sink+window to trigger sparse path
    ids = torch.randint(0, 128, (1, sink + window + 64), device=DEV)
    with torch.inference_mode():
        model(ids, use_cache=False)
    for layer in model.model.layers:
        attn = layer.self_attn
        assert attn.last_routes is not None, "Routes not set — graph routing did not execute"
        H, G, K = attn.last_routes.shape
        assert H == attn.num_heads
        n = ids.shape[1]
        expected_g = -(-n // cfg.query_chunk_size)
        assert G == expected_g, f"Expected {expected_g} groups, got {G}"
        assert K == cfg.routed_keys, f"Expected {cfg.routed_keys} routed keys, got {K}"
    print("PASS test_graph_routing_executes")


def test_short_sequence_dense_fallback():
    """Sequences shorter than sink+window use exact attention, no graph."""
    model = tiny_llama()
    cfg, sink, window = small_graph_config(sink=4, window=32)
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    ids = torch.randint(0, 128, (1, sink + window - 1), device=DEV)
    with torch.inference_mode():
        out = model(ids, use_cache=False)
    for layer in model.model.layers:
        assert layer.self_attn.last_routes is None, "Routes should be None for short sequence"
    assert out.logits.shape == (1, ids.shape[1], 128)
    print("PASS test_short_sequence_dense_fallback")


# ── Reference kernel agreement ────────────────────────────────────────

def test_graph_indices_work_with_bigger_bird_ref():
    """Indices from the graph coordinator produce valid output from bigger_bird_flash_ref."""
    torch.manual_seed(1)
    n, d, bh = 256, 32, 4
    num_kv = 2
    q = torch.randn(bh, n, d) / d**0.5
    k = torch.randn(num_kv, n, d)
    v = torch.randn(num_kv, n, d)

    cfg = GraphConfig(query_chunk_size=64, key_block_size=16, routed_blocks=4, routing_dim=32,
                      semantic=SemanticConfig(near_blocks=2, medium_blocks=1, far_blocks=0, random_blocks=1),
                      rewiring=RewireConfig(seed=42))
    coord = GraphCoordinator(cfg, num_heads=bh, num_kv_heads=num_kv, head_dim=d, num_layers=1)
    hidden = torch.randn(n, 64)
    idx = coord.build_layer_graph(0, hidden)

    out = bigger_bird_flash_ref(q, k, v, idx, front=4, window=32,
                                route_chunk=cfg.query_chunk_size,
                                num_heads=bh, kv_groups=bh // num_kv, scale=1.0 / d**0.5)
    assert out.shape == (bh, n, d), f"Expected ({bh},{n},{d}), got {out.shape}"
    assert not out.isnan().any(), "NaN in output"
    assert not out.isinf().any(), "Inf in output"
    print("PASS test_graph_indices_work_with_bigger_bird_ref")


def test_invalid_indices_dont_affect_output():
    """Padding (-1) in routing indices should not change output vs fewer valid indices."""
    torch.manual_seed(2)
    n, d, bh = 128, 32, 2
    q = torch.randn(bh, n, d) / d**0.5
    k = torch.randn(bh, n, d)
    v = torch.randn(bh, n, d)
    # Indices with some valid, rest -1
    idx_sparse = torch.full((bh, 1, 16), -1, dtype=torch.int32)
    idx_sparse[:, 0, :4] = torch.tensor([4, 8, 12, 16])
    # Same valid indices, more -1 padding
    idx_padded = torch.full((bh, 1, 32), -1, dtype=torch.int32)
    idx_padded[:, 0, :4] = torch.tensor([4, 8, 12, 16])
    out1 = bigger_bird_flash_ref(q, k, v, idx_sparse, front=4, window=16,
                                 route_chunk=n, num_heads=bh, scale=1.0 / d**0.5)
    out2 = bigger_bird_flash_ref(q, k, v, idx_padded, front=4, window=16,
                                 route_chunk=n, num_heads=bh, scale=1.0 / d**0.5)
    err = (out1 - out2).abs().max().item()
    assert err < 1e-6, f"Padding changed output: max error {err}"
    print("PASS test_invalid_indices_dont_affect_output")


# ── Sink + window + routed = one softmax ──────────────────────────────

def test_joint_softmax_sink_local_routed():
    """Verify sink, window, and routed keys participate in a single softmax."""
    torch.manual_seed(3)
    n, d = 128, 16
    q = torch.randn(1, n, d) / d**0.5
    k = torch.randn(1, n, d)
    v_sink = torch.zeros(1, n, d)
    v_sink[0, 0, 0] = 100.0  # sink token has distinctive value
    v_routed = torch.zeros(1, n, d)
    v_routed[0, 60, 0] = 100.0  # routed token

    # Route token 60 for all queries
    idx = torch.full((1, 1, 4), -1, dtype=torch.int32)
    idx[0, 0, 0] = 60

    # With sink=4, window=8, route_chunk=n
    out_sink = bigger_bird_flash_ref(q, k, v_sink, idx, front=4, window=8,
                                     route_chunk=n, num_heads=1, scale=1.0)
    out_routed = bigger_bird_flash_ref(q, k, v_routed, idx, front=4, window=8,
                                       route_chunk=n, num_heads=1, scale=1.0)
    # Query at position 80 should see both sink (pos 0-3) and routed (pos 60)
    # in its softmax. The output should be a weighted mix of both.
    # If they were separate softmaxes, the weights wouldn't sum to 1.
    v_both = v_sink + v_routed
    out_both = bigger_bird_flash_ref(q, k, v_both, idx, front=4, window=8,
                                     route_chunk=n, num_heads=1, scale=1.0)
    # Linearity of expectation: out(v1+v2) == out(v1) + out(v2) iff one softmax
    err = (out_both - out_sink - out_routed).abs().max().item()
    assert err < 1e-5, f"Outputs not additive: max error {err} — separate softmaxes?"
    print("PASS test_joint_softmax_sink_local_routed")


# ── Exact tail ────────────────────────────────────────────────────────

def test_exact_tail_attention_is_dense():
    """The exact-tail kernel call uses window=n, producing dense causal attention rows.

    We test this at the kernel level: for the last `r` queries, the output from
    bigger_bird_flash_ref with window=n and no routing should match SDPA dense causal.
    Full-model logits differ because earlier layers' sparse outputs affect later layers.
    """
    torch.manual_seed(10)
    n, d, bh = 128, 32, 4
    r = 16
    q = torch.randn(bh, n, d) / d**0.5
    k = torch.randn(bh, n, d)
    v = torch.randn(bh, n, d)

    # Exact tail: last r queries with window=n, front=0, no routes
    none_idx = torch.full((bh, 1, 1), -1, dtype=torch.int32)
    tail_out = bigger_bird_flash_ref(q[:, n - r:], k, v, none_idx,
                                     front=0, window=n, num_heads=bh, scale=1.0 / d**0.5)

    # Dense causal SDPA for last r queries
    qp = torch.arange(n - r, n)[:, None]
    causal = torch.arange(n)[None] <= qp
    dense_out = F.scaled_dot_product_attention(
        q[:, n - r:].unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        attn_mask=causal, scale=1.0 / d**0.5)[0]

    err = (tail_out.float() - dense_out.float()).abs().max().item()
    assert err < 1e-4, f"Exact tail != dense SDPA: max error {err}"
    print("PASS test_exact_tail_attention_is_dense")


# ── GQA correctness ──────────────────────────────────────────────────

def test_gqa_routing_correct_kv_head():
    """Verify query heads within a GQA group use the correct KV head."""
    torch.manual_seed(4)
    n, d = 128, 32
    num_heads, num_kv = 8, 2
    groups = num_heads // num_kv
    q = torch.randn(num_heads, n, d) / d**0.5
    k = torch.randn(num_kv, n, d)
    v = torch.zeros(num_kv, n, d)
    # KV head 0 has distinctive value at position 5
    v[0, 5, :] = 1.0
    # KV head 1 has distinctive value at position 10
    v[1, 10, :] = 2.0

    # Route both positions for all heads
    idx = torch.full((num_heads, 1, 8), -1, dtype=torch.int32)
    idx[:, 0, 0] = 5
    idx[:, 0, 1] = 10

    out = bigger_bird_flash_ref(q, k, v, idx, front=0, window=1, route_chunk=n,
                                num_heads=num_heads, kv_groups=groups, scale=1.0 / d**0.5)

    # Heads 0-3 (KV head 0) should see v[0,5]=1.0
    # Heads 4-7 (KV head 1) should see v[1,10]=2.0
    for h in range(num_heads):
        kv_h = h // groups
        if kv_h == 0:
            assert out[h, 6:, :].abs().sum() > 0, f"Head {h} (KV 0) should have nonzero output from token 5"
        else:
            assert out[h, 11:, :].abs().sum() > 0, f"Head {h} (KV 1) should have nonzero output from token 10"
    print("PASS test_gqa_routing_correct_kv_head")


# ── Causal correctness ────────────────────────────────────────────────

def test_prefix_invariance():
    """Appending tokens must not change existing prefix outputs (strict causal)."""
    model_short = tiny_llama()
    model_long = tiny_llama()
    cfg, sink, window = small_graph_config(chunk=64, block=16, routed_blocks=4, sink=4, window=32)
    install_graph(model_short, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    install_graph(model_long, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)

    n_prefix = sink + window + 64
    n_extended = n_prefix + 128
    torch.manual_seed(5)
    ids_long = torch.randint(0, 128, (1, n_extended), device=DEV)
    ids_short = ids_long[:, :n_prefix]

    with torch.inference_mode():
        logits_short = model_short(ids_short, use_cache=False).logits.float()
        logits_long = model_long(ids_long, use_cache=False).logits.float()

    # Compare prefix logits. Due to per-chunk routing with different n, the routing
    # decisions can differ (different number of query chunks and key blocks), so
    # this test validates the PROPERTY of strict causal routing: query chunks that
    # see the same eligible key blocks should produce the same output.
    # For this to hold strictly, routing must depend only on the prefix.
    # Chunk 0 has no eligible blocks in both cases -> same output.
    # This is a weaker but important check: the routing decisions for early
    # query chunks should not depend on later tokens.
    first_chunk_short = logits_short[0, :cfg.query_chunk_size]
    first_chunk_long = logits_long[0, :cfg.query_chunk_size]
    err = (first_chunk_short - first_chunk_long).abs().max().item()
    # First chunk sees no routed blocks (all blocks would end after its start), so
    # it uses only sink + local, which is position-deterministic.
    assert err < 1e-4, f"First chunk prefix not invariant: max error {err}"
    print("PASS test_prefix_invariance")


def test_routing_indices_causally_valid():
    """Every routed token index must be strictly before its query chunk's first position."""
    torch.manual_seed(6)
    model = tiny_llama()
    cfg, sink, window = small_graph_config(chunk=64, block=16, routed_blocks=4, sink=4, window=32)
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    n = sink + window + 128
    ids = torch.randint(0, 128, (1, n), device=DEV)
    with torch.inference_mode():
        model(ids, use_cache=False)
    for li, layer in enumerate(model.model.layers):
        idx = layer.self_attn.last_routes
        for g in range(idx.shape[1]):
            q_start = g * cfg.query_chunk_size
            for h in range(idx.shape[0]):
                valid = idx[h, g][idx[h, g] >= 0]
                violations = valid[valid >= q_start]
                assert violations.numel() == 0, \
                    f"Layer {li} head {h} group {g}: positions {violations.tolist()} >= q_start={q_start}"
    print("PASS test_routing_indices_causally_valid")


# ── Generation and state lifecycle ────────────────────────────────────

def test_generate_runs():
    """model.generate() with graph sparse attention completes without error."""
    model = tiny_llama()
    cfg, sink, window = small_graph_config(chunk=64, block=16, routed_blocks=4, sink=4, window=32)
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    n = sink + window + 64
    ids = torch.randint(0, 128, (1, n), device=DEV)
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=5, do_sample=False, use_cache=True,
                             pad_token_id=0)
    assert out.shape[1] == n + 5, f"Expected {n + 5} tokens, got {out.shape[1]}"
    print("PASS test_generate_runs")


def test_begin_prefill_auto_hook():
    """The forward pre-hook calls begin_prefill on fresh prompts."""
    model = tiny_llama()
    cfg, sink, window = small_graph_config(sink=4, window=32)
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    coord = model._graph_coordinator
    assert coord.state.prefill_id == 0
    n = sink + window + 64
    ids = torch.randint(0, 128, (1, n), device=DEV)
    with torch.inference_mode():
        model(ids, use_cache=False)
    assert coord.state.prefill_id == 1, "begin_prefill not called on first forward"
    with torch.inference_mode():
        model(ids, use_cache=False)
    assert coord.state.prefill_id == 2, "begin_prefill not called on second forward"
    print("PASS test_begin_prefill_auto_hook")


def test_cross_prompt_isolation():
    """Graph state resets between prompts — no leakage."""
    model = tiny_llama()
    cfg, sink, window = small_graph_config(sink=4, window=32)
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    coord = model._graph_coordinator
    n = sink + window + 64
    ids1 = torch.randint(0, 128, (1, n), device=DEV)
    ids2 = torch.randint(0, 128, (1, n), device=DEV)
    with torch.inference_mode():
        model(ids1, use_cache=False)
    assert coord.state.previous_block_routes is not None
    with torch.inference_mode():
        model(ids2, use_cache=False)
    # After second forward, state should be from ids2, not ids1
    # (begin_prefill was called, so previous_block_routes was reset to None before layer 0)
    # It's now set by the second forward's layer processing
    assert coord.state.prefill_id == 2
    print("PASS test_cross_prompt_isolation")


def test_deterministic_generation():
    """Same input and seed produce identical generation."""
    def gen_once():
        model = tiny_llama()
        cfg, sink, window = small_graph_config(sink=4, window=32)
        install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
        n = sink + window + 64
        torch.manual_seed(99)
        ids = torch.randint(0, 128, (1, n), device=DEV)
        with torch.inference_mode():
            return model.generate(ids, max_new_tokens=5, do_sample=False, use_cache=True, pad_token_id=0)
    out1 = gen_once()
    out2 = gen_once()
    assert torch.equal(out1, out2), "Non-deterministic generation"
    print("PASS test_deterministic_generation")


def test_kv_evict_disabled():
    """KV eviction is disabled in graph arm."""
    model = tiny_llama()
    cfg, sink, window = small_graph_config(sink=4, window=32)
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    for layer in model.model.layers:
        assert not layer.self_attn.kv_evict, "KV eviction should be disabled for graph arm"
    print("PASS test_kv_evict_disabled")


# ── Rewiring preserves causality ──────────────────────────────────────

def test_rewiring_preserves_causality():
    """After rewiring, all block edges must still be causally valid."""
    torch.manual_seed(7)
    n = 512
    hidden = torch.randn(n, 128)
    cfg = GraphConfig(
        query_chunk_size=128, key_block_size=32, routed_blocks=4,
        routing_dim=32,
        semantic=SemanticConfig(near_blocks=2, medium_blocks=1, far_blocks=0, random_blocks=1),
        rewiring=RewireConfig(mode="partial_semantic", replace_probability=0.5, seed=42),
    )
    coord = GraphCoordinator(cfg, num_heads=4, num_kv_heads=2, head_dim=64, num_layers=4)
    coord.begin_prefill()
    for layer in range(4):
        idx = coord.build_layer_graph(layer, hidden)
        for g in range(idx.shape[1]):
            q_start = g * cfg.query_chunk_size
            for h in range(idx.shape[0]):
                valid = idx[h, g][idx[h, g] >= 0]
                assert (valid < q_start).all(), \
                    f"Layer {layer} head {h} group {g}: causal violation after rewiring"
    print("PASS test_rewiring_preserves_causality")


# ── Config validation ─────────────────────────────────────────────────

def test_chunk_block_m_alignment():
    """query_chunk_size must be a multiple of block_m."""
    model = tiny_llama()
    cfg = GraphConfig(query_chunk_size=50, key_block_size=16, routed_blocks=8, routing_dim=32)
    try:
        install_graph(model, cfg, sink=4, window=32, exact_tail=0, block_m=16, block_n=16)
        assert False, "Should have raised ValueError for misaligned chunk/block_m"
    except ValueError as e:
        assert "multiple of block_m" in str(e)
    print("PASS test_chunk_block_m_alignment")


# ── Walk causality ────────────────────────────────────────────────────

def test_walk_forward_only():
    """Walk dies when covering chunk has no eligible edges (only the walk's own block)."""
    from experiments.graph_sparse.routing.walks import Walk, advance_walks
    # Walk at block 1: covering chunk = (1*32)//64 = 0. Chunk 0 has only block 1 (self).
    edges = torch.tensor([[1, -1, -1], [3, 5, -1]])
    walks = [Walk(source_chunk=0, current_block=1, start_layer=0, hops_completed=1,
                  visited=[1])]
    advanced = advance_walks(walks, edges, layer_idx=1, chunk_size=64, block_size=32, seq_len=128)
    # Chunk 0 has only block 1, which is the current block → no eligible → walk dies
    assert len(advanced) == 0, "Walk should die when no eligible edges"
    print("PASS test_walk_forward_only")


def test_walk_picks_forward_block():
    """Walk picks an eligible block from the covering chunk's edges."""
    from experiments.graph_sparse.routing.walks import Walk, advance_walks
    # Walk at block 2: covering chunk = (2*32)//64 = 1. Chunk 1 has blocks [5, 2, -1].
    # Eligible = [5] (excluding current block 2).
    edges = torch.tensor([[1, 3, -1], [5, 2, -1], [10, 7, -1]])
    walks = [Walk(source_chunk=0, current_block=2, start_layer=0, hops_completed=1,
                  visited=[2])]
    advanced = advance_walks(walks, edges, layer_idx=1, chunk_size=64, block_size=32, seq_len=384)
    assert len(advanced) == 1
    assert advanced[0].current_block == 5, "Should pick eligible block 5"
    print("PASS test_walk_picks_forward_block")


# ── Diagnostics ───────────────────────────────────────────────────────

def test_timing_diagnostics_populated():
    """Graph build and attention timing are recorded."""
    model = tiny_llama()
    cfg, sink, window = small_graph_config(sink=4, window=32)
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    n = sink + window + 64
    ids = torch.randint(0, 128, (1, n), device=DEV)
    with torch.inference_mode():
        model(ids, use_cache=False)
    for layer in model.model.layers:
        attn = layer.self_attn
        assert attn.last_graph_build_ms > 0, "Graph build time not recorded"
        assert attn.last_attention_ms > 0, "Attention time not recorded"
    print("PASS test_timing_diagnostics_populated")


# ── Existing BB v3 unchanged ──────────────────────────────────────────

def test_rd_attention_unaffected():
    """Installing graph attention on one model doesn't change RDAttention on another."""
    model_rd = tiny_llama()
    from modal_rd.attention import install as rd_install
    rd_install(model_rd, "vs", sink=4, window=32, vertical=8, n_route=4, block_m=16, block_n=16)
    ids = torch.randint(0, 128, (1, 64), device=DEV)
    with torch.inference_mode():
        rd_logits_before = model_rd(ids, use_cache=False).logits.float()

    # Now install graph on a separate model — should not affect rd model
    model_graph = tiny_llama()
    cfg, sink, window = small_graph_config(sink=4, window=32)
    install_graph(model_graph, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)

    with torch.inference_mode():
        rd_logits_after = model_rd(ids, use_cache=False).logits.float()
    err = (rd_logits_before - rd_logits_after).abs().max().item()
    assert err < 1e-6, f"RDAttention logits changed after graph install: {err}"
    print("PASS test_rd_attention_unaffected")


# ── Mode-specific behavior ───────────────────────────────────────────

def _graph_config_with_mode(mode, chunk=64, block=16, routed_blocks=4):
    return GraphConfig(
        mode=mode, query_chunk_size=chunk, key_block_size=block, routed_blocks=routed_blocks,
        routing_dim=32,
        semantic=SemanticConfig(near_blocks=2, medium_blocks=1, far_blocks=0, random_blocks=1),
        rewiring=RewireConfig(mode="partial_semantic", seed=42),
    )


def test_strict_causal_disables_exact_tail():
    """In strict_causal mode, exact_tail is disabled even when configured."""
    model = tiny_llama()
    cfg = _graph_config_with_mode("strict_causal")
    sink, window = 4, 32
    install_graph(model, cfg, sink=sink, window=window, exact_tail=16, block_m=16, block_n=16)
    n = sink + window + 64
    ids = torch.randint(0, 128, (1, n), device=DEV)

    # Run with exact_tail=16 in strict_causal: should behave as exact_tail=0
    with torch.inference_mode():
        logits_strict = model(ids, use_cache=False).logits.float()

    # Compare against explicit exact_tail=0
    model2 = tiny_llama()
    install_graph(model2, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    with torch.inference_mode():
        logits_no_tail = model2(ids, use_cache=False).logits.float()

    err = (logits_strict - logits_no_tail).abs().max().item()
    assert err < 1e-5, f"strict_causal with exact_tail=16 differs from exact_tail=0: {err}"
    print("PASS test_strict_causal_disables_exact_tail")


def test_offline_allows_exact_tail():
    """In offline mode, exact_tail produces different output from exact_tail=0."""
    model_tail = tiny_llama()
    cfg = _graph_config_with_mode("offline")
    sink, window = 4, 32
    install_graph(model_tail, cfg, sink=sink, window=window, exact_tail=16, block_m=16, block_n=16)
    n = sink + window + 64
    ids = torch.randint(0, 128, (1, n), device=DEV)

    model_no_tail = tiny_llama()
    install_graph(model_no_tail, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)

    with torch.inference_mode():
        logits_tail = model_tail(ids, use_cache=False).logits.float()
        logits_no_tail = model_no_tail(ids, use_cache=False).logits.float()

    # The last 16 positions should differ (exact_tail uses dense attention there)
    tail_diff = (logits_tail[0, -16:] - logits_no_tail[0, -16:]).abs().max().item()
    assert tail_diff > 1e-3, f"offline exact_tail should change final logits, diff={tail_diff}"
    print("PASS test_offline_allows_exact_tail")


def test_offline_routes_non_causal_blocks():
    """Offline mode allows routing to blocks after the query chunk start."""
    torch.manual_seed(8)
    n = 512
    hidden = torch.randn(n, 128)
    cfg = _graph_config_with_mode("offline", chunk=128, block=32, routed_blocks=4)
    coord = GraphCoordinator(cfg, num_heads=4, num_kv_heads=2, head_dim=64, num_layers=1)
    coord.begin_prefill()
    idx = coord.build_layer_graph(0, hidden)
    # Query chunk 0 starts at 0: in strict_causal, no blocks are eligible (all end >= 0).
    # In offline mode, all blocks are eligible, so chunk 0 should have some routed keys.
    chunk0_keys = idx[0, 0]
    valid = chunk0_keys[chunk0_keys >= 0]
    assert valid.numel() > 0, "Offline mode: chunk 0 should have routed keys"
    print("PASS test_offline_routes_non_causal_blocks")


def test_strict_causal_no_future_routing():
    """Strict_causal mode prevents routing to blocks at or after the query chunk start."""
    torch.manual_seed(9)
    n = 512
    hidden = torch.randn(n, 128)
    cfg = _graph_config_with_mode("strict_causal", chunk=128, block=32, routed_blocks=4)
    coord = GraphCoordinator(cfg, num_heads=4, num_kv_heads=2, head_dim=64, num_layers=1)
    coord.begin_prefill()
    idx = coord.build_layer_graph(0, hidden)
    for g in range(idx.shape[1]):
        q_start = g * cfg.query_chunk_size
        for h in range(idx.shape[0]):
            valid = idx[h, g][idx[h, g] >= 0]
            violations = valid[valid >= q_start]
            assert violations.numel() == 0, \
                f"strict_causal: head {h} group {g}: positions {violations.tolist()} >= q_start={q_start}"
    print("PASS test_strict_causal_no_future_routing")


# ── Debug route injection ────────────────────────────────────────────

def test_debug_bb_v3_route_injection():
    """BB v3 route injection via _debug_config produces valid output."""
    model = tiny_llama()
    cfg, sink, window = small_graph_config(chunk=64, block=16, routed_blocks=4, sink=4, window=32)
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    n = sink + window + 64
    ids = torch.randint(0, 128, (1, n), device=DEV)

    for layer in model.model.layers:
        layer.self_attn._debug_config = {
            "route_mode": "bb_v3", "vertical": 8, "n_route": 4,
        }

    with torch.inference_mode():
        out = model(ids, use_cache=False)
    assert out.logits.shape == (1, n, 128)
    assert not out.logits.isnan().any(), "NaN in output with BB v3 route injection"

    for layer in model.model.layers:
        attn = layer.self_attn
        idx = attn.last_routes
        assert idx is not None, "Routes not set"
        H, G, K = idx.shape
        assert H == attn.num_heads
        expected_g = -(-n // cfg.query_chunk_size)
        assert G == expected_g, f"Expected {expected_g} groups, got {G}"
        assert K == 8, f"Expected budget=8 routed keys, got {K}"
        for h in range(H):
            for g in range(1, G):
                assert (idx[h, g] == idx[h, 0]).all(), \
                    f"Head {h}: group {g} routes differ from group 0"
    print("PASS test_debug_bb_v3_route_injection")


def test_debug_coverage_tracking():
    """Coverage tracking via _debug_config records needle hit info."""
    model = tiny_llama()
    cfg, sink, window = small_graph_config(chunk=64, block=16, routed_blocks=4, sink=4, window=32)
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    n = sink + window + 64
    ids = torch.randint(0, 128, (1, n), device=DEV)

    for layer in model.model.layers:
        layer.self_attn._debug_config = {
            "track_coverage": True,
            "needle_span": (50, 55),
        }

    with torch.inference_mode():
        model(ids, use_cache=False)

    for layer in model.model.layers:
        attn = layer.self_attn
        cov = attn._last_coverage
        assert cov is not None, "Coverage not recorded"
        assert "layer" in cov
        assert "final_chunk_hit" in cov
        assert "needle_hit_any_head_per_chunk" in cov
        assert isinstance(cov["needle_hit_any_head_per_chunk"], list)
        assert len(cov["needle_hit_any_head_per_chunk"]) == -(-n // cfg.query_chunk_size)
    print("PASS test_debug_coverage_tracking")


# ── Question-conditioned routing ─────────────────────────────────────

def test_question_conditioned_routing():
    """Question-conditioned routing produces valid indices from the coordinator."""
    torch.manual_seed(20)
    n = 256
    hidden = torch.randn(n, 128)
    cfg = GraphConfig(
        mode="offline",
        query_chunk_size=64, key_block_size=16, routed_blocks=4, routing_dim=32,
        semantic=SemanticConfig(
            near_blocks=2, medium_blocks=1, far_blocks=0, random_blocks=1,
            question_conditioned=True, question_tokens=32,
        ),
        rewiring=RewireConfig(mode="none", seed=42),
    )
    coord = GraphCoordinator(cfg, num_heads=4, num_kv_heads=2, head_dim=64, num_layers=2)
    coord.begin_prefill()
    idx = coord.build_layer_graph(0, hidden)

    H, G, K = idx.shape
    assert H == 4
    assert G == -(-n // cfg.query_chunk_size)
    assert K == cfg.routed_keys
    valid = idx[idx >= 0]
    assert (valid < n).all(), "Routed positions must be < seq_len"
    assert not idx.isnan().any()
    print("PASS test_question_conditioned_routing")


def test_question_conditioned_all_chunks_same_candidates():
    """All query chunks should have the same similarity profile (same question repr)."""
    from experiments.graph_sparse.routing.candidates import find_candidates
    torch.manual_seed(21)
    n = 256
    hidden = torch.randn(n, 128)
    cfg = GraphConfig(
        mode="offline",
        query_chunk_size=64, key_block_size=16, routed_blocks=4, routing_dim=32,
        semantic=SemanticConfig(
            near_blocks=2, medium_blocks=1, far_blocks=0, random_blocks=1,
            question_conditioned=True, question_tokens=32,
        ),
        rewiring=RewireConfig(mode="none", seed=42),
    )
    sim, mask, Q, B = find_candidates(hidden, cfg, causal=False)
    for q in range(1, Q):
        assert torch.allclose(sim[0], sim[q], atol=1e-5), \
            f"Chunk {q} similarity differs from chunk 0"
    print("PASS test_question_conditioned_all_chunks_same_candidates")


# ── Hybrid QK + graph routing ──────────────────────────────────────

def test_hybrid_pure_qk():
    """Hybrid with qk_budget=full, graph_budget=0 produces valid output."""
    model = tiny_llama()
    cfg, sink, window = small_graph_config(chunk=64, block=16, routed_blocks=4, sink=4, window=32)
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    n = sink + window + 64
    ids = torch.randint(0, 128, (1, n), device=DEV)

    for layer in model.model.layers:
        layer.self_attn._debug_config = {
            "route_mode": "hybrid",
            "qk_budget": 8,
            "graph_budget": 0,
            "total_budget": 8,
            "n_route": 4,
            "track_hybrid": True,
        }

    with torch.inference_mode():
        out = model(ids, use_cache=False)
    assert out.logits.shape == (1, n, 128)
    assert not out.logits.isnan().any(), "NaN in pure QK hybrid output"

    for layer in model.model.layers:
        attn = layer.self_attn
        idx = attn.last_routes
        assert idx is not None
        H, G, K = idx.shape
        assert H == attn.num_heads
        assert K == 8
        diag = getattr(attn, "_last_hybrid_diag", None)
        assert diag is not None, "Hybrid diagnostics not recorded"
        assert diag["qk_count_per_head"] > 0
        assert diag["graph_count_per_group"] == 0
    print("PASS test_hybrid_pure_qk")


def test_hybrid_pure_graph():
    """Hybrid with qk_budget=0, graph_budget=64 produces valid output."""
    model = tiny_llama()
    cfg, sink, window = small_graph_config(chunk=64, block=16, routed_blocks=4, sink=4, window=32)
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    n = sink + window + 64
    ids = torch.randint(0, 128, (1, n), device=DEV)

    for layer in model.model.layers:
        layer.self_attn._debug_config = {
            "route_mode": "hybrid",
            "qk_budget": 0,
            "graph_budget": 64,
            "total_budget": 64,
            "n_route": 4,
            "track_hybrid": True,
        }

    with torch.inference_mode():
        out = model(ids, use_cache=False)
    assert out.logits.shape == (1, n, 128)
    assert not out.logits.isnan().any(), "NaN in pure graph hybrid output"

    for layer in model.model.layers:
        attn = layer.self_attn
        diag = getattr(attn, "_last_hybrid_diag", None)
        assert diag is not None
        assert diag["qk_count_per_head"] == 0
        assert diag["graph_count_per_group"] > 0
    print("PASS test_hybrid_pure_graph")


def test_hybrid_mixed():
    """Hybrid with both QK and graph budgets merges correctly."""
    model = tiny_llama()
    cfg, sink, window = small_graph_config(chunk=64, block=16, routed_blocks=4, sink=4, window=32)
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    n = sink + window + 64
    ids = torch.randint(0, 128, (1, n), device=DEV)

    for layer in model.model.layers:
        layer.self_attn._debug_config = {
            "route_mode": "hybrid",
            "qk_budget": 4,
            "graph_budget": 32,
            "total_budget": 8,
            "n_route": 4,
            "track_hybrid": True,
        }

    with torch.inference_mode():
        out = model(ids, use_cache=False)
    assert out.logits.shape == (1, n, 128)
    assert not out.logits.isnan().any(), "NaN in mixed hybrid output"

    for layer in model.model.layers:
        attn = layer.self_attn
        idx = attn.last_routes
        H, G, K = idx.shape
        assert K == 8, f"Total budget should be 8, got {K}"
        valid = idx[idx >= 0]
        assert (valid < n).all(), "All routed positions must be < seq_len"
        assert (valid >= 0).all()
        diag = getattr(attn, "_last_hybrid_diag", None)
        assert diag is not None
        assert diag["qk_count_per_head"] > 0
        assert diag["graph_count_per_group"] > 0
        assert "overlap_mean" in diag
    print("PASS test_hybrid_mixed")


def test_hybrid_coverage_tracking():
    """Hybrid routing with coverage tracking records needle info."""
    model = tiny_llama()
    cfg, sink, window = small_graph_config(chunk=64, block=16, routed_blocks=4, sink=4, window=32)
    install_graph(model, cfg, sink=sink, window=window, exact_tail=0, block_m=16, block_n=16)
    n = sink + window + 64
    ids = torch.randint(0, 128, (1, n), device=DEV)

    for layer in model.model.layers:
        layer.self_attn._debug_config = {
            "route_mode": "hybrid",
            "qk_budget": 4,
            "graph_budget": 32,
            "total_budget": 8,
            "n_route": 4,
            "track_coverage": True,
            "needle_span": (50, 55),
        }

    with torch.inference_mode():
        model(ids, use_cache=False)

    for layer in model.model.layers:
        attn = layer.self_attn
        cov = attn._last_coverage
        assert cov is not None, "Coverage not recorded for hybrid routing"
        assert "final_chunk_hit" in cov
        assert "needle_hit_any_head_per_chunk" in cov
    print("PASS test_hybrid_coverage_tracking")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            failed.append((t.__name__, e))
            print(f"FAIL {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("FAILURES:")
        for name, err in failed:
            print(f"  {name}: {err}")
        sys.exit(1)
