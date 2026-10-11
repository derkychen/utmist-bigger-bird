"""kv_evict: prefill unchanged, cache shrinks to sink + routed + window, decode == masked-full-cache decode."""
import types

import pytest
import torch
import torch.nn.functional as F
from transformers import DynamicCache

from modal_rd import attention as rd
from tests.test_rd_attention_gqa import DEV, tiny_llama

KW = dict(sink=8, window=96, vertical=24, n_route=16, exact_tail=24, block_m=16, block_n=16)
N = 400


def prompt():
    torch.manual_seed(4)
    return torch.randint(0, 512, (1, N), device=DEV)


def prefill(model, ids):
    cache = DynamicCache()
    with torch.inference_mode():
        logits = model(ids, past_key_values=cache, use_cache=True).logits
    return logits, cache


def test_prefill_logits_unchanged_and_cache_shrinks():
    ids = prompt()
    model = tiny_llama()
    rd.install(model, "vs", **KW)
    want, full_cache = prefill(model, ids)
    rd.install(model, "vs", kv_evict=True, **KW)
    got, cache = prefill(model, ids)
    assert torch.equal(got, want), "eviction must not change prefill output"
    cfg = model.config
    g = cfg.num_attention_heads // cfg.num_key_value_heads
    m = KW["sink"] + KW["window"] + g * KW["vertical"]
    for i, layer in enumerate(cache.layers):
        assert layer.keys.shape == (1, cfg.num_key_value_heads, m, cfg.hidden_size // cfg.num_attention_heads)
        assert full_cache.layers[i].keys.shape[-2] == N
    print(f"\ncache tokens/layer: {N} -> {m}")


def test_retained_set_and_values():
    ids = prompt()
    model = tiny_llama()
    rd.install(model, "vs", **KW)
    _, full = prefill(model, ids)
    rd.install(model, "vs", kv_evict=True, **KW)
    _, cache = prefill(model, ids)
    g = model.config.num_attention_heads // model.config.num_key_value_heads
    for i, layer in enumerate(model.model.layers):
        a = layer.self_attn
        kept = a.kept                                              # [Hkv, m]
        assert (kept[:, :-1] < kept[:, 1:]).all(), "positions sorted and unique"
        assert (kept[:, :KW["sink"]] == torch.arange(KW["sink"], device=DEV)).all()
        assert (kept[:, -KW["window"]:] == torch.arange(N - KW["window"], N, device=DEV)).all()
        for h in range(model.config.num_attention_heads):          # every routed column survives
            kv = h // g
            assert torch.isin(a.last_routes[h].long(), kept[kv]).all()
        gi = kept[None, :, :, None].expand(1, -1, -1, full.layers[i].keys.shape[-1])
        assert torch.equal(cache.layers[i].keys, full.layers[i].keys.gather(2, gi))
        assert torch.equal(cache.layers[i].values, full.layers[i].values.gather(2, gi))


def test_decode_matches_masked_full_cache_reference():
    ids = prompt()
    # evicted model: real shrunken cache through HF generate (also checks RoPE positions after eviction)
    ev = tiny_llama()
    rd.install(ev, "vs", kv_evict=True, **KW)
    with torch.inference_mode():
        out = ev.generate(ids, max_new_tokens=6, min_new_tokens=6, do_sample=False, use_cache=True, pad_token_id=0,
                          output_scores=True, return_dict_in_generate=True)
    kept = [l.self_attn.kept for l in ev.model.layers]

    # reference: full cache, decode masked to the same retained positions (+ every generated token)
    ref = tiny_llama()
    rd.install(ref, "vs", **KW)
    for layer, kp in zip(ref.model.layers, kept):
        a = layer.self_attn
        orig = a.exact_attention

        def masked(self, q, k, v, t, n, _orig=orig, _kp=kp):
            if t == n:
                return _orig(q, k, v, t, n)
            keep = torch.zeros(self.num_kv_heads, n, dtype=torch.bool, device=q.device)
            keep.scatter_(1, _kp, True)
            keep[:, N:] = True
            mask = keep.repeat_interleave(self.groups, 0)[None, :, None, :]
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scaling, enable_gqa=True)

        a.exact_attention = types.MethodType(masked, a)
    with torch.inference_mode():
        want = ref.generate(ids, max_new_tokens=6, min_new_tokens=6, do_sample=False, use_cache=True, pad_token_id=0,
                            output_scores=True, return_dict_in_generate=True)
    assert len(out.scores) == 6, "all decode steps must run"
    assert torch.equal(out.sequences, want.sequences)
    for a, b in zip(out.scores, want.scores):
        tol = 1e-4 if DEV == "cpu" else 2e-2  # BF16 on GPU; eviction itself moves scores by ~1.0
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)


@pytest.mark.parametrize("tune", [dict(evict_keep=160), dict(evict_window=160), dict(evict_pool=8),
                                  dict(evict_obs=24), dict(evict_full_layers=1), dict(evict_pool_back=8),
                                  dict(evict_keep=160, evict_window=160, evict_pool=8, evict_obs=24)],
                         ids=lambda t: ",".join(f"{k}={v}" for k, v in t.items()))
def test_tuning_knobs(tune):
    """Tuned eviction: prefill unchanged, routed columns + sink + (longer) window kept, cache sized as configured."""
    ids = prompt()
    model = tiny_llama()
    rd.install(model, "vs", **KW)
    want, full = prefill(model, ids)
    rd.install(model, "vs", kv_evict=True, **KW, **tune)
    got, cache = prefill(model, ids)
    assert torch.equal(got, want), "eviction must not change prefill output"
    g = model.config.num_attention_heads // model.config.num_key_value_heads
    w = max(KW["window"], tune.get("evict_window", 0))
    mid = max(g * KW["vertical"], tune.get("evict_keep", 0))
    for i, layer in enumerate(model.model.layers):
        a, kept = layer.self_attn, layer.self_attn.kept
        if i < tune.get("evict_full_layers", 0):
            assert cache.layers[i].keys.shape[-2] == N, "full layers are not evicted"
            continue
        assert kept.shape[1] == KW["sink"] + w + mid
        assert (kept[:, :-1] < kept[:, 1:]).all()
        assert (kept[:, -w:] == torch.arange(N - w, N, device=DEV)).all()
        for h in range(model.config.num_attention_heads):
            assert torch.isin(a.last_routes[h].long(), kept[h // g]).all()
        gi = kept[None, :, :, None].expand(1, -1, -1, full.layers[i].keys.shape[-1])
        assert torch.equal(cache.layers[i].keys, full.layers[i].keys.gather(2, gi))


def test_forward_pool_keeps_span_continuation():
    """evict_pool=p: the p-1 keys after a high-scoring key inherit its score; others are untouched."""
    score = torch.zeros(2, 20)
    score[0, 5] = 1.0
    score[1, 18] = 0.5
    pooled = rd.forward_pool(score, 4)
    assert pooled.shape == score.shape
    assert pooled[0, 5:9].eq(1).all() and pooled[0, :5].eq(0).all() and pooled[0, 9:].eq(0).all()
    assert pooled[1, 18:].eq(0.5).all() and pooled[1, :18].eq(0).all()


def test_forward_pool_back_keeps_span_lead_in():
    """evict_pool_back=b: the b keys before a high-scoring key inherit its score too."""
    score = torch.zeros(1, 20)
    score[0, 10] = 1.0
    pooled = rd.forward_pool(score, 3, back=4)
    assert pooled[0, 6:13].eq(1).all() and pooled[0, :6].eq(0).all() and pooled[0, 13:].eq(0).all()
    assert rd.forward_pool(score, 0, back=2)[0, 8:11].eq(1).all()
