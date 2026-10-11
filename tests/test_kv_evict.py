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
