"""INT8 KV cache: small roundtrip error, prefill untouched, ~0.51x storage, composes with kv_evict."""
import pytest
import torch
from transformers import DynamicCache

from modal_rd import attention as rd
from modal_rd.kv_int8 import Int8KVLayer, dequantize, make_int8_cache, quantize
from tests.test_kv_evict import KW, N, prompt
from tests.test_rd_attention_gqa import DEV, tiny_llama


def gen(model, ids, cache):
    with torch.inference_mode():
        return model.generate(ids, max_new_tokens=6, min_new_tokens=6, do_sample=False, pad_token_id=0,
                              past_key_values=cache, output_scores=True, return_dict_in_generate=True)


def max_diff(a, b):
    ok = torch.isfinite(a) & torch.isfinite(b)          # min_new_tokens puts -inf on EOS in both
    assert torch.equal(torch.isfinite(a), torch.isfinite(b))
    return (a - b)[ok].abs().max().item()


def test_roundtrip_error_is_small():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 64, 128) * torch.rand(1, 4, 64, 1) * 5       # per-token scale spread
    q, s = quantize(x)
    assert q.dtype == torch.int8 and s.dtype == torch.float16 and s.shape == (1, 4, 64, 1)
    err = (dequantize(q, s, torch.float32) - x).abs()
    rel = (err.pow(2).mean() / x.pow(2).mean()).sqrt().item()
    print(f"\nint8 per-token roundtrip: rms rel err={rel:.4f}  max abs err/absmax={(err / x.abs().amax(-1, keepdim=True)).max():.4f}")
    assert rel < 0.01 and (err <= s.float() * 0.6).all()               # half a step + fp16 rounding of the scale


def test_storage_is_about_half_of_bf16():
    layer = Int8KVLayer()
    k = torch.randn(1, 8, 1000, 128, dtype=torch.bfloat16)
    layer.update(k, k.clone())
    ratio = layer.nbytes() / (2 * k.numel() * 2)
    print(f"\nint8 layer bytes / bf16 bytes = {ratio:.4f}")
    assert abs(ratio - (1 + 2 / 128) / 2) < 1e-3


def test_prefill_unchanged_and_decode_close():
    ids = prompt()
    model = tiny_llama()
    rd.install(model, "vs", **KW)
    want = gen(model, ids, DynamicCache())
    cache = make_int8_cache(model.config)
    got = gen(model, ids, cache)
    assert torch.equal(got.scores[0], want.scores[0]), "prefill logits must be identical"
    diffs = [max_diff(a, b) for a, b in zip(got.scores, want.scores)]
    print("\nmax |logit diff| per step, int8 vs fp32 cache:", [round(d, 5) for d in diffs])
    assert max(diffs) < 0.05
    assert all(isinstance(l, Int8KVLayer) and l.get_seq_length() == N + 5 for l in cache.layers)


def test_composes_with_kv_evict():
    ids = prompt()
    model = tiny_llama()
    rd.install(model, "vs", kv_evict=True, **KW)
    want = gen(model, ids, DynamicCache())
    cache = make_int8_cache(model.config)
    got = gen(model, ids, cache)
    cfg = model.config
    g = cfg.num_attention_heads // cfg.num_key_value_heads
    m = KW["sink"] + KW["window"] + g * KW["vertical"]
    assert all(l.get_seq_length() == m + 5 for l in cache.layers), "int8 layer must hold the evicted length"
    diffs = [max_diff(a, b) for a, b in zip(got.scores, want.scores)]
    print("\nevict+int8 vs evict-only, max |logit diff| per step:", [round(d, 5) for d in diffs])
    assert torch.equal(got.scores[0], want.scores[0]) and max(diffs) < 0.05
    full_bytes = 2 * N * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 4  # fp32 test model
    print(f"per-layer cache bytes: full fp32={full_bytes}  evict+int8={cache.layers[0].nbytes()}")


def test_decode_appends_past_spare_capacity_and_crop():
    """Token-by-token appends across several buffer growths equal quantising the whole sequence at once."""
    torch.manual_seed(1)
    k = torch.randn(1, 2, 700, 16, device=DEV)
    v = torch.randn(1, 2, 700, 16, device=DEV)
    layer = Int8KVLayer()
    pk, pv = layer.update(k[..., :100, :], v[..., :100, :])
    assert torch.equal(pk, k[..., :100, :]) and torch.equal(pv, v[..., :100, :]), "prefill returns full precision"
    for i in range(100, 700):
        got_k, got_v = layer.update(k[..., i:i + 1, :], v[..., i:i + 1, :])
    want_k, want_v = (dequantize(*quantize(x), torch.float32) for x in (k, v))
    assert layer.get_seq_length() == 700 and got_k.shape[-2] == 700
    assert torch.equal(got_k, want_k) and torch.equal(got_v, want_v)
    assert layer.nbytes() == 2 * (700 * 2 * 16 + 700 * 2 * 2)
    layer.crop(-50)
    assert layer.get_seq_length() == 650 and torch.equal(layer.keys, want_k[..., :650, :])
    layer.update(k[..., :1, :], v[..., :1, :])
    assert torch.equal(layer.keys[..., 650:, :], want_k[..., :1, :])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")
def test_triton_kernels_match_reference_bitwise():
    from kernels import kv_int8_triton
    from modal_rd.kv_int8 import MIN_SCALE
    torch.manual_seed(2)
    for shape in [(1, 8, 128), (777, 8, 128), (4096, 8, 128)]:
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * torch.rand(shape[0], 8, 1, device="cuda") * 4
        x[0, 0] = 0                                                         # all-zero row: MIN_SCALE path
        for inp in (x, x.transpose(0, 1).contiguous().transpose(0, 1)):     # contiguous and strided rows
            q, s = quantize(inp)
            tq, ts = kv_int8_triton.quantize(inp, MIN_SCALE)
            assert torch.equal(q, tq) and torch.equal(s, ts)
            for dt in (torch.bfloat16, torch.float32):
                assert torch.equal(kv_int8_triton.dequantize(tq, ts, dt), dequantize(q, s, dt))
