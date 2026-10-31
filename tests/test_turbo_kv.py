"""TurboQuant-style KV quantizer (modal_rd/turbo_kv.py): codebook, distortion, cache round trip on a tiny Llama."""
import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from modal_rd.turbo_kv import TurboKVCache, TurboQuantizer, lloyd_max_gaussian

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_lloyd_max_known_values():
    c3 = lloyd_max_gaussian(3)
    torch.testing.assert_close(c3[4:], torch.tensor([0.2451, 0.7560, 1.3439, 2.1520]), atol=2e-3, rtol=0)
    c2 = lloyd_max_gaussian(2)
    torch.testing.assert_close(c2[2:], torch.tensor([0.4528, 1.5104]), atol=2e-3, rtol=0)


@pytest.mark.parametrize("bits,max_rel_mse", [(4, 0.012), (3, 0.04), (2, 0.13)])
def test_distortion(bits, max_rel_mse):
    torch.manual_seed(0)
    x = torch.randn(4, 5000, 128, device="cuda") * torch.rand(4, 5000, 1, device="cuda") * 3   # varied norms
    x[..., 7] += 5                                                                              # an outlier channel
    q = TurboQuantizer(128, bits, "cuda")
    codes, norm = q.encode(x)
    y = q.decode(codes, norm, torch.float32)
    rel = ((y - x) ** 2).sum() / (x ** 2).sum()
    assert rel < max_rel_mse, rel                 # Lloyd-Max N(0,1): ~0.0095 (4 b), ~0.0345 (3 b), ~0.118 (2 b)


@torch.inference_mode()
def test_cache_on_tiny_llama():
    torch.manual_seed(1)
    cfg = LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096)
    cfg._attn_implementation = "sdpa"
    model = LlamaForCausalLM(cfg).to("cuda", torch.bfloat16).eval()
    ids = torch.randint(0, 512, (1, 400), device="cuda")
    outs = []
    for cache in (DynamicCache(config=cfg), TurboKVCache(model, 500, bits=4), TurboKVCache(model, 500, 4, stage_on_cpu=True)):
        first = model(input_ids=ids, past_key_values=cache, use_cache=True)
        if hasattr(cache, "after_prefill"):
            cache.after_prefill()
        nxt = first.logits[0, -1].argmax().view(1, 1)
        o = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        outs.append((first.logits[0, -1].float(), o.logits[0, -1].float()))
    assert torch.equal(outs[0][0], outs[1][0])                     # prefill attends over exact K/V
    assert torch.equal(outs[1][1], outs[2][1])                     # staging on the CPU changes nothing
    cos = torch.nn.functional.cosine_similarity(outs[0][1], outs[1][1], dim=0)
    assert cos > 0.98, cos                                         # decode over the 4-bit cache
