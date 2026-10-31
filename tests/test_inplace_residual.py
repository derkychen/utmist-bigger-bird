"""In-place residual adds (modal_rd/inplace_residual.py) are bit-identical and lower the prefill peak."""
import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from modal_rd import attention as rd
from modal_rd.cpu_embed import install_cpu_embedding
from modal_rd.inplace_residual import install_inplace_residual
from modal_rd.mlp_chunk import install_chunked_mlp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def tiny(hidden=256, layers=2):
    cfg = LlamaConfig(vocab_size=512, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                      num_attention_heads=hidden // 64, num_key_value_heads=2, max_position_embeddings=32768)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).to("cuda", torch.bfloat16).eval()


@torch.inference_mode()
def run(model, ids, steps=6):
    cache = DynamicCache(config=model.config)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    logits, nxt, toks = [out.logits.float()], out.logits[0, -1].argmax().view(1, 1), []
    for _ in range(steps):
        o = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        logits.append(o.logits.float())
        nxt = o.logits[0, -1].argmax().view(1, 1)
        toks.append(int(nxt))
    return logits, toks


@pytest.mark.parametrize("arm", [None, "dense_rd", "vs"])
@pytest.mark.parametrize("cpu_embed", [False, True])
def test_bit_identical(arm, cpu_embed):
    model = tiny()
    install_chunked_mlp(model, 64)
    if arm:
        rd.install(model, arm, sink=4, window=64, vertical=16, exact_tail=8)
    if cpu_embed:
        install_cpu_embedding(model)
    ids = torch.randint(0, 512, (1, 300), device="cuda")
    want = run(model, ids)
    install_inplace_residual(model)
    got = run(model, ids)
    assert got[1] == want[1]
    assert all(torch.equal(a, b) for a, b in zip(got[0], want[0]))


@torch.inference_mode()
def test_lower_prefill_peak():
    model = tiny(hidden=1024, layers=4)
    ids = torch.randint(0, 512, (1, 16384), device="cuda")
    peaks = []
    for patch in (False, True):
        if patch:
            install_inplace_residual(model)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        model(input_ids=ids, logits_to_keep=1)
        peaks.append(torch.cuda.max_memory_allocated() - base)
    saved = (peaks[0] - peaks[1]) / (16384 * 1024 * 2)          # in units of one [n, hidden] BF16 tensor
    assert saved >= 0.9, (peaks, saved)
