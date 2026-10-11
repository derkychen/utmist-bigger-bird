"""Chunked MLP is exact and bounds the per-op transient size."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from modal_rd.mlp_chunk import install_chunked_mlp
from tests.test_rd_attention_gqa import AllocLog, DEV


def tiny(n_layers=2):
    cfg = LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=1024, num_hidden_layers=n_layers,
                      num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=8192)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).to(DEV).eval()


@pytest.mark.parametrize("n,chunk", [(300, 64), (300, 300), (300, 1000), (257, 128)])
def test_chunked_mlp_matches_unchunked(n, chunk):
    model = tiny()
    ids = torch.randint(0, 512, (1, n), device=DEV)
    with torch.inference_mode():
        want = model(ids).logits
        install_chunked_mlp(model, chunk)
        got = model(ids).logits
        install_chunked_mlp(model, 0)
        restored = model(ids).logits
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    assert torch.equal(restored, want)


def test_chunking_bounds_mlp_transients():
    n, chunk = 4096, 512
    model = tiny(1)
    x = torch.randn(1, n, 256, device=DEV)
    mlp = model.model.layers[0].mlp

    def peak_op_bytes():
        with torch.inference_mode(), AllocLog() as log:
            mlp(x)
        return max(b for b, _ in log.records)

    full = peak_op_bytes()
    install_chunked_mlp(model, chunk)
    chunked = peak_op_bytes()
    print(f"\nlargest MLP op output: full={full/1e6:.1f}MB chunked={chunked/1e6:.1f}MB")
    # the full-size output buffer remains; the 4x-wide gate/up/act transients do not
    assert full >= n * 1024 * 4                     # unchunked gate/up/act are [n, intermediate] fp32
    assert chunked < full
