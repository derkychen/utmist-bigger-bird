"""In-place RoPE, chunked RMSNorm and the CPU embedding table are bit-identical to the HF ops they replace."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from modal_rd import attention as rd
from modal_rd.cpu_embed import CPUEmbedding, install_cpu_embedding
from modal_rd.mlp_chunk import install_chunked_mlp
from tests.test_rd_attention_gqa import DEV, DT


def tiny(n_layers=2):
    cfg = LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=1024, num_hidden_layers=n_layers,
                      num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=8192)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).to(DEV, DT).eval()


@pytest.mark.parametrize("n,chunk", [(300, 64), (300, 300), (257, 1000), (1, 64)])
def test_rope_inplace_is_bit_identical(n, chunk):
    model = tiny(1)
    torch.manual_seed(1)
    q = torch.randn(1, 8, n, 32, device=DEV, dtype=DT)
    k = torch.randn(1, 2, n, 32, device=DEV, dtype=DT)
    cos, sin = model.model.rotary_emb(q, torch.arange(n, device=DEV)[None])
    want_q, want_k = apply_rotary_pos_emb(q, k, cos, sin)
    got_q = rd.rope_inplace(q.clone(), cos, sin, chunk)
    got_k = rd.rope_inplace(k.transpose(1, 2).contiguous().transpose(1, 2), cos, sin, chunk)   # strided view
    assert torch.equal(got_q, want_q) and torch.equal(got_k, want_k)


def test_rd_attention_logits_unchanged_by_inplace_rope(monkeypatch):
    """Whole model with the sparse arm: in-place RoPE path == HF apply_rotary_pos_emb path."""
    model = tiny()
    rd.install(model, "vs", sink=4, window=32, vertical=16, n_route=8, exact_tail=16, block_m=16, block_n=16)
    ids = torch.randint(0, 512, (1, 200), device=DEV)
    with torch.inference_mode():
        got = model(ids).logits
        monkeypatch.setattr(rd, "rope_inplace", lambda x, cos, sin, chunk=0: apply_rotary_pos_emb(x, x, cos, sin)[0])
        want = model(ids).logits
    assert torch.equal(got, want)


@pytest.mark.parametrize("n,chunk", [(300, 64), (257, 128), (100, 1000)])
def test_chunked_norms_are_bit_identical(n, chunk):
    model = tiny(1)
    x = torch.randn(1, n, 256, device=DEV, dtype=DT)
    norms = [model.model.layers[0].input_layernorm, model.model.layers[0].post_attention_layernorm, model.model.norm]
    with torch.inference_mode():
        want = [m(x) for m in norms]
        install_chunked_mlp(model, chunk)
        got = [m(x) for m in norms]
        install_chunked_mlp(model, 0)
        restored = [m(x) for m in norms]
    assert all(torch.equal(g, w) for g, w in zip(got, want))
    assert all(torch.equal(r, w) for r, w in zip(restored, want))


def test_norms_flag_off_leaves_norms_alone():
    model = tiny(1)
    install_chunked_mlp(model, 64, norms=False)
    assert not hasattr(model.model.norm, "_unchunked_forward")


def test_cpu_embedding_is_bit_identical_and_frees_gpu():
    model = tiny()
    ids = torch.randint(0, 512, (1, 300), device=DEV)
    with torch.inference_mode():
        want = model(ids).logits
        gen_want = model.generate(ids[:, :50], max_new_tokens=5, do_sample=False)
    n_gpu = sum(p.numel() for p in model.parameters())
    install_cpu_embedding(model)
    install_cpu_embedding(model)                                       # idempotent
    assert isinstance(model.model.embed_tokens, CPUEmbedding)
    assert model.model.embed_tokens.weight_cpu.device.type == "cpu"
    assert sum(p.numel() for p in model.parameters()) == n_gpu - 512 * 256
    assert model.device.type == torch.device(DEV).type                 # generate() still puts inputs on DEV
    with torch.inference_mode():
        got = model(ids).logits
        gen_got = model.generate(ids[:, :50], max_new_tokens=5, do_sample=False)
    assert torch.equal(got, want) and torch.equal(gen_got, gen_want)
