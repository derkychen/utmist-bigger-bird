"""FP8 linear layers (modal_rd/fp8.py): quantization error, chunking, install on a tiny Llama, CUDA-graph decode.

On GPUs without FP8 tensor cores the emulated path runs (same quantized numbers, FP32 matmul)."""
import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from modal_rd import attention as rd
from kernels.fp8_gemv import fp8_gemv
from modal_rd.fp8 import FP8Linear, install_fp8, quantize_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("n,k", [(384, 512), (1024, 4096), (333, 1000)])
def test_gemv_matches_dequantized(n, k):
    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    q, s = quantize_rows(w)
    x = torch.randn(k, device="cuda", dtype=torch.bfloat16)
    y = fp8_gemv(x, q.view(torch.uint8), s.view(-1))
    ref = (q.float() * s) @ x.float()
    torch.testing.assert_close(y.float(), ref, atol=2e-2, rtol=2e-2)


def test_dequant_matches_reference():
    from kernels.fp8_gemv import fp8_dequant
    torch.manual_seed(2)
    w = torch.randn(300, 700, device="cuda", dtype=torch.bfloat16)
    q, s = quantize_rows(w)
    ref = (q.float() * s).to(torch.bfloat16)
    torch.testing.assert_close(fp8_dequant(q.view(torch.uint8), s.view(-1)), ref, atol=0, rtol=0)


def test_linear_error_one_and_many_tokens():
    torch.manual_seed(0)
    lin = torch.nn.Linear(512, 384, bias=False).cuda().to(torch.bfloat16)
    f = FP8Linear(lin)
    for shape in ((1, 1, 512), (3, 1000, 512)):
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        ref = lin(x).float()
        y = f(x)
        assert y.shape == ref.shape and y.dtype == torch.bfloat16
        rel = (y.float() - ref).norm() / ref.norm()
        assert rel < 0.05, rel                              # FP8 E4M3 weights: a few percent per matmul
    q, s = quantize_rows(lin.weight)
    assert q.dtype == torch.float8_e4m3fn and s.shape == (384, 1)


def _tiny():
    torch.manual_seed(1)
    cfg = LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096)
    cfg._attn_implementation = "sdpa"
    return LlamaForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16).eval()


@torch.inference_mode()
def test_install_on_llama_with_rd_wrappers():
    model = _tiny()
    ids = torch.randint(0, 512, (1, 300), device="cuda")
    ref = model(ids).logits.float()
    rd.install(model, "dense_rd", sink=4, window=64, vertical=16, exact_tail=0)
    install_fp8(model)
    for layer in model.model.layers:
        base = layer.self_attn._rd_base
        assert isinstance(base.q_proj, FP8Linear) and layer.self_attn.q_proj is base.q_proj
        assert isinstance(layer.mlp.down_proj, FP8Linear)
    out = model(ids).logits.float()
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    assert cos > 0.99, cos
    rd.install(model, "vs", sink=4, window=64, vertical=16, exact_tail=0)   # re-install keeps FP8 projections
    assert isinstance(model.model.layers[0].self_attn.k_proj, FP8Linear)


@torch.inference_mode()
def test_graph_decode_matches_static_step():
    from modal_rd.graph_decode import StaticDecoder

    model = _tiny()
    install_fp8(model)
    ids = torch.randint(0, 512, (1, 200), device="cuda")
    tokens = []
    for graph in (False, True):
        cache = DynamicCache(config=model.config)
        first = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[0, -1].argmax().item()
        dec = StaticDecoder(model, 300)
        dec.load(cache, 200, first)
        dec.run(20, graph=graph)
        tokens.append(dec.generated())
    assert tokens[0] == tokens[1]
