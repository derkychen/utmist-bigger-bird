"""Exp 1 sparse flash backend: kernel semantics and backend selection.

Runs under pytest or `python scripts/run_tests_plain.py tests/test_exp1_flash_backend.py`.
Semantics (spec): window q-W < j <= q; routed r <= q-W; padded keys excluded;
padded query rows output 0.
"""
import unittest

import torch

from sparse_attn_utils import causal_sparse_attention

WINDOW = 256


def _require_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA required")


def _flash():
    _require_cuda()
    from kernels.bigger_bird_flash import bigger_bird_flash
    return bigger_bird_flash


def _qkv(bh, t, d=128, dtype=torch.bfloat16, device="cuda", seed=0):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(bh, t, d, generator=g) * d ** -0.5).to(device, dtype)
    k = torch.randn(bh, t, d, generator=g).to(device, dtype)
    v = torch.randn(bh, t, d, generator=g).to(device, dtype)
    return q, k, v


def _routed(bh, n, k, high=None, seed=1, device="cuda"):
    """Unique random key positions per head, drawn from [0, high)."""
    g = torch.Generator().manual_seed(seed)
    high = n if high is None else high
    k = min(k, high)
    return torch.stack([torch.randperm(high, generator=g)[:k] for _ in range(bh)]).to(device)


def reference(q, k, v, routed_idx, window=WINDOW, token_mask=None, num_heads=1):
    """Explicit-mask fp32 reference of the spec's dedup semantics."""
    bh, t, _ = q.shape
    n = k.shape[1]
    qpos = torch.arange(t, device=q.device) + (n - t)
    kpos = torch.arange(n, device=q.device)
    allowed = ((kpos[None, :] <= qpos[:, None])
               & (kpos[None, :] > qpos[:, None] - window)).expand(bh, t, n).clone()
    routed = torch.zeros(bh, n, dtype=torch.bool, device=q.device)
    routed.scatter_(1, routed_idx.to(q.device), True)
    allowed |= routed[:, None, :] & (kpos[None, None, :] <= qpos[None, :, None] - window)
    key_ok = None
    if token_mask is not None:
        key_ok = token_mask.repeat_interleave(num_heads, 0)
        allowed &= key_ok[:, None, :]
    scores = q.float() @ k.float().transpose(1, 2)
    probs = scores.masked_fill(~allowed, float("-inf")).softmax(-1).nan_to_num(0.0)
    out = probs @ v.float()
    if key_ok is not None:
        out = out * key_ok[:, qpos].unsqueeze(-1)
    return out


def _run(flash, q, k, v, idx, token_mask=None, num_heads=None):
    return flash(q, k, v, idx[:, None, :], front=0, window=WINDOW,
                 token_mask=token_mask, num_heads=num_heads or q.shape[0], scale=1.0)


def test_matches_reference_across_shapes():
    flash = _flash()
    for t in (1, 7, 300, 1000, 5000):
        for k in (1, 64, 512):
            q, kk, v = _qkv(4, t)
            idx = _routed(4, t, k)
            err = (_run(flash, q, kk, v, idx).float()
                   - reference(q, kk, v, idx)).abs().max().item()
            assert err < 2e-2, (t, k, err)


def test_matches_reference_fp16():
    flash = _flash()
    q, kk, v = _qkv(4, 2000, dtype=torch.float16)
    idx = _routed(4, 2000, 512)
    err = (_run(flash, q, kk, v, idx).float() - reference(q, kk, v, idx)).abs().max().item()
    assert err < 5e-3, err


def test_left_padded_batch():
    flash = _flash()
    heads, t = 4, 700
    q, kk, v = _qkv(2 * heads, t)
    idx = _routed(2 * heads, t, 64)
    mask = torch.ones(2, t, dtype=torch.bool, device="cuda")
    mask[1, :150] = False
    out = _run(flash, q, kk, v, idx, token_mask=mask, num_heads=heads)
    ref = reference(q, kk, v, idx, token_mask=mask, num_heads=heads)
    assert (out.float() - ref).abs().max().item() < 2e-2
    assert out[heads:, :150].abs().max().item() == 0


def test_non_contiguous_inputs():
    flash = _flash()
    q, kk, v = _qkv(4, 900)
    idx = _routed(4, 900, 128)
    base = _run(flash, q, kk, v, idx)
    qn, kn, vn = (x.transpose(0, 1).contiguous().transpose(0, 1) for x in (q, kk, v))
    assert not qn.is_contiguous()
    assert torch.equal(_run(flash, qn, kn, vn, idx), base)


def test_unselected_keys_never_read():
    flash = _flash()
    bh, t = 4, 4096
    q, kk, v = _qkv(bh, t)
    idx = _routed(bh, t, 128, high=2048)
    tail = slice(t - 64, t)  # exactly one 64-query tile
    out = _run(flash, q, kk, v, idx)
    used = torch.zeros(bh, t, dtype=torch.bool, device="cuda")
    used[:, t - 64 - WINDOW + 1:] = True
    used.scatter_(1, idx, True)
    k2, v2 = kk.clone(), v.clone()
    k2[~used] = (1e3 * torch.randn_like(k2[~used].float())).to(k2.dtype)
    v2[~used] = (1e3 * torch.randn_like(v2[~used].float())).to(v2.dtype)
    out2 = _run(flash, q, k2, v2, idx)
    assert torch.equal(out[:, tail], out2[:, tail])


def test_agrees_with_legacy_torch_path_away_from_overlap():
    flash = _flash()
    bh, t = 4, 3000
    q, kk, v = _qkv(bh, t)
    idx = _routed(bh, t, 256, high=1000)
    out = _run(flash, q, kk, v, idx)
    legacy = causal_sparse_attention(q, kk, v, idx, local_window=WINDOW,
                                     bsz=1, num_heads=bh)
    far = slice(1000 + WINDOW, t)  # every routed key is outside these windows
    err = (out[:, far].float() - legacy[:, far].float()).abs().max().item()
    assert err < 2e-2, err


def _exp1_layer(backend, device="cuda", dtype=torch.bfloat16, heads=32, kv_heads=8,
                head_dim=128, **kwargs):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention
    from experiments.exp_1_deepseek_topk.model_llama import DeepSeekTopKAttention
    cfg = LlamaConfig(hidden_size=heads * head_dim, num_attention_heads=heads,
                      num_key_value_heads=kv_heads, head_dim=head_dim)
    layer = DeepSeekTopKAttention(LlamaAttention(cfg, layer_idx=0), top_k=512,
                                  low_rank_dim=128, attn_backend=backend, **kwargs)
    layer.capture_first_route = True
    return layer.to(device, dtype).eval()


def _layer_vs_reference(layer, t, heads=32, d=128):
    q, kk, v = _qkv(heads, t, d=d)
    out = layer.sparse_attention(q, kk, v, None, 1, heads, is_causal=True)
    idx = layer.first_routed_indices.to("cuda")
    assert idx.shape[0] == heads
    assert all(row.unique().numel() == row.numel() for row in idx)
    err = (out.float() - reference(q, kk, v, idx)).abs().max().item()
    assert torch.isfinite(out).all() and out.shape == q.shape
    assert err < 2e-2, err


def test_invalid_backend_rejected():
    try:
        _exp1_layer("flashy", device="cpu", dtype=torch.float32, heads=4, kv_heads=2, head_dim=64)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_cpu_falls_back_to_torch():
    kw = dict(device="cpu", dtype=torch.float32, heads=4, kv_heads=2, head_dim=64)
    q, kk, v = _qkv(4, 600, d=64, dtype=torch.float32, device="cpu")
    outs = [_exp1_layer(b, **kw).sparse_attention(q, kk, v, None, 1, 4, is_causal=True)
            for b in ("flash", "torch")]
    assert torch.equal(outs[0], outs[1])


def test_cuda_fp32_falls_back():
    _require_cuda()
    kw = dict(device="cuda", dtype=torch.float32, heads=4, kv_heads=2, head_dim=64)
    q, kk, v = _qkv(4, 600, d=64, dtype=torch.float32)
    outs = [_exp1_layer(b, **kw).sparse_attention(q, kk, v, None, 1, 4, is_causal=True)
            for b in ("flash", "torch")]
    assert torch.equal(outs[0], outs[1])


def test_flash_layer_real_shapes():
    _require_cuda()
    _layer_vs_reference(_exp1_layer("flash"), 4096)


def test_flash_layer_short_prompt():
    _require_cuda()
    _layer_vs_reference(_exp1_layer("flash"), 7)


def test_flash_layer_dense_budget():
    _require_cuda()
    _layer_vs_reference(_exp1_layer("flash"), 1500)


def test_audit_records_exp1_kernel():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_dashboard.py"
    spec = importlib.util.spec_from_file_location("build_dashboard", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report = mod.audit_sparse_models()
    assert report["status"] == "pass", report["violations"]
    exp1 = [f for f in report["files"] if f["experiment"] == "exp_1"]
    assert exp1 and exp1[0]["attention_kernel"] == "bigger_bird_flash"


def test_records_effective_backend():
    _require_cuda()
    q, kk, v = _qkv(32, 300)
    layer = _exp1_layer("flash")
    layer.sparse_attention(q, kk, v, None, 1, 32, is_causal=True)
    assert layer.last_backend == "flash"
    q32, k32, v32 = _qkv(4, 300, d=64, dtype=torch.float32)
    layer32 = _exp1_layer("flash", dtype=torch.float32, heads=4, kv_heads=2, head_dim=64)
    layer32.sparse_attention(q32, k32, v32, None, 1, 4, is_causal=True)
    assert layer32.last_backend == "torch"


def test_registry_records_exp1_backend():
    from eval.ruler_llama import run_generative as ruler
    assert ruler.EXP_REGISTRY[1][2]["attn_backend"] == "flash"
