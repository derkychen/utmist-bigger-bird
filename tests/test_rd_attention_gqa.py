"""modal_rd.attention: GQA-in-kernel + kernel-based exact tail must match the previous implementation.

Runs on CPU (pure-PyTorch kernel reference) or CUDA (Triton). The previous implementation is vendored
in tests/fixtures as an equivalence oracle; delete it once the memory fix is confirmed on GPU.
"""
import importlib.util
import itertools
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import LlamaConfig, LlamaForCausalLM

from kernels.bigger_bird_flash_ref import bigger_bird_flash_ref
from modal_rd import attention as new_attn

kernel = new_attn.bigger_bird_flash  # Triton on CUDA BF16, reference otherwise

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# The Triton kernel only takes FP16/BF16, so on GPU everything runs in BF16.
DT = torch.bfloat16 if DEV == "cuda" else torch.float32


def load_old():
    """Previous modal_rd/attention.py, wired to whichever flash kernel is available."""
    stub = types.ModuleType("kernels.bigger_bird_flash")
    stub.bigger_bird_flash = kernel
    saved = sys.modules.get("kernels.bigger_bird_flash")
    sys.modules["kernels.bigger_bird_flash"] = stub
    try:
        path = Path(__file__).parent / "fixtures" / "rd_attention_before_gqa_fix.py"
        spec = importlib.util.spec_from_file_location("rd_attention_old", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is None:
            del sys.modules["kernels.bigger_bird_flash"]
        else:
            sys.modules["kernels.bigger_bird_flash"] = saved
    return mod


old_attn = load_old()


def tiny_llama(dtype=DT):
    # 16 query / 4 KV heads: the 4:1 GQA ratio of Llama-3-8B.
    cfg = LlamaConfig(vocab_size=512, hidden_size=512, intermediate_size=1024, num_hidden_layers=2,
                      num_attention_heads=16, num_key_value_heads=4, max_position_embeddings=8192)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).to(DEV, dtype).eval()


def prefill_logits(model, ids):
    with torch.inference_mode():
        return model(ids, use_cache=True, logits_to_keep=0).logits.float()


def test_ref_kernel_grouped_equals_repeated():
    torch.manual_seed(1)
    h, g, n, d = 8, 4, 384, 32
    q = torch.randn(h, n, d, device=DEV, dtype=DT)
    k, v = torch.randn(h // g, n, d, device=DEV, dtype=DT), torch.randn(h // g, n, d, device=DEV, dtype=DT)
    idx = torch.stack([torch.randperm(n - 160, device=DEV)[:24] + 16 for _ in range(h)]).sort(-1).values.int()
    for fn in {kernel, bigger_bird_flash_ref}:
        grouped = fn(q, k, v, idx[:, None], front=16, window=64, num_heads=h, kv_groups=g, scale=0.2)
        repeated = fn(q, k.repeat_interleave(g, 0), v.repeat_interleave(g, 0), idx[:, None],
                      front=16, window=64, num_heads=h, scale=0.2)
        torch.testing.assert_close(grouped, repeated, atol=2e-4 if DEV == "cpu" else 2e-2, rtol=1e-3)


def test_ref_kernel_matches_independent_union_reference():
    torch.manual_seed(2)
    h, n, d, front, window = 4, 256, 32, 8, 48
    q, k, v = (torch.randn(h, n, d, device=DEV) for _ in range(3))  # FP32: reference-only check
    idx = torch.stack([torch.randperm(n - window - front, device=DEV)[:20] + front for _ in range(h)]).sort(-1).values.int()
    pos = torch.arange(n, device=DEV)
    allowed = (pos[None] <= pos[:, None]) & ((pos[None] < front) | (pos[None] > pos[:, None] - window))
    routed = torch.zeros(h, n, dtype=torch.bool, device=DEV).scatter_(1, idx.long(), True)
    allowed = allowed[None] | (routed[:, None] & (pos[None] <= pos[:, None])[None])
    p = (q @ k.transpose(-1, -2)).masked_fill(~allowed, -float("inf")).softmax(-1)
    torch.testing.assert_close(bigger_bird_flash_ref(q, k, v, idx[:, None], front=front, window=window,
                                                     num_heads=h), p @ v, atol=1e-4, rtol=1e-4)


def test_tail_as_window_n_equals_dense_causal_rows():
    torch.manual_seed(3)
    h, g, n, d, r = 8, 4, 300, 32, 37
    q = torch.randn(h, n, d, device=DEV, dtype=DT)
    k, v = torch.randn(h // g, n, d, device=DEV, dtype=DT), torch.randn(h // g, n, d, device=DEV, dtype=DT)
    none = torch.full((h, 1, 1), -1, dtype=torch.int32, device=DEV)
    for fn in {kernel, bigger_bird_flash_ref}:
        tail = fn(q[:, n - r:], k, v, none, front=0, window=n, num_heads=h, kv_groups=g, scale=0.3)
        qp = torch.arange(n - r, n, device=DEV)[:, None]
        mask = torch.arange(n, device=DEV)[None] <= qp
        want = F.scaled_dot_product_attention(q[None, :, n - r:], k[None], v[None], attn_mask=mask,
                                              scale=0.3, enable_gqa=True)[0]
        torch.testing.assert_close(tail, want, atol=2e-4 if DEV == "cpu" else 2e-2, rtol=1e-3)


CONFIGS = [dict(route_pool=p, route_queries=rq, exact_tail=t, vertical_scope=vs)
           for p, rq, t, vs in itertools.product(["mean", "max"], ["last", "spread"], [0, 24, 96], [0, 128])]


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: "-".join(f"{k[:6]}={v}" for k, v in c.items()))
def test_new_matches_old_logits_and_routes(cfg):
    kw = dict(sink=8, window=96, vertical=24, n_route=16, scope_chunk=64, block_m=16, block_n=16, **cfg)
    torch.manual_seed(4)
    ids = torch.randint(0, 512, (1, 400), device=DEV)
    model = tiny_llama()
    old_attn.install(model, "vs", **kw)
    want = prefill_logits(model, ids)
    old_routes = [layer.self_attn.last_routes.clone() for layer in model.model.layers]
    new_attn.install(model, "vs", **kw)
    got = prefill_logits(model, ids)
    new_routes = [layer.self_attn.last_routes for layer in model.model.layers]
    for a, b in zip(old_routes, new_routes):
        if DEV == "cpu":
            assert torch.equal(a, b), "routed columns changed"
        else:
            # BF16 route scores round differently (grouped vs repeated matmul): allow near-tie flips.
            overlap = min(len(set(x.tolist()) & set(y.tolist())) / a.shape[-1] for x, y in zip(a, b))
            assert overlap >= 0.9, f"routed columns changed beyond BF16 tie flips (overlap {overlap:.2f})"
    torch.testing.assert_close(got, want, atol=2e-4 if DEV == "cpu" else 5e-2, rtol=1e-3)


class AllocLog(TorchDispatchMode):
    """Bytes of every aten op output (allocation upper bound, ignores frees)."""

    def __init__(self):
        super().__init__()
        self.records = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in torch.utils._pytree.tree_leaves(out):
            if isinstance(t, torch.Tensor) and t.numel() * t.element_size() >= 1 << 16:
                self.records.append((t.numel() * t.element_size(), str(func)))
        return out


def alloc_report(module):
    """Run one sparse prefill of a single layer with the flash kernel stubbed out."""
    n = 2048
    model = tiny_llama(torch.bfloat16)
    cfg = model.config
    kw = dict(sink=16, window=256, vertical=64, n_route=32, exact_tail=64, block_m=16, block_n=16)
    module.install(model, "vs", **kw)
    attn = model.model.layers[0].self_attn
    hd = attn.head_dim
    q = torch.randn(1, cfg.num_attention_heads, n, hd, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, cfg.num_key_value_heads, n, hd, device=DEV, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    real = module.bigger_bird_flash
    module.bigger_bird_flash = lambda Q, *a, **kw_: torch.zeros(Q.shape, dtype=Q.dtype, device=Q.device)
    try:
        with torch.inference_mode(), AllocLog() as log:
            attn.sparse_prefill(q, k, v)
    finally:
        module.bigger_bird_flash = real
    return log.records


VIEW_OPS = ("view", "unsqueeze", "squeeze", "transpose", "permute", "expand", "select", "slice", "alias", "t.default")
COPY_OPS = ("repeat_interleave", "index_select", "clone", "contiguous", "_to_copy")


def real_allocs(records):
    """Drop views and the stubbed kernel's output (aten.zeros); keep ops that allocate."""
    return [(b, f) for b, f in records if not any(f".{v}" in f or f"aten.{v}" in f for v in VIEW_OPS)
            and "aten.zeros" not in f]


def test_sparse_prefill_allocates_less_than_before():
    old, new = real_allocs(alloc_report(old_attn)), real_allocs(alloc_report(new_attn))
    by_op = lambda recs: {f: sum(b for b, g in recs if g == f) / 1e6 for f in {g for _, g in recs}}
    print("\nMB by aten op, old:", dict(sorted(by_op(old).items(), key=lambda x: -x[1])[:8]))
    print("MB by aten op, new:", dict(sorted(by_op(new).items(), key=lambda x: -x[1])[:8]))
    print(f"total MB old={sum(b for b, _ in old)/1e6:.1f} new={sum(b for b, _ in new)/1e6:.1f}")
    h_wide = 16 * 2048 * 32 * 2                            # one [H, n, d] bf16 tensor (4 KV heads -> 1/4 of it)
    big_copies = [(b, f) for b, f in new if b >= h_wide // 4 and any(c in f for c in COPY_OPS)]
    assert not big_copies, f"K/V-sized copies still made: {big_copies}"
    assert sum(b for b, _ in new) < sum(b for b, _ in old)
