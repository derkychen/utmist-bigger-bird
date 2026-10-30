"""Single-layer decode attention microbenchmark + parity check (random tensors, no model).

One new token, Llama-8B shapes (32 query heads, 8 KV heads, head dim 128), BF16 cache [Hkv, n, D]:
  dense     SDPA over the whole cache (v3 decode)
  torch     exp 21 PyTorch sparse decode: gather sink + window + routed, then attend (pooled routes)
  bbflash   kernels/bigger_bird_flash.py called with one query (no gather; one program per head)
  bbdecode  kernels/bigger_bird_decode.py (split-K, pooled routes read once per KV group)
  anchor    the anchor layers' extra routing step in PyTorch: score every cached key, pool, top-k
  route     the same step fused (kernels/bigger_bird_route.py)
Max error is against fp32 attention over the same sparse key set. Times are per call, launched from
Python ("ms") and replayed from a CUDA graph ("graph", GPU time without Python launch overhead).

  python -m experiments.exp_21_kascade.bench_decode --n 32768,131072 --window 1024 --routed 1024
"""
import argparse
import json

import torch
import torch.nn.functional as F

from kernels.bigger_bird_decode import bigger_bird_decode
from kernels.bigger_bird_flash import bigger_bird_flash
from kernels.bigger_bird_route import bigger_bird_route

H, HK, D, SINK = 32, 8, 128, 64
G = H // HK


def sparse_positions(n, window, idx):
    return torch.cat([torch.arange(SINK, device="cuda").expand(HK, -1),
                      torch.arange(n - window, n, device="cuda").expand(HK, -1), idx], 1)       # [HK, S]


def reference(q, k, v, n, window, idx, scale):
    pos = sparse_positions(n, window, idx)
    heads = torch.arange(HK, device="cuda")[:, None]
    ks, vs = k[heads, pos].float(), v[heads, pos].float()                                       # [HK, S, D]
    att = (q.view(HK, G, D).float() @ ks.transpose(1, 2) * scale).softmax(-1)
    return (att @ vs).reshape(H, D)


def torch_sparse(q, k, v, n, window, idx, scale):
    """exp 21 sparse_decode.sparse_decode_pooled without the routing step."""
    pos = sparse_positions(n, window, idx)
    heads = torch.arange(HK, device="cuda")[:, None]
    ks, vs = k[heads, pos], v[heads, pos]
    att = (q.view(HK, G, D).float() @ ks.float().transpose(1, 2) * scale).softmax(-1)
    return (att @ vs.float()).to(q.dtype).reshape(H, D)


def anchor_routes(q, k, n, window, routed, scale):
    p = (q.view(HK, G, D) @ k.transpose(1, 2) * scale).float().softmax(-1).mean(1)
    p[:, :SINK] = -1
    p[:, n - window:] = -1
    return p.topk(routed, dim=-1).indices


def timed(fn, iters):
    for _ in range(3):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def graphed(fn, iters):
    """Time `fn` replayed from a CUDA graph (GPU time without Python launch overhead)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return timed(g.replay, iters)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", default="32768,131072")
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--routed", type=int, default=1024)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    scale = D ** -0.5
    rows = []
    for n in map(int, a.n.split(",")):
        torch.manual_seed(0)
        # cache as HF stores it ([1, Hkv, n, D] -> view [Hkv, n, D]); large-ish values make softmax peaky
        k = torch.randn(1, HK, n, D, device="cuda", dtype=torch.bfloat16)[0] * 2
        v = torch.randn(1, HK, n, D, device="cuda", dtype=torch.bfloat16)[0]
        q = torch.randn(H, D, device="cuda", dtype=torch.bfloat16)
        idx = torch.stack([SINK + torch.randperm(n - a.window - SINK, device="cuda")[:a.routed] for _ in range(HK)])
        ref = reference(q, k, v, n, a.window, idx, scale)
        idx_h = idx.repeat_interleave(G, 0).to(torch.int32)[:, None, :]                      # [H, 1, R]
        arms = {
            "dense": lambda: F.scaled_dot_product_attention(q[None, :, None], k[None], v[None], scale=scale,
                                                            enable_gqa=True)[0, :, 0],
            "torch": lambda: torch_sparse(q, k, v, n, a.window, idx, scale),
            "bbflash": lambda: bigger_bird_flash(q[:, None], k, v, idx_h, front=SINK, window=a.window,
                                                 num_heads=H, kv_groups=G, scale=scale)[:, 0],
            "bbdecode": lambda: bigger_bird_decode(q, k, v, idx, front=SINK, window=a.window, scale=scale),
            "anchor": lambda: anchor_routes(q, k, n, a.window, a.routed, scale),
            "route": lambda: bigger_bird_route(q, k, a.routed, front=SINK, window=a.window, scale=scale),
        }
        row = {"n": n, "window": a.window, "routed": a.routed}
        for name, fn in arms.items():
            out = fn()
            err = None if name in ("dense", "anchor", "route") else (out.float() - ref).abs().max().item()
            row[name] = {"ms": timed(fn, a.iters), "graph_ms": graphed(fn, a.iters), "max_err": err}
        rows.append(row)
        print(f"n={n:>7} " + "  ".join(f"{k}={v['ms']:.3f}/{v['graph_ms']:.3f}ms"
                                       + (f"(err {v['max_err']:.0e})" if v['max_err'] is not None else "")
                                       for k, v in row.items() if isinstance(v, dict)), flush=True)
        del k, v
        torch.cuda.empty_cache()
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"gpu": torch.cuda.get_device_name(0), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
