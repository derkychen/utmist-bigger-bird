"""Benchmark causal local + routed attention: PyTorch path vs Triton kernels.

    python -m kernels.bench_routed_window --seqs 32768,131072 --top-k 512,2048,4096
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from kernels import routed_window_attention_fast, routed_window_attention_flash
from sparse_attn_utils import causal_sparse_attention


def _time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / iters * 1000
    return ms, (torch.cuda.max_memory_allocated() - base) / 1e9, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", default="4096,32768,131072")
    parser.add_argument("--top-k", default="512,2048,4096")
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--skip-old", action="store_true", help="skip routed_window_attention_fast")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    groups = args.heads // args.kv_heads
    print(f"GPU: {torch.cuda.get_device_name(0)} capability {torch.cuda.get_device_capability()}")
    rows = []
    for seq in (int(s) for s in args.seqs.split(",")):
        torch.manual_seed(0)
        q = torch.randn(args.heads, seq, args.head_dim, device="cuda", dtype=dtype) * 0.3
        k_kv, v_kv = (
            torch.randn(args.kv_heads, seq, args.head_dim, device="cuda", dtype=dtype) * 0.3
            for _ in range(2)
        )
        k, v = k_kv.repeat_interleave(groups, 0), v_kv.repeat_interleave(groups, 0)
        for top_k in (int(t) for t in args.top_k.split(",")):
            top_k = min(top_k, seq)
            routed = torch.stack([
                torch.randperm(seq, device="cuda")[:top_k] for _ in range(args.heads)
            ])
            row = {"seq": seq, "top_k": top_k}
            variants = {
                "torch": lambda: causal_sparse_attention(
                    q, k, v, routed, local_window=args.window, bsz=1,
                    num_heads=args.heads, dedupe_local=True),
                "flash": lambda: routed_window_attention_flash(
                    q, k, v, routed, args.window, num_heads=args.heads, dedupe_local=True),
                "flash_gqa": lambda: routed_window_attention_flash(
                    q, k_kv, v_kv, routed, args.window, num_heads=args.heads, dedupe_local=True),
            }
            if not args.skip_old:
                variants["old_fast"] = lambda: routed_window_attention_fast(
                    q, k, v, routed, args.window, scale=1.0)
            ref = None
            for name, fn in variants.items():
                try:
                    ms, gb, out = _time(fn, args.warmup, args.iters)
                except torch.cuda.OutOfMemoryError:
                    row[name] = "oom"
                    torch.cuda.empty_cache()
                    continue
                if ref is None:
                    ref = out.float()
                if name == "old_fast":  # no dedupe support; compare to matching semantics
                    ref_old = causal_sparse_attention(
                        q, k, v, routed, local_window=args.window, bsz=1, num_heads=args.heads)
                    err = (out.float() - ref_old.float()).abs().max().item()
                else:
                    err = (out.float() - ref).abs().max().item()
                row[name] = {"ms": round(ms, 2), "peak_extra_gb": round(gb, 3), "max_err_vs_torch": err}
            rows.append(row)
            print(json.dumps(row))
            del routed, ref
        del q, k, v, k_kv, v_kv
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
