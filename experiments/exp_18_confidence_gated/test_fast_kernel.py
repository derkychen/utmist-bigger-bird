#!/usr/bin/env python3
"""Test the fast routed-window kernel against PyTorch reference and dense SDPA."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F


def _event_seconds(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0 / max(1, iters)


def correctness_test(seq_len, batch, heads, head_dim, window, top_k, dtype=torch.float16):
    """Verify fast kernel matches PyTorch causal_sparse_attention."""
    from sparse_attn_utils import (
        effective_top_k,
        last_query_topk_indices,
        causal_sparse_attention,
    )
    from kernels import routed_window_attention_fast, triton_available

    if not triton_available():
        print("  SKIP: Triton not available")
        return False

    device = torch.device("cuda")
    torch.manual_seed(42)
    BH = batch * heads
    q = torch.randn(BH, seq_len, head_dim, device=device, dtype=dtype) * (head_dim ** -0.5)
    k = torch.randn(BH, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(BH, seq_len, head_dim, device=device, dtype=dtype)

    d_low = min(64, head_dim)
    k_eff = effective_top_k(top_k, seq_len, min_k=64, ratio=2)
    indices = last_query_topk_indices(
        q[:, :, :d_low], k[:, :, :d_low], k_eff, None, batch, heads,
    )

    # PyTorch reference
    ref = causal_sparse_attention(
        q, k, v, indices, local_window=window,
        token_mask=None, bsz=batch, num_heads=heads,
    )

    # Fast kernel
    fast = routed_window_attention_fast(q, k, v, indices, window, scale=1.0)

    max_diff = (ref.float() - fast.float()).abs().max().item()
    mean_diff = (ref.float() - fast.float()).abs().mean().item()
    print(f"  correctness: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    return max_diff < 0.05  # allow small fp16 differences


def benchmark(seq_len, batch, heads, head_dim, window, top_k, warmup=3, iters=10):
    """Benchmark fast kernel vs PyTorch reference vs dense SDPA."""
    from sparse_attn_utils import (
        effective_top_k,
        last_query_topk_indices,
        causal_sparse_attention,
    )
    from kernels import routed_window_attention_fast, triton_available

    device = torch.device("cuda")
    dtype = torch.float16
    BH = batch * heads

    torch.manual_seed(42)
    q_raw = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k_raw = torch.randn_like(q_raw)
    v_raw = torch.randn_like(q_raw)
    q_flat = q_raw.reshape(BH, seq_len, head_dim)
    k_flat = k_raw.reshape(BH, seq_len, head_dim)
    v_flat = v_raw.reshape(BH, seq_len, head_dim)

    # Dense SDPA (FlashAttention)
    dense_time = _event_seconds(
        lambda: F.scaled_dot_product_attention(q_raw, k_raw, v_raw, is_causal=True),
        warmup, iters,
    )

    # PyTorch sparse reference
    d_low = min(64, head_dim)
    k_eff = effective_top_k(top_k, seq_len, min_k=64, ratio=2)
    indices = last_query_topk_indices(
        q_flat[:, :, :d_low], k_flat[:, :, :d_low], k_eff, None, batch, heads,
    )

    def run_pytorch():
        return causal_sparse_attention(
            q_flat, k_flat, v_flat, indices, local_window=window,
            token_mask=None, bsz=batch, num_heads=heads,
        )

    pytorch_time = _event_seconds(run_pytorch, warmup, iters)

    # Fast Triton kernel
    fast_time = None
    if triton_available():
        fast_time = _event_seconds(
            lambda: routed_window_attention_fast(q_flat, k_flat, v_flat, indices, window, scale=1.0),
            warmup, iters,
        )

    result = {
        "seq_len": seq_len,
        "dense_seconds": dense_time,
        "pytorch_sparse_seconds": pytorch_time,
        "fast_triton_seconds": fast_time,
        "dense_vs_pytorch": dense_time / max(pytorch_time, 1e-9),
        "dense_vs_fast": dense_time / max(fast_time, 1e-9) if fast_time else None,
        "fast_vs_pytorch": pytorch_time / max(fast_time, 1e-9) if fast_time else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", default="2048,4096,8192,16384,32768,65536,131072")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    seqs = [int(s) for s in args.seqs.split(",") if s]

    # Correctness tests first
    print("=" * 70)
    print("CORRECTNESS TESTS")
    print("=" * 70)
    for seq in [2048, 4096]:
        print(f"\nseq={seq}, window={args.window}, top_k={args.top_k}")
        ok = correctness_test(
            seq, args.batch, args.heads, args.head_dim,
            args.window, args.top_k,
        )
        if not ok:
            print("  CORRECTNESS FAILED — skipping benchmarks")
            return

    # Benchmarks
    print("\n" + "=" * 70)
    print("BENCHMARKS")
    print("=" * 70)
    results = []
    for seq in seqs:
        r = benchmark(
            seq, args.batch, args.heads, args.head_dim,
            args.window, args.top_k, args.warmup, args.iters,
        )
        results.append(r)
        if r['fast_triton_seconds']:
            print(
                f"  seq={seq:>6d} | "
                f"dense={r['dense_seconds']:.6f}s | "
                f"pytorch={r['pytorch_sparse_seconds']:.6f}s | "
                f"fast={r['fast_triton_seconds']:.6f}s | "
                f"dense/fast={r['dense_vs_fast']:.2f}x"
            )
        else:
            print(
                f"  seq={seq:>6d} | "
                f"dense={r['dense_seconds']:.6f}s | "
                f"pytorch={r['pytorch_sparse_seconds']:.6f}s | "
                f"fast=N/A"
            )

    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
