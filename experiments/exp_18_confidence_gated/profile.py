#!/usr/bin/env python3
"""Microbenchmark Exp 18 attention against dense SDPA.

This intentionally benchmarks one attention layer, not the full 8B model. Use
it to identify kernel regressions before expensive RULER runs.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from .attention_core import ConfidenceGatedAttentionCore


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


def benchmark(seq_len: int, args) -> dict:
    B, H, D = args.batch, args.heads, args.head_dim
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    q_raw = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    k = torch.randn_like(q_raw)
    v = torch.randn_like(q_raw)
    q_scaled = q_raw.reshape(B * H, seq_len, D) * (D**-0.5)
    k_flat = k.reshape(B * H, seq_len, D)
    v_flat = v.reshape(B * H, seq_len, D)

    dense_time = _event_seconds(
        lambda: F.scaled_dot_product_attention(q_raw, k, v, is_causal=True),
        args.warmup,
        args.iters,
    )
    torch.cuda.reset_peak_memory_stats()
    core = ConfidenceGatedAttentionCore(
        top_k=args.top_k,
        low_rank_dim=args.low_rank_dim,
        window_size=args.window_size,
        gate_threshold=args.gate_threshold,
        peak_threshold=args.peak_threshold,
        linear_weight=args.linear_weight,
        use_triton=True,
        always_global=args.always_global,
    )

    def run_core():
        return core(
            q_scaled,
            k_flat,
            v_flat,
            None,
            B,
            H,
            is_causal=True,
            training=False,
        )

    sparse_time = _event_seconds(run_core, args.warmup, args.iters)
    stats = core.last_stats
    result = {
        "seq_len": seq_len,
        "dense_seconds": dense_time,
        "exp18_seconds": sparse_time,
        "speedup_vs_dense": dense_time / max(sparse_time, 1e-9),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "active_fraction": stats.active_fraction,
        "margin_mean": stats.margin_mean,
        "margin_max": stats.margin_max,
        "peakiness_mean": stats.peakiness_mean,
        "peakiness_max": stats.peakiness_max,
        "route_k": stats.route_k,
        "used_triton_local": stats.used_triton_local,
        "used_triton_linear": stats.used_triton_linear,
        "used_triton_global": stats.used_triton_global,
        "kernel_failures": stats.kernel_failures,
    }

    if args.train_micro:
        q_train = q_scaled.detach().clone().requires_grad_(True)
        k_train = k_flat.detach().clone().requires_grad_(True)
        v_train = v_flat.detach().clone().requires_grad_(True)
        train_core = ConfidenceGatedAttentionCore(
            top_k=args.top_k,
            low_rank_dim=args.low_rank_dim,
            window_size=args.window_size,
            gate_threshold=args.gate_threshold,
            peak_threshold=args.peak_threshold,
            linear_weight=args.linear_weight,
            use_triton=True,
            always_global=args.always_global,
        )

        def train_step():
            if q_train.grad is not None:
                q_train.grad = None
            if k_train.grad is not None:
                k_train.grad = None
            if v_train.grad is not None:
                v_train.grad = None
            out = train_core(
                q_train,
                k_train,
                v_train,
                None,
                B,
                H,
                is_causal=True,
                training=True,
            )
            out.float().square().mean().backward()

        result["exp18_train_step_seconds"] = _event_seconds(
            train_step, max(0, args.warmup - 1), max(1, args.train_iters)
        )
        result["train_active_fraction"] = train_core.last_stats.active_fraction
        del q_train, k_train, v_train, train_core

    del q_raw, q_scaled, k, k_flat, v, v_flat, core
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", default="2048,4096,8192")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--low-rank-dim", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--peak-threshold", type=float, default=-1.0)
    parser.add_argument("--linear-weight", type=float, default=0.5)
    parser.add_argument("--always-global", action="store_true", default=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--train-micro", action="store_true")
    parser.add_argument("--train-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    rows = [benchmark(int(seq), args) for seq in args.seqs.split(",") if seq]
    print(json.dumps(rows, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
