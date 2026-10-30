"""Attention-only timing per layer (routing + sparse kernel + exact tail) vs dense FlashAttention.

Synthetic BF16 Q/K/V with Llama-8B shapes (32 query heads, 8 KV heads, d=128), so it can
go past the model's 128K position limit. Not a model-quality result.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from modal_rd.attention import RDAttention


def timed(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / 1000)
    return min(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqs", default="32768,65536,131072,262144,524288,1048576")
    p.add_argument("--attn", default='{"sink":64,"window":16384,"vertical":512,"exact_tail":256}')
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    base = SimpleNamespace(config=SimpleNamespace(num_attention_heads=32, num_key_value_heads=8), layer_idx=0,
                           head_dim=128, scaling=128 ** -0.5, q_proj=None, k_proj=None, v_proj=None, o_proj=None)
    attn = RDAttention(base, **json.loads(a.attn))
    rows = []
    for n in map(int, a.seqs.split(",")):
        torch.manual_seed(0)
        q = torch.randn(1, 32, n, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 8, n, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 8, n, 128, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            dense = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=base.scaling, enable_gqa=True))
            sparse = timed(lambda: attn.sparse_prefill(q, k, v))
        rows.append({"tokens": n, "dense_s": dense, "sparse_s": sparse, "speedup": dense / sparse})
        print(f"BENCH tokens={n} dense={dense * 1000:.1f}ms sparse={sparse * 1000:.1f}ms speedup={dense / sparse:.2f}x", flush=True)
        del q, k, v
        torch.cuda.empty_cache()
        out = Path("/vol/results") / f"{a.tag}.json"
        out.write_text(json.dumps({"attn": json.loads(a.attn), "gpu": torch.cuda.get_device_name(0),
                                   "per_layer": True, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
