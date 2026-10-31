"""Single-layer decode benchmark for the compressed (TurboQuant-style) KV cache kernel; no model.

Per context length: BF16 SDPA over an uncompressed cache vs kernels/turbo_decode.py ("dot" and "fma" variants,
plus the auto choice), CUDA-graph GPU time, 32 query heads / 8 KV heads / head dim 128, 4-bit codes.

  python -m experiments.exp_21_kascade.bench_turbo --n 131072,524288,1048576
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.exp_21_kascade.bench_decode import graphed
from experiments.exp_21_kascade.common import OUT
from kernels import turbo_decode as td
from modal_rd.turbo_kv import TurboQuantizer


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", default="131072,524288,1048576")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", default=str(OUT / "phase4" / "bench_turbo_h100.json"))
    a = p.parse_args(argv)
    tq = TurboQuantizer(128, a.bits, "cuda")
    rows = []
    for n in map(int, a.n.split(",")):
        torch.manual_seed(0)
        k = torch.randn(8, n, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        (kc, kn), (vc, vn) = tq.encode(k), tq.encode(v)
        q = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        r = {"sdpa_bf16": [], "dot": [], "fma": []}
        for _ in range(a.reps):
            r["sdpa_bf16"].append(graphed(lambda: F.scaled_dot_product_attention(q[None, :, None], k[None], v[None],
                                                                                enable_gqa=True), 20))
            for var in ("dot", "fma"):
                r[var].append(graphed(lambda: td.turbo_decode(q, kc, kn, vc, vn, tq.c_list, tq.rot, scale=128 ** -0.5,
                                                              variant=var), 20))
        td.turbo_decode(q, kc, kn, vc, vn, tq.c_list, tq.rot, scale=128 ** -0.5)        # auto choice
        row = {"n": n, **{name: statistics.median(t) for name, t in r.items()}, "auto": dict(td._BEST).popitem()[1],
               "bf16_bytes_gb": 2 * k.numel() * 2 / 1e9, "turbo_bytes_gb": 2 * (kc.numel() + kn.numel() * 2) / 1e9}
        rows.append(row)
        print("TURBO BENCH", json.dumps(row), flush=True)
        del k, v, kc, kn, vc, vn
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "bits": a.bits, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
