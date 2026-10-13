"""Phase 0: record v3's routed columns at every layer (no behaviour change).

Runs Bigger Bird v3 (`modal_rd.attention`, arm "vs") with every layer routing as
usual and saves each layer's per-head routed key columns plus the needle spans.
`analyze_overlap.py` then measures how similar the selections are across layers,
i.e. whether Kascade-style anchor-layer reuse could work.

  python -m experiments.exp_21_kascade.probe_overlap --bytes 64000 --n 6 --tag p0_16k
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.exp_21_kascade import model as kascade
from experiments.exp_21_kascade.common import exact_number, greedy, load_model_4bit
from modal_rd import attention as rd
from modal_rd.niah import load_tokenizer, prepare

OUT = Path(__file__).parents[2] / "benchmarks" / "exp_21_kascade" / "phase0"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bytes", type=int, default=64000, help="haystack bytes (~4 bytes/token)")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--depths", default="0.5")
    p.add_argument("--attn", default='{"sink":64,"window":4096,"vertical":512,"exact_tail":256}')
    p.add_argument("--pool", action="store_true", help="record KV-head-pooled routes (Kascade GQA pooling)")
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    attn = json.loads(a.attn)
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer()
    model = load_model_4bit()
    if a.pool:
        kascade.install(model, list(range(len(model.model.layers))), pool=True, **attn)
    else:
        rd.install(model, "vs", **attn)
    summary = {"pool": a.pool, "attn": attn, "bytes": a.bytes, "gpu": torch.cuda.get_device_name(0), "quant": "nf4", "runs": []}
    for depth in map(float, a.depths.split(",")):
        for row in prepare(tok, a.bytes, a.n, depth=depth, verify=False):
            gen, peak = greedy(model, tok, row["text"])
            routes = np.stack([layer.self_attn.last_routes.cpu().numpy() for layer in model.model.layers])
            name = f"d{depth}_i{row['idx']}"
            np.savez_compressed(out / f"{name}.npz", routes=routes, number=row["spans"]["number"],
                                needle=row["spans"]["needle"], tokens=row["tokens"])
            run = {"name": name, "idx": row["idx"], "depth": depth, "tokens": row["tokens"], "answer": row["answer"],
                   "generated": gen, "exact": exact_number(gen, row["answer"]), "peak_gb": peak}
            summary["runs"].append(run)
            print(json.dumps(run), flush=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
