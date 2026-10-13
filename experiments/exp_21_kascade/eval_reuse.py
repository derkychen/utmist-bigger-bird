"""Phase 2a: NIAH accuracy of anchor-layer route reuse vs v3 (all layers route).

Anchors come from `analyze_overlap.py` on the tuning prompts (idx 0-5); evaluate on
held-out prompts (default idx 6-11). Also reports the time spent inside routes()
(CUDA-synchronised) vs the whole prefill+decode, to size the possible speedup.

  python -m experiments.exp_21_kascade.eval_reuse --sets "all;0;0,11;0,7,13,23" --start 6 --n 6 --tag p2a_16k
  python -m experiments.exp_21_kascade.eval_reuse --task niah_multikey_1 --sets "dense;all;0,11" --max-new 32 --tag x

Flags after the anchor list (e.g. "0,2,8,13,14:remap:pool"): "remap" uses Kascade head
remapping built from the Phase 0 run given by --map-tag; "pool" shares one column set per
KV head (Kascade GQA pooling). Arm "dense" = exact attention everywhere (v3 wrapper, dense_rd). Multi-value tasks are
scored like modal_rd.eval_niah: exact only if every gold value appears in the output.
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch

from experiments.exp_21_kascade import model as kascade
from experiments.exp_21_kascade.common import exact_number, greedy, load_model_4bit
from modal_rd import attention as rd
from modal_rd.niah import load_tokenizer, prepare_task

OUT = Path(__file__).parents[2] / "benchmarks" / "exp_21_kascade" / "phase2a"


def score(gen, row):
    if "answers" in row:
        flat = re.sub(r"\s", "", gen)
        hits = [x in flat for x in row["answers"]]
        return all(hits), sum(hits) / len(hits)
    ok = exact_number(gen, row["answer"])
    return ok, float(ok)


def time_routes(model, clock):
    """Wrap every layer's routes() to accumulate CUDA-synchronised seconds into clock[0]."""
    for layer in model.model.layers:
        f = layer.self_attn.routes
        def timed(q, k, _f=f):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = _f(q, k)
            torch.cuda.synchronize(); clock[0] += time.perf_counter() - t0
            return out
        layer.self_attn.routes = timed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", required=True, help='";"-separated anchor lists, "all" = every layer (plain v3)')
    p.add_argument("--bytes", type=int, default=64000)
    p.add_argument("--start", type=int, default=6)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--depths", default="0.5")
    p.add_argument("--task", default="niah", help="RULER NIAH variant, e.g. niah_multikey_1, niah_multivalue")
    p.add_argument("--max-new", type=int, default=10)
    p.add_argument("--map-tag", default="p0_16k_tail0", help="Phase 0 run whose overlaps build the head map")
    p.add_argument("--attn", default='{"sink":64,"window":4096,"vertical":512,"exact_tail":256}')
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    attn = json.loads(a.attn)
    OUT.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer()
    model = load_model_4bit()
    L = len(model.model.layers)
    rows = [r for d in map(float, a.depths.split(","))
            for r in prepare_task(tok, a.task, a.bytes, a.n, depth=d, start=a.start)[0]]
    report = {"task": a.task, "max_new": a.max_new, "attn": attn, "bytes": a.bytes, "gpu": torch.cuda.get_device_name(0), "quant": "nf4", "arms": []}
    for spec in a.sets.split(";"):
        if spec == "dense":
            anchors = []
            rd.install(model, "dense_rd", **attn)
        else:
            layers, *flags = spec.split(":")
            anchors = list(range(L)) if layers == "all" else [int(x) for x in layers.split(",")]
            hmap = kascade.load_head_map(a.map_tag, anchors) if "remap" in flags else None
            kascade.install(model, anchors, head_map=hmap, pool="pool" in flags, **attn)
        clock = [0.0]
        time_routes(model, clock)
        runs = []
        for row in rows:
            clock[0] = 0.0
            torch.cuda.synchronize(); t0 = time.perf_counter()
            gen, peak = greedy(model, tok, row["text"], a.max_new)
            torch.cuda.synchronize(); total = time.perf_counter() - t0
            exact, partial = score(gen, row)
            runs.append({"idx": row["idx"], "depth": row["depth"], "tokens": row["tokens"],
                         "answers": row.get("answers", [row["answer"]]), "generated": gen, "exact": exact,
                         "partial": partial, "peak_gb": peak,
                         "total_s": total, "routes_s": clock[0]})
            print(f"anchors={spec} idx={row['idx']} d={row['depth']} exact={runs[-1]['exact']} total={total:.1f}s "
                  f"routes={clock[0]:.2f}s gen={gen!r}", flush=True)
        arm = {"spec": spec, "anchors": anchors, "exact": sum(r["exact"] for r in runs), "n": len(runs),
               "partial": sum(r["partial"] for r in runs) / len(runs),
               "routes_s_mean": sum(r["routes_s"] for r in runs) / len(runs),
               "total_s_mean": sum(r["total_s"] for r in runs) / len(runs), "runs": runs}
        report["arms"].append(arm)
        print(f"== {a.task} anchors={spec}: {arm['exact']}/{arm['n']} partial={arm['partial']:.2f} routes {arm['routes_s_mean']:.2f}s of {arm['total_s_mean']:.1f}s", flush=True)
        (OUT / f"{a.tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
