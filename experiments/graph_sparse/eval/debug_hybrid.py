"""Debug evaluation for hybrid QK + semantic graph routing.

Sweeps 6 arms from pure QK (512/0) through pure graph (0/512) with
needle coverage and hybrid diagnostics at each point.

Usage (single GPU):
    python -m experiments.graph_sparse.eval.debug_hybrid \
        --bytes 57600 --n 1

    python -m experiments.graph_sparse.eval.debug_hybrid \
        --bytes 57600,230400,512000 --n 5

Usage (Modal):
    modal run modal_rd/app.py --module experiments.graph_sparse.eval.debug_hybrid \
        --args "--bytes 57600,230400,512000 --n 5"
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from modal_rd.niah import load_tokenizer, prepare_task
from modal_rd.eval_niah import build, run_one

from experiments.graph_sparse.eval.arms import STAGE_1C, BB_V3_MATCHED
from experiments.graph_sparse.attention import GraphSparseAttention


def _locate_needle(tokenizer, row):
    text = row["text"]
    answer = row["answer"]
    pos = text.find(answer)
    if pos < 0:
        return None
    prefix_ids = tokenizer(text[:pos], add_special_tokens=False)["input_ids"]
    span_ids = tokenizer(text[:pos + len(answer)], add_special_tokens=False)["input_ids"]
    lo = len(prefix_ids)
    hi = len(span_ids)
    return (lo, hi) if lo < hi else None


def _set_debug_config(model, debug_cfg):
    for layer in model.model.layers:
        a = layer.self_attn
        if isinstance(a, GraphSparseAttention):
            a._debug_config = dict(debug_cfg)


def _collect_coverage(model):
    results = []
    for layer in model.model.layers:
        a = layer.self_attn
        if isinstance(a, GraphSparseAttention) and a._last_coverage is not None:
            results.append(a._last_coverage)
    return results


def _collect_hybrid_diag(model):
    results = []
    for layer in model.model.layers:
        a = layer.self_attn
        if isinstance(a, GraphSparseAttention) and hasattr(a, "_last_hybrid_diag"):
            diag = getattr(a, "_last_hybrid_diag", None)
            if diag is not None:
                results.append(diag)
    return results


def _build_model(arm_def):
    m = build(arm_def["arm"], arm_def.get("attn", {}),
              dense_layers=[], graph_config=arm_def.get("graph_config"))
    debug = arm_def.get("debug", {})
    if debug:
        _set_debug_config(m, debug)
    return m


def _generate(model, tokenizer, row, max_new=10):
    import re
    inputs = tokenizer(row["text"], return_tensors="pt", truncation=False).to("cuda")
    n_tok = inputs["input_ids"].shape[1]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                                              SDPBackend.EFFICIENT_ATTENTION,
                                              SDPBackend.MATH]):
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             use_cache=True, pad_token_id=tokenizer.pad_token_id,
                             return_dict_in_generate=True, output_logits=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    gen_ids = out.sequences[0, n_tok:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    first = re.search(r"\d[\d,]*", gen_text)
    answer_str = first[0].replace(",", "") if first else ""
    exact = answer_str == row["answer"]
    return {
        "generated": gen_text,
        "gen_ids": gen_ids,
        "exact": exact,
        "tokens": n_tok,
        "wall_s": wall,
        "peak_gb": peak_gb,
    }


# BB v3 matched baseline arm
ARM_BASELINE = {
    "name": "bb_v3_matched",
    "arm": "vs",
    "attn": BB_V3_MATCHED,
    "graph_config": None,
    "description": "BB v3 matched (sink64 + window4K + vert512 + tail256).",
}


def main():
    p = argparse.ArgumentParser(description="Hybrid QK+graph routing ablation")
    p.add_argument("--bytes", default="57600",
                   help="Comma-separated byte lengths")
    p.add_argument("--n", type=int, default=1,
                   help="Examples per length")
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--out", default=None,
                   help="JSON output path")
    a = p.parse_args()

    bytes_list = [int(x) for x in a.bytes.split(",")]
    tokenizer = load_tokenizer()

    all_arms = [ARM_BASELINE] + STAGE_1C

    print("=" * 72)
    print("HYBRID QK + GRAPH ROUTING ABLATION")
    print("=" * 72)
    print(f"Arms: {len(all_arms)}")
    for arm in all_arms:
        print(f"  {arm['name']}: {arm['description']}")
    print(f"Bytes: {bytes_list}, n={a.n}, depth={a.depth}")
    print()

    results = []
    for nbytes in bytes_list:
        rows, _ = prepare_task(tokenizer, "niah", nbytes, a.n, depth=a.depth)
        for row in rows:
            needle_span = _locate_needle(tokenizer, row)
            print(f"\n--- bytes={nbytes} idx={row['idx']} tokens={row['tokens']} "
                  f"needle={needle_span} ---")

            for arm_def in all_arms:
                name = arm_def["name"]
                print(f"  [{name}]")
                model = _build_model(arm_def)

                if needle_span and arm_def["arm"] == "graph":
                    debug = dict(arm_def.get("debug", {}))
                    debug["track_coverage"] = True
                    debug["needle_span"] = needle_span
                    _set_debug_config(model, debug)

                res = _generate(model, tokenizer, row)

                coverage = _collect_coverage(model) if arm_def["arm"] == "graph" else []
                hybrid_diag = _collect_hybrid_diag(model) if arm_def["arm"] == "graph" else []
                del model; torch.cuda.empty_cache()

                final_hit_layers = sum(1 for c in coverage if c["final_chunk_hit"]) if coverage else None
                total_layers = len(coverage) if coverage else None

                entry = {
                    "arm": name, "bytes": nbytes, "idx": row["idx"],
                    "tokens": res["tokens"], "exact": res["exact"],
                    "generated": res["generated"],
                    "wall_s": res["wall_s"], "peak_gb": res["peak_gb"],
                    "needle_span": needle_span,
                    "final_chunk_needle_layers": final_hit_layers,
                    "total_layers": total_layers,
                }

                if hybrid_diag:
                    entry["qk_ms_mean"] = sum(d.get("qk_ms", 0) for d in hybrid_diag) / len(hybrid_diag)
                    entry["graph_ms_mean"] = sum(d.get("graph_ms", 0) for d in hybrid_diag) / len(hybrid_diag)
                    entry["overlap_mean"] = sum(d.get("overlap_mean", 0) for d in hybrid_diag if "overlap_mean" in d) / max(1, sum(1 for d in hybrid_diag if "overlap_mean" in d))

                results.append(entry)

                needle_info = (f"needle={final_hit_layers}/{total_layers}"
                               if final_hit_layers is not None else "")
                print(f"    exact={'YES' if res['exact'] else 'NO'}  "
                      f"gen={res['generated']!r}  "
                      f"wall={res['wall_s']:.1f}s  peak={res['peak_gb']:.1f}GB  "
                      f"{needle_info}")

    # Summary table
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Arm':<22s} {'Bytes':>8s} {'Tok':>7s} {'Exact':>6s} "
          f"{'Needle':>8s} {'Wall':>6s} {'Peak':>6s}  Generated")
    for r in results:
        nc = (f"{r['final_chunk_needle_layers']}/{r['total_layers']}"
              if r["final_chunk_needle_layers"] is not None else "—")
        print(f"{r['arm']:<22s} {r['bytes']:>8d} {r['tokens']:>7d} "
              f"{'YES' if r['exact'] else 'NO':>6s} "
              f"{nc:>8s} {r['wall_s']:>5.1f}s {r['peak_gb']:>5.1f}G  "
              f"{r['generated']!r}")

    # Per-arm accuracy summary
    print("\n" + "-" * 72)
    print("PER-ARM ACCURACY")
    from collections import defaultdict
    acc = defaultdict(lambda: [0, 0])
    for r in results:
        acc[r["arm"]][0] += r["exact"]
        acc[r["arm"]][1] += 1
    for arm_name, (correct, total) in acc.items():
        print(f"  {arm_name:<22s} {correct}/{total} = {100*correct/max(1,total):.0f}%")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {a.out}")


if __name__ == "__main__":
    main()
