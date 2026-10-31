"""Debug integration tests for graph-sparse attention.

Diagnoses why Stage 1 graph routing scores 0/5 on NIAH.

Tasks:
  verify    — inject BB v3 QK routes into the graph kernel path; compare
               logits/tokens to the real BB v3 matched arm.
  coverage  — instrument needle coverage per query chunk, head, and layer
               for graph routing vs BB v3 routing.
  ablation  — five-arm comparison (A-E) at 16K/32K/64K/128K tokens.

Usage (single GPU):
    python -m experiments.graph_sparse.eval.debug_integration \
        --task verify --bytes 57600 --n 1

    python -m experiments.graph_sparse.eval.debug_integration \
        --task coverage --bytes 57600,115200,230400,512000 --n 1

    python -m experiments.graph_sparse.eval.debug_integration \
        --task ablation --bytes 57600,115200,230400,512000 --n 1
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from modal_rd.niah import MODEL_PATH, load_tokenizer, prepare_task
from modal_rd.eval_niah import build, run_one

from experiments.graph_sparse.eval.arms import BB_V3_MATCHED, GRAPH_BASE, _graph
from experiments.graph_sparse.config import GraphConfig, SemanticConfig, RewireConfig, WalkConfig, MemoryLayoutConfig


# ── Arm definitions for ablations ─────────────────────────────────────

def _build_graph_config(overrides=None, sem_overrides=None):
    """Convenience: build a graph config dict with the standard base."""
    return _graph(overrides=overrides, sem_overrides=sem_overrides)


ARM_A = {
    "name": "A_bb_v3_matched",
    "arm": "vs",
    "attn": BB_V3_MATCHED,
    "graph_config": None,
    "description": "BB v3 matched (sink64 + window4K + vert512 + tail256).",
}

ARM_B = {
    "name": "B_graph_inject_bb",
    "arm": "graph",
    "attn": {},
    "graph_config": _build_graph_config(),
    "debug": {"route_mode": "bb_v3", "vertical": 512, "n_route": 64},
    "description": "Graph kernel with injected BB v3 QK routes.",
}

ARM_C = {
    "name": "C_graph_semantic",
    "arm": "graph",
    "attn": {},
    "graph_config": _build_graph_config(),
    "description": "Graph semantic routing (current default, 8 blocks).",
}

ARM_D = {
    "name": "D_graph_question",
    "arm": "graph",
    "attn": {},
    "graph_config": _build_graph_config(
        sem_overrides={"question_conditioned": True, "question_tokens": 64},
    ),
    "description": "Graph question-conditioned routing (8 blocks).",
}

ARM_E = {
    "name": "E_graph_budget_32",
    "arm": "graph",
    "attn": {},
    "graph_config": _build_graph_config(
        overrides={"routed_blocks": 32},
        sem_overrides={"near_blocks": 12, "medium_blocks": 8,
                       "far_blocks": 8, "random_blocks": 4},
    ),
    "description": "Graph semantic routing, 32 blocks (2048 routed tokens).",
}

ALL_ABLATION_ARMS = [ARM_A, ARM_B, ARM_C, ARM_D, ARM_E]


# ── Utilities ─────────────────────────────────────────────────────────

def _locate_needle(tokenizer, row):
    """Find the token span [lo, hi) of the needle answer in the tokenized prompt."""
    text = row["text"]
    answer = row["answer"]
    pos = text.find(answer)
    if pos < 0:
        return None
    prefix_ids = tokenizer(text[:pos], add_special_tokens=False)["input_ids"]
    span_ids = tokenizer(text[:pos + len(answer)], add_special_tokens=False)["input_ids"]
    lo = len(prefix_ids)
    hi = len(span_ids)
    if lo >= hi:
        return None
    return (lo, hi)


def _set_debug_config(model, debug_cfg):
    """Set _debug_config on all GraphSparseAttention layers."""
    from experiments.graph_sparse.attention import GraphSparseAttention
    for layer in model.model.layers:
        a = layer.self_attn
        if isinstance(a, GraphSparseAttention):
            a._debug_config = dict(debug_cfg)


def _collect_coverage(model):
    """Collect _last_coverage from all GraphSparseAttention layers."""
    from experiments.graph_sparse.attention import GraphSparseAttention
    results = []
    for layer in model.model.layers:
        a = layer.self_attn
        if isinstance(a, GraphSparseAttention) and a._last_coverage is not None:
            results.append(a._last_coverage)
    return results


def _build_model(arm_def):
    """Build a model for the given arm definition."""
    m = build(arm_def["arm"], arm_def.get("attn", {}),
              dense_layers=[], graph_config=arm_def.get("graph_config"))
    debug = arm_def.get("debug", {})
    if debug:
        _set_debug_config(m, debug)
    return m


def _generate(model, tokenizer, row, max_new=10):
    """Run generation, return (generated_text, output_ids, logits_at_answer)."""
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
    first_logits = out.logits[0][0].float().cpu() if out.logits else None
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    first = re.search(r"\d+", gen_text)
    exact = bool(first and first[0] == row["answer"])
    return {
        "generated": gen_text,
        "gen_ids": gen_ids,
        "exact": exact,
        "tokens": n_tok,
        "wall_s": wall,
        "peak_gb": peak_gb,
        "first_logits_top10": _top10(first_logits) if first_logits is not None else None,
    }


def _top10(logits):
    """Top-10 (token_id, logit_value) pairs."""
    vals, ids = logits.topk(10)
    return [(int(i), float(v)) for i, v in zip(ids, vals)]


# ── Task: verify ──────────────────────────────────────────────────────

def task_verify(tokenizer, bytes_list, n, depth):
    """Inject BB v3 routes into graph kernel, compare to real BB v3 matched."""
    print("=" * 72)
    print("TASK: verify — graph kernel with injected BB v3 routes vs real BB v3")
    print("=" * 72)

    for nbytes in bytes_list:
        rows, _ = prepare_task(tokenizer, "niah", nbytes, n, depth=depth)
        for row in rows:
            print(f"\n--- bytes={nbytes} idx={row['idx']} tokens={row['tokens']} ---")
            needle_span = _locate_needle(tokenizer, row)
            print(f"  needle span: {needle_span}  answer: {row['answer']}")

            # Arm A: real BB v3 matched
            print("  Building arm A (BB v3 matched)...")
            model_a = _build_model(ARM_A)
            res_a = _generate(model_a, tokenizer, row)
            del model_a; torch.cuda.empty_cache()

            # Arm B: graph kernel with injected BB v3 routes
            print("  Building arm B (graph + injected BB v3 routes)...")
            model_b = _build_model(ARM_B)
            res_b = _generate(model_b, tokenizer, row)
            del model_b; torch.cuda.empty_cache()

            match = res_a["gen_ids"] == res_b["gen_ids"]
            print(f"  Arm A: exact={res_a['exact']}  gen={res_a['generated']!r}")
            print(f"  Arm B: exact={res_b['exact']}  gen={res_b['generated']!r}")
            print(f"  Token IDs match: {match}")
            if not match:
                print(f"    A ids: {res_a['gen_ids']}")
                print(f"    B ids: {res_b['gen_ids']}")
            if res_a["first_logits_top10"] and res_b["first_logits_top10"]:
                a_top = res_a["first_logits_top10"][0]
                b_top = res_b["first_logits_top10"][0]
                print(f"  Top-1 logit: A=({a_top[0]}, {a_top[1]:.4f})  "
                      f"B=({b_top[0]}, {b_top[1]:.4f})")
            print(f"  VERDICT: {'PASS' if match else 'FAIL'}")


# ── Task: coverage ────────────────────────────────────────────────────

def task_coverage(tokenizer, bytes_list, n, depth):
    """Instrument needle coverage for graph vs BB v3 routing."""
    print("=" * 72)
    print("TASK: coverage — needle coverage per chunk/head/layer")
    print("=" * 72)

    for nbytes in bytes_list:
        rows, _ = prepare_task(tokenizer, "niah", nbytes, n, depth=depth)
        for row in rows:
            print(f"\n--- bytes={nbytes} idx={row['idx']} tokens={row['tokens']} ---")
            needle_span = _locate_needle(tokenizer, row)
            if needle_span is None:
                print("  SKIP: could not locate needle in tokenized text")
                continue
            lo, hi = needle_span
            print(f"  needle tokens [{lo}, {hi})  answer={row['answer']}")

            for arm_label, arm_def in [("C_semantic", ARM_C), ("D_question", ARM_D),
                                       ("B_bb_v3_inject", ARM_B)]:
                print(f"\n  [{arm_label}]")
                model = _build_model(arm_def)
                debug = dict(arm_def.get("debug", {}))
                debug["track_coverage"] = True
                debug["needle_span"] = (lo, hi)
                _set_debug_config(model, debug)

                res = _generate(model, tokenizer, row)
                coverage = _collect_coverage(model)
                del model; torch.cuda.empty_cache()

                print(f"    exact={res['exact']}  gen={res['generated']!r}")
                if not coverage:
                    print("    (no coverage data)")
                    continue

                # Summarize: final chunk hit rate, and total layers with any needle hit
                final_hits = sum(1 for c in coverage if c["final_chunk_hit"])
                any_hits = sum(1 for c in coverage
                               if any(c["needle_hit_any_head_per_chunk"]))
                total = len(coverage)
                print(f"    layers with needle in final chunk: {final_hits}/{total}")
                print(f"    layers with needle in any chunk: {any_hits}/{total}")

                # Detail for first, middle, last layer
                for label, idx in [("first", 0), ("mid", total // 2), ("last", -1)]:
                    c = coverage[idx]
                    chunk_hits = c["needle_hit_any_head_per_chunk"]
                    n_chunks_hit = sum(chunk_hits)
                    head_frac = c["needle_frac_heads_per_chunk"]
                    final_frac = sum(c["final_chunk_per_head"]) / len(c["final_chunk_per_head"])
                    print(f"    layer {c['layer']:2d} ({label}): "
                          f"chunks_with_needle={n_chunks_hit}/{len(chunk_hits)}  "
                          f"final_chunk_head_frac={final_frac:.2f}")


# ── Task: ablation ────────────────────────────────────────────────────

def task_ablation(tokenizer, bytes_list, n, depth):
    """Five-arm controlled comparison."""
    print("=" * 72)
    print("TASK: ablation — A/B/C/D/E arms")
    print("=" * 72)

    results = []
    for nbytes in bytes_list:
        rows, _ = prepare_task(tokenizer, "niah", nbytes, n, depth=depth)
        for row in rows:
            needle_span = _locate_needle(tokenizer, row)
            print(f"\n--- bytes={nbytes} idx={row['idx']} tokens={row['tokens']} "
                  f"needle={needle_span} ---")

            for arm_def in ALL_ABLATION_ARMS:
                name = arm_def["name"]
                print(f"  [{name}] {arm_def['description']}")
                model = _build_model(arm_def)

                # Enable coverage if we have a needle span and it's a graph arm
                if needle_span and arm_def["arm"] == "graph":
                    debug = dict(arm_def.get("debug", {}))
                    debug["track_coverage"] = True
                    debug["needle_span"] = needle_span
                    _set_debug_config(model, debug)

                res = _generate(model, tokenizer, row)
                coverage = _collect_coverage(model) if arm_def["arm"] == "graph" else []
                del model; torch.cuda.empty_cache()

                final_hit_layers = sum(1 for c in coverage if c["final_chunk_hit"]) if coverage else None
                total_layers = len(coverage) if coverage else None

                entry = {
                    "arm": name, "bytes": nbytes, "idx": row["idx"],
                    "tokens": res["tokens"], "exact": res["exact"],
                    "generated": res["generated"], "gen_ids": res["gen_ids"],
                    "wall_s": res["wall_s"], "peak_gb": res["peak_gb"],
                    "needle_span": needle_span,
                    "final_chunk_needle_layers": final_hit_layers,
                    "total_layers": total_layers,
                    "top10": res["first_logits_top10"],
                }
                results.append(entry)
                needle_info = (f"needle_final_chunk={final_hit_layers}/{total_layers}"
                               if final_hit_layers is not None else "")
                print(f"    exact={res['exact']}  gen={res['generated']!r}  "
                      f"wall={res['wall_s']:.1f}s  peak={res['peak_gb']:.1f}GB  "
                      f"{needle_info}")

    # Summary table
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Arm':<25s} {'Bytes':>8s} {'Tok':>7s} {'Exact':>6s} "
          f"{'NeedleCov':>10s} {'Wall':>6s} {'Peak':>6s}  Generated")
    for r in results:
        nc = (f"{r['final_chunk_needle_layers']}/{r['total_layers']}"
              if r["final_chunk_needle_layers"] is not None else "—")
        print(f"{r['arm']:<25s} {r['bytes']:>8d} {r['tokens']:>7d} "
              f"{'YES' if r['exact'] else 'NO':>6s} "
              f"{nc:>10s} {r['wall_s']:>5.1f}s {r['peak_gb']:>5.1f}G  "
              f"{r['generated']!r}")

    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Graph-sparse debug integration")
    p.add_argument("--task", required=True, choices=["verify", "coverage", "ablation"])
    p.add_argument("--bytes", default="57600", help="Comma-separated byte lengths")
    p.add_argument("--n", type=int, default=1, help="Examples per length")
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--out", default=None, help="JSON output path")
    a = p.parse_args()

    bytes_list = [int(x) for x in a.bytes.split(",")]
    tokenizer = load_tokenizer()

    if a.task == "verify":
        task_verify(tokenizer, bytes_list, a.n, a.depth)
    elif a.task == "coverage":
        task_coverage(tokenizer, bytes_list, a.n, a.depth)
    elif a.task == "ablation":
        results = task_ablation(tokenizer, bytes_list, a.n, a.depth)
        if a.out:
            Path(a.out).write_text(json.dumps(results, indent=2))
            print(f"\nResults written to {a.out}")


if __name__ == "__main__":
    main()
