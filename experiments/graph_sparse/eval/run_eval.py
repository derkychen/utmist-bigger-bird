"""Run graph-sparse attention evaluation across all configured arms.

Executes arms sequentially within one GPU process for paired comparison.
Results are saved per-example to JSONL for resumable runs.

Usage (local):
    python -m experiments.graph_sparse.eval.run_eval \
        --stage 0 --task niah --bytes 512000 --n 10 --tag graph_s0_niah

Usage (Modal):
    modal run modal_rd/app.py --module experiments.graph_sparse.eval.run_eval \
        --args "--stage 1 --task niah --bytes 512000 --n 10 --tag graph_s1"

Supported tasks:
    niah             -- RULER single-needle retrieval.
    niah_multikey_1  -- RULER multi-key NIAH.
    niah_multivalue  -- RULER multi-value NIAH.
    multihop_2       -- Multi-hop NIAH (2 hops).
    multihop_3       -- Multi-hop NIAH (3 hops).
    lbv2             -- LongBench v2 (exp 20 tasks module).
    codedebug        -- InfiniteBench Code.Debug.
    lcqa32k          -- LongCodeQA 32K.
    lcqa64k          -- LongCodeQA 64K.
    lcqa128k         -- LongCodeQA 128K.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from experiments.graph_sparse.eval.arms import get_arms, ALL_ARMS


RESULTS_BASE = Path(os.environ.get("BB_RESULTS",
                    Path(os.environ.get("SCRATCH", "/vol")) / "results")) / "graph_eval"


def _load_existing(path):
    """Load previously completed example indices from a JSONL file."""
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add(r["idx"])
    return done


def run_niah_task(arm_def, task, bytes_list, n, depths, tag, max_new=10):
    """Run RULER-style NIAH evaluation for one arm."""
    from modal_rd.niah import load_tokenizer, prepare_task
    from modal_rd.eval_niah import build, run_one

    tokenizer = load_tokenizer()
    arm_name = arm_def["name"]
    graph_config = arm_def.get("graph_config")
    attn = arm_def.get("attn", {})

    model = build(arm_def["arm"], attn,
                  dense_layers=[],
                  graph_config=graph_config)

    results_dir = RESULTS_BASE / tag
    results_dir.mkdir(parents=True, exist_ok=True)

    for nbytes in bytes_list:
        for depth in depths:
            out_path = results_dir / f"{arm_name}_{task}_b{nbytes}_d{depth}.jsonl"
            done = _load_existing(out_path)

            rows, skipped = prepare_task(tokenizer, task, nbytes, n, depth=depth)
            pending = [r for r in rows if r["idx"] not in done]

            if not pending:
                print(f"SKIP {arm_name} {task} bytes={nbytes} depth={depth} "
                      f"(all {len(rows)} done)")
                continue

            print(f"RUN  {arm_name} {task} bytes={nbytes} depth={depth} "
                  f"({len(pending)}/{len(rows)} pending)")

            exact_total, n_run = 0, 0
            with open(out_path, "a") as f:
                for row in pending:
                    r = run_one(model, tokenizer, row, arm_def["arm"],
                                max_new, kv_int8=False)
                    r["arm"] = arm_name
                    r["task"] = task
                    r["bytes"] = nbytes
                    r["depth"] = depth
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    exact_total += r["exact"]
                    n_run += 1
                    print(f"  idx={r['idx']} exact={r['exact']} "
                          f"prefill={r['prefill_s']:.2f}s "
                          f"peak={r['peak_gb']:.1f}GB "
                          f"gen={r['generated']!r}")

            total_done = len(done) + n_run
            print(f"DONE {arm_name} {task} bytes={nbytes} depth={depth} "
                  f"exact={exact_total}/{n_run} "
                  f"(total {total_done}/{len(rows)})")

    del model
    torch.cuda.empty_cache()


def run_multihop_task(arm_def, hops, bytes_val, n, tag, max_new=10):
    """Run multi-hop NIAH for one arm."""
    from modal_rd.niah import load_tokenizer
    from modal_rd.eval_niah import build, run_one
    from experiments.graph_sparse.eval.multihop_niah import prepare_multihop

    tokenizer = load_tokenizer()
    arm_name = arm_def["name"]

    model = build(arm_def["arm"], arm_def.get("attn", {}),
                  dense_layers=[],
                  graph_config=arm_def.get("graph_config"))

    results_dir = RESULTS_BASE / tag
    results_dir.mkdir(parents=True, exist_ok=True)

    task_name = f"multihop_{hops}hop"
    out_path = results_dir / f"{arm_name}_{task_name}_b{bytes_val}.jsonl"
    done = _load_existing(out_path)

    rows = prepare_multihop(tokenizer, bytes_val, n, hops=hops)
    pending = [r for r in rows if r["idx"] not in done]

    if not pending:
        print(f"SKIP {arm_name} {task_name} (all {len(rows)} done)")
        del model
        torch.cuda.empty_cache()
        return

    print(f"RUN  {arm_name} {task_name} bytes={bytes_val} "
          f"({len(pending)}/{len(rows)} pending)")

    exact_total = 0
    with open(out_path, "a") as f:
        for row in pending:
            r = run_one(model, tokenizer, row, arm_def["arm"],
                        max_new, kv_int8=False)
            r["arm"] = arm_name
            r["task"] = task_name
            r["hops"] = hops
            f.write(json.dumps(r) + "\n")
            f.flush()
            exact_total += r["exact"]
            print(f"  idx={r['idx']} exact={r['exact']} "
                  f"prefill={r['prefill_s']:.2f}s "
                  f"gen={r['generated']!r}")

    print(f"DONE {arm_name} {task_name} exact={exact_total}/{len(pending)}")
    del model
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, default=None,
                    help="Run only arms from this stage (0-4).")
    p.add_argument("--arms", nargs="+", default=None,
                    help="Run only these arm names.")
    p.add_argument("--task", default="niah",
                    help="Task to evaluate.")
    p.add_argument("--bytes", default="512000",
                    help="Comma-separated byte lengths.")
    p.add_argument("--n", type=int, default=10,
                    help="Examples per (bytes, depth) combination.")
    p.add_argument("--depths", default="0.5",
                    help="Comma-separated depth values.")
    p.add_argument("--max-new", type=int, default=10)
    p.add_argument("--tag", required=True,
                    help="Experiment tag for results directory.")
    a = p.parse_args()

    arms = get_arms(stage=a.stage, names=a.arms)
    bytes_list = [int(x) for x in a.bytes.split(",")]
    depths = [float(x) for x in a.depths.split(",")]

    print(f"Graph-sparse evaluation: {len(arms)} arms, task={a.task}, "
          f"bytes={bytes_list}, n={a.n}")
    for arm in arms:
        print(f"  {arm['name']}: {arm['description']}")
    print()

    for arm_def in arms:
        t0 = time.perf_counter()
        if a.task.startswith("multihop_"):
            hops = int(a.task.split("_")[1].replace("hop", ""))
            run_multihop_task(arm_def, hops, bytes_list[0], a.n, a.tag,
                              max_new=a.max_new)
        else:
            run_niah_task(arm_def, a.task, bytes_list, a.n, depths, a.tag,
                          max_new=a.max_new)
        elapsed = time.perf_counter() - t0
        print(f"ARM {arm_def['name']} completed in {elapsed:.1f}s\n")

    print("ALL DONE")


if __name__ == "__main__":
    main()
