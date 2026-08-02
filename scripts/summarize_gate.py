#!/usr/bin/env python3
"""Report the verdict for the two gate checks in run_gate_trillium.sbatch.

Reads benchmark JSON only, so it is safe to run on a login node.

  python scripts/summarize_gate.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load(path: str) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def accuracy(doc: dict) -> float | None:
    """Accuracy of a finished run.

    The exporter nests this under performance_metrics.eval; older files and the
    Llama harness use a flat evaluation_results block, so both are accepted.
    """
    for section in (
        (doc.get("performance_metrics") or {}).get("eval") or {},
        doc.get("evaluation_results") or {},
    ):
        if "eval_accuracy" in section:
            return section["eval_accuracy"]
    return None


def runs_in(directory: str, seq: int | None = None, preset: str | None = None) -> list[dict]:
    out = []
    for path in glob.glob(str(REPO / "benchmarks" / directory / "eval_*.json")):
        doc = load(path)
        if doc is None:
            continue
        acc = accuracy(doc)
        if acc is None:
            continue
        meta = doc.get("experiment_metadata") or {}
        cfg = meta.get("model_config") or {}
        if seq is not None and meta.get("seq_length") != seq:
            continue
        if preset is not None and cfg.get("compute_preset") != preset:
            continue
        out.append({
            "acc": acc,
            "seed": (meta.get("training_config") or {}).get("seed"),
            "ts": meta.get("timestamp") or "",
            "labels": (meta.get("dataset_info") or {}).get("num_labels"),
            "file": os.path.basename(path),
        })
    return sorted(out, key=lambda r: r["ts"])


def nolima_gate() -> None:
    runs = runs_in("nolima_onehop_exp_0_baseline")
    print("STAGE 1 — NoLiMa gate")
    if not runs:
        print("  no finished runs found")
        return
    newest = runs[-1]
    chance = 1.0 / (newest["labels"] or 10)
    print(f"  newest dense baseline: {newest['acc']:.3f}   (chance {chance:.3f}, {newest['ts']})")
    if newest["acc"] > chance * 2.5:
        print("  PASS — the task is learnable, NoLiMa deltas mean something now")
    elif newest["acc"] > chance * 1.3:
        print("  WEAK — above chance but far from solved; deltas will be very noisy")
    else:
        print("  FAIL — still at chance. The prompt fix was not enough; the recipe")
        print("         or the model is the problem, so more NoLiMa runs are wasted.")
    if len(runs) > 1:
        prev = [f"{r['acc']:.3f}" for r in runs[-4:-1]]
        print(f"  previous runs in this cell: {', '.join(prev)}")


def seed_study(seq: int, preset: str) -> None:
    print(f"\nSTAGE 2 — seed study, lra/listops seq={seq} {preset}")
    groups = {
        "dense (exp 0)": runs_in("lra_listops_exp_0_baseline", seq, preset),
        "layer-adaptive (exp 7)": runs_in("lra_listops_exp_7_layer_adaptive", seq, preset),
    }
    # Only runs that recorded a seed came from the seeded code path; anything
    # older trained with the HuggingFace default and would fake a zero spread.
    seeded = {k: [r for r in v if r["seed"] is not None] for k, v in groups.items()}
    if not all(len(v) > 1 for v in seeded.values()):
        for name, runs in groups.items():
            got = len(seeded[name])
            print(f"  {name}: {got} seeded run(s) of {len(runs)} total — need at least 2")
        print("  Not enough seeded runs to measure spread.")
        return

    spreads = []
    for name, runs in seeded.items():
        accs = [r["acc"] for r in runs]
        spread = max(accs) - min(accs)
        spreads.append(spread)
        detail = ", ".join(f"seed {r['seed']}: {r['acc']:.3f}" for r in runs)
        print(f"  {name:24s} mean {statistics.mean(accs):.3f}   spread {spread * 100:4.1f} pts   ({detail})")

    noise = max(spreads)
    gap = statistics.mean([r["acc"] for r in seeded["layer-adaptive (exp 7)"]]) - \
        statistics.mean([r["acc"] for r in seeded["dense (exp 0)"]])
    print(f"\n  measured noise floor (widest within-config spread): {noise * 100:.1f} pts")
    print(f"  layer-adaptive vs dense:                            {gap * 100:+.1f} pts")
    if abs(gap) > noise:
        print("  -> REAL: the gap survives the noise floor")
    else:
        print("  -> NOISE: the gap is smaller than the run-to-run spread, so this")
        print("     leaderboard position is not measurable at this number of seeds.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--preset", default="lra-full")
    args = ap.parse_args()
    print("=" * 66)
    nolima_gate()
    seed_study(args.seq, args.preset)
    print("=" * 66)


if __name__ == "__main__":
    main()
