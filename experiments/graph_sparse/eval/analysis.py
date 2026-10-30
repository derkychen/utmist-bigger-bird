"""Post-hoc analysis: which graph features help, hurt, or add only overhead.

Reads completed JSONL results and produces a structured analysis comparing
graph-sparse variants against baselines. Identifies:
  - Which features improve accuracy over BB v3.
  - Which features add latency/memory overhead without accuracy gain.
  - Optimal configuration recommendations.

Usage:
    python -m experiments.graph_sparse.eval.analysis --tag graph_s0

Output: prints analysis to stdout and writes to <results>/<tag>/ANALYSIS.md.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from experiments.graph_sparse.eval.report import load_results, summarize, RESULTS_BASE


def _acc(summaries, arm_name, task="niah"):
    """Get accuracy for an arm on a task, averaged across conditions."""
    rows = [r for r in summaries
            if r["arm"] == arm_name and r["task"] == task]
    if not rows:
        return None
    return sum(r["exact_acc"] for r in rows) / len(rows)


def _prefill(summaries, arm_name, task="niah"):
    """Get mean prefill for an arm on a task."""
    rows = [r for r in summaries
            if r["arm"] == arm_name and r["task"] == task]
    if not rows:
        return None
    return sum(r["mean_prefill_s"] for r in rows) / len(rows)


def _peak(summaries, arm_name):
    """Get max peak memory for an arm."""
    rows = [r for r in summaries if r["arm"] == arm_name]
    if not rows:
        return None
    return max(r["max_peak_gb"] for r in rows)


def analyze_stage(stage_name, arm_names, summaries, baseline_name):
    """Analyze one ablation stage against a baseline."""
    bl_acc = _acc(summaries, baseline_name)
    bl_pre = _prefill(summaries, baseline_name)
    bl_mem = _peak(summaries, baseline_name)

    if bl_acc is None:
        return {
            "stage": stage_name,
            "status": "MISSING",
            "note": f"Baseline {baseline_name} has no results.",
        }

    results = []
    for arm in arm_names:
        acc = _acc(summaries, arm)
        pre = _prefill(summaries, arm)
        mem = _peak(summaries, arm)
        if acc is None:
            results.append({
                "arm": arm,
                "status": "NOT_RUN",
            })
            continue

        acc_diff = acc - bl_acc
        pre_ratio = pre / bl_pre if bl_pre and bl_pre > 0 else None
        mem_diff = (mem - bl_mem) if mem and bl_mem else None

        if acc_diff > 0.05:
            verdict = "HELPS"
        elif acc_diff < -0.05:
            verdict = "HURTS"
        elif pre_ratio and pre_ratio > 1.2:
            verdict = "OVERHEAD_ONLY"
        else:
            verdict = "NEUTRAL"

        results.append({
            "arm": arm,
            "status": "RUN",
            "accuracy": acc,
            "acc_diff": acc_diff,
            "prefill_ratio": pre_ratio,
            "memory_diff_gb": mem_diff,
            "verdict": verdict,
        })

    helping = [r for r in results if r.get("verdict") == "HELPS"]
    hurting = [r for r in results if r.get("verdict") == "HURTS"]
    overhead = [r for r in results if r.get("verdict") == "OVERHEAD_ONLY"]
    not_run = [r for r in results if r["status"] == "NOT_RUN"]

    best = max(
        [r for r in results if r["status"] == "RUN"],
        key=lambda r: (r["accuracy"], -(r.get("prefill_ratio") or 999)),
        default=None,
    )

    return {
        "stage": stage_name,
        "status": "COMPLETE" if not not_run else "PARTIAL",
        "baseline": baseline_name,
        "baseline_acc": bl_acc,
        "results": results,
        "helping": [r["arm"] for r in helping],
        "hurting": [r["arm"] for r in hurting],
        "overhead_only": [r["arm"] for r in overhead],
        "not_run": [r["arm"] for r in not_run],
        "best_arm": best["arm"] if best else None,
        "best_acc": best["accuracy"] if best else None,
    }


def full_analysis(summaries):
    """Run the full staged analysis."""
    from experiments.graph_sparse.eval.arms import STAGES

    stages_config = [
        ("Stage 0: Baselines", 0, "dense"),
        ("Stage 1: Static graph", 1, "bb_v3_matched"),
        ("Stage 2: Rewiring", 2, "graph_static_b8"),
        ("Stage 3: Walks", 3, "graph_rewire_sem_05"),
        ("Stage 4: Memory layout", 4, "graph_walk_random_l5"),
    ]

    analyses = []
    for stage_name, stage_num, baseline in stages_config:
        arms = STAGES.get(stage_num, [])
        arm_names = [a["name"] for a in arms]
        analysis = analyze_stage(stage_name, arm_names, summaries, baseline)
        analyses.append(analysis)

    return analyses


def write_analysis(analyses, out_path, tag):
    """Write analysis to Markdown."""
    lines = [
        f"# Graph-Sparse Attention Analysis: `{tag}`\n",
        "",
        "**This analysis is based only on completed GPU results.** "
        "Arms marked NOT_RUN have no data yet.\n",
        "",
    ]

    for a in analyses:
        lines.append(f"## {a['stage']}\n")

        if a["status"] == "MISSING":
            lines.append(f"*{a['note']}*\n")
            continue

        lines.append(f"Baseline: **{a.get('baseline', '?')}** "
                      f"(acc={a.get('baseline_acc', 0):.1%})\n")

        if a.get("helping"):
            lines.append(f"**Helps** (>5% acc gain): {', '.join(a['helping'])}\n")
        if a.get("hurting"):
            lines.append(f"**Hurts** (>5% acc loss): {', '.join(a['hurting'])}\n")
        if a.get("overhead_only"):
            lines.append(f"**Overhead only** (>20% slower, no acc gain): "
                          f"{', '.join(a['overhead_only'])}\n")
        if a.get("not_run"):
            lines.append(f"*Not run yet*: {', '.join(r['arm'] for r in a['not_run'])}\n")

        if a.get("best_arm"):
            lines.append(f"**Best arm**: {a['best_arm']} "
                          f"(acc={a['best_acc']:.1%})\n")

        lines.append("")
        lines.append("| Arm | Acc | Acc Diff | Prefill Ratio | Mem Diff (GB) | Verdict |")
        lines.append("|-----|-----|----------|---------------|---------------|---------|")
        for r in a.get("results", []):
            if r["status"] == "NOT_RUN":
                lines.append(f"| {r['arm']} | — | — | — | — | NOT_RUN |")
            else:
                lines.append(
                    f"| {r['arm']} | {r['accuracy']:.1%} | "
                    f"{r['acc_diff']:+.1%} | "
                    f"{r.get('prefill_ratio', 0):.2f}x | "
                    f"{r.get('memory_diff_gb', 0):+.1f} | "
                    f"{r['verdict']} |")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations\n")

    all_helping = []
    all_hurting = []
    all_overhead = []
    for a in analyses:
        all_helping.extend(a.get("helping", []))
        all_hurting.extend(a.get("hurting", []))
        all_overhead.extend(a.get("overhead_only", []))

    if all_helping:
        lines.append(f"**Keep**: {', '.join(all_helping)} — "
                      "these features improve accuracy.\n")
    if all_hurting:
        lines.append(f"**Remove**: {', '.join(all_hurting)} — "
                      "these features hurt accuracy.\n")
    if all_overhead:
        lines.append(f"**Reconsider**: {', '.join(all_overhead)} — "
                      "these add latency without accuracy gain.\n")

    incomplete = sum(1 for a in analyses if a.get("not_run"))
    if incomplete:
        lines.append(f"\n*{incomplete} stage(s) have unrun arms. "
                      "Conclusions may change with more data.*\n")

    lines.append("\n---\n")
    lines.append("*Analysis generated by `experiments/graph_sparse/eval/analysis.py`.*\n")
    lines.append("*Only completed GPU results are shown. "
                 "Do not invent or extrapolate missing measurements.*\n")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Analysis written to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="Experiment tag.")
    p.add_argument("--out", default=None,
                    help="Output analysis path.")
    a = p.parse_args()

    results = load_results(a.tag)
    if not results:
        print("No results found.")
        sys.exit(1)

    summaries = summarize(results)
    analyses = full_analysis(summaries)

    for ana in analyses:
        print(f"\n{ana['stage']}: {ana['status']}")
        if ana.get("best_arm"):
            print(f"  Best: {ana['best_arm']} ({ana['best_acc']:.1%})")
        if ana.get("helping"):
            print(f"  Helps: {', '.join(ana['helping'])}")
        if ana.get("hurting"):
            print(f"  Hurts: {', '.join(ana['hurting'])}")

    out_path = a.out or str(RESULTS_BASE / a.tag / "ANALYSIS.md")
    write_analysis(analyses, out_path, a.tag)


if __name__ == "__main__":
    main()
