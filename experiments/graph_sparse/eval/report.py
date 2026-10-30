"""Build report tables and plots from graph-sparse evaluation results.

Reads per-example JSONL files from the results directory and produces:
  - Markdown summary tables (accuracy, latency, memory).
  - Parameter sensitivity plots.
  - Accuracy vs throughput scatter.
  - Memory comparison bars.

Usage:
    python -m experiments.graph_sparse.eval.report --tag graph_s0_niah \
        --out experiments/graph_sparse/eval/RESULTS.md

Results directory structure:
    <RESULTS_BASE>/<tag>/<arm>_<task>_b<bytes>_d<depth>.jsonl
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_BASE = Path(os.environ.get("BB_RESULTS",
                    Path(os.environ.get("SCRATCH", "/vol")) / "results")) / "graph_eval"


def load_results(tag):
    """Load all JSONL result files for a tag.

    Returns:
        dict: {arm_name: {(task, bytes, depth): [result_dicts]}}
    """
    results_dir = RESULTS_BASE / tag
    if not results_dir.exists():
        print(f"No results directory: {results_dir}")
        return {}

    data = defaultdict(lambda: defaultdict(list))
    for f in sorted(results_dir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            arm = r.get("arm", f.stem.split("_")[0])
            task = r.get("task", "niah")
            nbytes = r.get("bytes", 0)
            depth = r.get("depth", 0.5)
            data[arm][(task, nbytes, depth)].append(r)
    return dict(data)


def summarize(results):
    """Compute per-arm aggregated statistics.

    Returns:
        list of dicts with: arm, task, bytes, depth, n, exact_acc, partial_acc,
        mean_prefill_s, mean_wall_s, max_peak_gb, graph_build_ms, attention_ms.
    """
    rows = []
    for arm, conditions in sorted(results.items()):
        for (task, nbytes, depth), examples in sorted(conditions.items()):
            n = len(examples)
            if n == 0:
                continue
            exact = sum(r["exact"] for r in examples)
            partial = sum(r.get("partial", r["exact"]) for r in examples)
            prefill = [r["prefill_s"] for r in examples]
            wall = [r.get("wall_s", 0) for r in examples]
            peak = [r.get("peak_gb", 0) for r in examples]

            graph_build = []
            attn_ms = []
            for r in examples:
                gs = r.get("graph_stats")
                if gs:
                    graph_build.append(gs.get("graph_build_ms", 0))
                    attn_ms.append(gs.get("attention_ms", 0))

            rows.append({
                "arm": arm,
                "task": task,
                "bytes": nbytes,
                "depth": depth,
                "n": n,
                "exact_acc": exact / n,
                "partial_acc": partial / n,
                "mean_prefill_s": sum(prefill) / n,
                "mean_wall_s": sum(wall) / n if wall else 0,
                "max_peak_gb": max(peak) if peak else 0,
                "mean_graph_build_ms": sum(graph_build) / len(graph_build) if graph_build else 0,
                "mean_attention_ms": sum(attn_ms) / len(attn_ms) if attn_ms else 0,
            })
    return rows


def write_markdown(summaries, out_path, tag):
    """Write a Markdown report from summary rows."""
    lines = [
        f"# Graph-Sparse Attention Evaluation: `{tag}`\n",
        "",
        "**Status**: Results below are from completed GPU runs only. "
        "Arms not listed have not been run yet.\n",
        "",
    ]

    tasks = sorted(set(r["task"] for r in summaries))
    for task in tasks:
        task_rows = [r for r in summaries if r["task"] == task]
        if not task_rows:
            continue

        lines.append(f"## {task}\n")

        bytes_depths = sorted(set((r["bytes"], r["depth"]) for r in task_rows))
        for nbytes, depth in bytes_depths:
            cond_rows = [r for r in task_rows
                         if r["bytes"] == nbytes and r["depth"] == depth]
            if not cond_rows:
                continue

            lines.append(f"### bytes={nbytes}, depth={depth}\n")
            lines.append(
                "| Arm | N | Exact Acc | Partial Acc | Prefill (s) | "
                "Peak (GB) | Graph Build (ms) | Attn (ms) |")
            lines.append(
                "|-----|---|-----------|-------------|-------------|"
                "-----------|------------------|-----------|")

            for r in sorted(cond_rows, key=lambda x: -x["exact_acc"]):
                lines.append(
                    f"| {r['arm']} | {r['n']} | "
                    f"{r['exact_acc']:.1%} | {r['partial_acc']:.1%} | "
                    f"{r['mean_prefill_s']:.3f} | "
                    f"{r['max_peak_gb']:.1f} | "
                    f"{r['mean_graph_build_ms']:.1f} | "
                    f"{r['mean_attention_ms']:.1f} |")
            lines.append("")

    # Paired error differences (if dense baseline exists)
    dense_results = {}
    for r in summaries:
        if r["arm"] == "dense":
            dense_results[(r["task"], r["bytes"], r["depth"])] = r

    if dense_results:
        lines.append("## Paired Differences vs Dense\n")
        lines.append(
            "| Arm | Task | Bytes | Acc Diff | Prefill Diff (s) | Peak Diff (GB) |")
        lines.append(
            "|-----|------|-------|----------|------------------|----------------|")
        for r in sorted(summaries, key=lambda x: (x["task"], x["bytes"], x["arm"])):
            if r["arm"] == "dense":
                continue
            key = (r["task"], r["bytes"], r["depth"])
            if key not in dense_results:
                continue
            d = dense_results[key]
            lines.append(
                f"| {r['arm']} | {r['task']} | {r['bytes']} | "
                f"{r['exact_acc'] - d['exact_acc']:+.1%} | "
                f"{r['mean_prefill_s'] - d['mean_prefill_s']:+.3f} | "
                f"{r['max_peak_gb'] - d['max_peak_gb']:+.1f} |")
        lines.append("")

    lines.append("---\n")
    lines.append("*Report generated by `experiments/graph_sparse/eval/report.py`.*\n")
    lines.append("*Only completed GPU results are shown. "
                 "Do not invent or extrapolate missing measurements.*\n")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Report written to {out_path}")


def make_plots(summaries, tag, out_dir=None):
    """Generate evaluation plots (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    if out_dir is None:
        out_dir = RESULTS_BASE / tag / "figures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Accuracy bar chart ──
    tasks = sorted(set(r["task"] for r in summaries))
    for task in tasks:
        task_rows = [r for r in summaries if r["task"] == task]
        if not task_rows:
            continue

        bytes_vals = sorted(set(r["bytes"] for r in task_rows))
        if len(bytes_vals) <= 1:
            arms = sorted(set(r["arm"] for r in task_rows))
            accs = [next((r["exact_acc"] for r in task_rows if r["arm"] == a), 0)
                    for a in arms]
            fig, ax = plt.subplots(figsize=(max(6, len(arms) * 0.8), 4))
            ax.bar(range(len(arms)), accs)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels(arms, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Exact Accuracy")
            ax.set_title(f"{task} Accuracy")
            ax.set_ylim(0, 1.05)
            fig.tight_layout()
            fig.savefig(out_dir / f"accuracy_{task}.png", dpi=150)
            plt.close(fig)

    # ── Accuracy vs prefill scatter ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in sorted(set(r["arm"] for r in summaries)):
        arm_rows = [r for r in summaries if r["arm"] == arm]
        xs = [r["mean_prefill_s"] for r in arm_rows]
        ys = [r["exact_acc"] for r in arm_rows]
        ax.scatter(xs, ys, label=arm, s=40, alpha=0.7)
    ax.set_xlabel("Mean Prefill (s)")
    ax.set_ylabel("Exact Accuracy")
    ax.set_title("Accuracy vs Prefill Latency")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_prefill.png", dpi=150)
    plt.close(fig)

    # ── Peak memory comparison ──
    arms = sorted(set(r["arm"] for r in summaries))
    peak_by_arm = {}
    for r in summaries:
        peak_by_arm[r["arm"]] = max(peak_by_arm.get(r["arm"], 0), r["max_peak_gb"])
    if peak_by_arm:
        fig, ax = plt.subplots(figsize=(max(6, len(arms) * 0.8), 4))
        vals = [peak_by_arm.get(a, 0) for a in arms]
        ax.bar(range(len(arms)), vals)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Peak GPU Memory (GB)")
        ax.set_title("Peak Memory by Arm")
        fig.tight_layout()
        fig.savefig(out_dir / "peak_memory.png", dpi=150)
        plt.close(fig)

    print(f"Plots saved to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="Experiment tag.")
    p.add_argument("--out", default=None,
                    help="Output markdown path (default: <results>/<tag>/RESULTS.md).")
    p.add_argument("--plots", action="store_true", help="Generate plots.")
    a = p.parse_args()

    results = load_results(a.tag)
    if not results:
        print("No results found.")
        sys.exit(1)

    summaries = summarize(results)
    total_examples = sum(r["n"] for r in summaries)
    print(f"Loaded {len(results)} arms, {len(summaries)} conditions, "
          f"{total_examples} total examples.")

    out_path = a.out or str(RESULTS_BASE / a.tag / "RESULTS.md")
    write_markdown(summaries, out_path, a.tag)

    if a.plots:
        make_plots(summaries, a.tag)


if __name__ == "__main__":
    main()
