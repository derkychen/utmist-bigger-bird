#!/usr/bin/env python3
"""
Generate report figures and appendix table for the Bigger Bird report.

Figures show four highlighted experiments:
  0  Dense Baseline      (reference)
  5  Bigger Bird Unified (the target design)
  13 Dynamic Context     (#1 overall score across F1/speed/memory/sparsity axes)
  8  Token Drop          (#2 overall score; only method to complete 65k tokens)

All remaining experiments are collected into a LaTeX appendix table.

Usage (from project root):
  python viz/report_selected_figures.py

Outputs:
  report/figures/sel_loglog.{pdf,png}
  report/figures/sel_linear_short.{pdf,png}
  report/figures/sel_linear_long.{pdf,png}
  report/figures/sel_f1.{pdf,png}
  report/figures/sel_latency.{pdf,png}
  report/figures/sel_memory.{pdf,png}
  report/figures/sel_traintime.{pdf,png}
  report/appendix_omitted.tex
"""

import json
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
BDIR = os.path.join(ROOT, "benchmarks")
OUT  = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)

# ── experiments to highlight in figures ───────────────────────────────────────
# complexity_results.json uses short keys (exp_1, exp_2, …) except exp_0_baseline
KERNEL_SHORTNAME = {
    "exp_0_baseline":         "exp_0_baseline",
    "exp_1_deepseek_topk":    "exp_1",
    "exp_2_lightning_hybrid": "exp_2",
    "exp_3_dynamic_globals":  "exp_3",
    "exp_4_pbs_attn":         "exp_4",
    "exp_5_bigger_bird":      "exp_5",
    "exp_6_deepseek_pbs":     "exp_6",
    "exp_7_layer_adaptive":   "exp_7",
    "exp_8_token_drop":       "exp_8",
    "exp_9_attn_specul":      "exp_9",
    "exp_10_gqa_sparse":      "exp_10",
}
KERNEL_LONGNAME = {v: k for k, v in KERNEL_SHORTNAME.items()}

SELECTED_KERNEL = [
    "exp_0_baseline",
    "exp_5_bigger_bird",
    "exp_7_layer_adaptive",
    "exp_8_token_drop",
]
# efficiency figures use Dynamic Context instead of Layer-Adaptive
# (exp_13 has no kernel benchmark data so it can't appear in figs 1-2)
SELECTED_EFFICIENCY = [
    "exp_0_baseline",
    "exp_5_bigger_bird",
    "exp_13_dynamic_context",
    "exp_8_token_drop",
]

# ── visual identity ───────────────────────────────────────────────────────────
EXP_STYLE = {
    "exp_0_baseline":         {"label": "Dense Baseline (0)",   "color": "#1a1a1a", "ls": "-",  "marker": "s", "lw": 1.6},
    "exp_5_bigger_bird":      {"label": "Bigger Bird (5)",      "color": "#7c3aed", "ls": "--", "marker": "D", "lw": 1.8},
    "exp_7_layer_adaptive":   {"label": "Layer-Adaptive (7)",   "color": "#0ea5e9", "ls": "-",  "marker": "o", "lw": 1.8},
    "exp_8_token_drop":       {"label": "Token Drop (8)",       "color": "#16a34a", "ls": "-",  "marker": "^", "lw": 1.8},
    "exp_13_dynamic_context": {"label": "Dynamic Context (13)", "color": "#dc2626", "ls": "-.", "marker": "P", "lw": 1.8},
}

# all exp names that exist in the benchmark data
ALL_EXPS_META = {
    "exp_0_baseline":         "Dense Baseline",
    "exp_1_deepseek_topk":    "DeepSeek Top-k",
    "exp_2_lightning_hybrid": "Lightning Hybrid",
    "exp_3_dynamic_globals":  "Dynamic Global Tokens",
    "exp_4_pbs_attn":         "PBS Attention",
    "exp_5_bigger_bird":      "Bigger Bird Unified",
    "exp_6_deepseek_pbs":     "DeepSeek+PBS Hybrid",
    "exp_7_layer_adaptive":   "Layer-Adaptive Sparsity",
    "exp_8_token_drop":       "Token Dropping",
    "exp_9_attn_specul":      "Attention Speculation",
    "exp_10_gqa_sparse":      "GQA+Sparse",
    "exp_11_nsa":             "Native Sparse Attn (NSA)",
    "exp_12_s2_hhst":         "S2-HHST",
    "exp_13_dynamic_context": "Dynamic Context",
    "exp_14_token_drop_deepseek": "Token Drop + DeepSeek",
    "exp_15_bigger_bird":     "Bigger Bird (BigBird-RoBERTa)",
}

# ── plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8.5,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linestyle":    "--",
})


# ── data loaders ───────────────────────────────────────────────────────────────

def load_complexity():
    """Return {full_exp_name: {seq: time_ms}} from complexity_results.json."""
    p = os.path.join(BDIR, "complexity_results.json")
    raw = defaultdict(dict)
    with open(p) as f:
        d = json.load(f)
    for r in d["results"]:
        v = None if r.get("oom") else r.get("time_ms")
        raw[r["exp_name"]][r["seq_length"]] = v
    # normalise short keys → full keys
    data = defaultdict(dict)
    for key, seqmap in raw.items():
        full = KERNEL_LONGNAME.get(key, key)
        data[full].update(seqmap)
    return data


def load_imdb_task():
    """
    Return {exp_name: {seq: {f1, accuracy, train_time_s, peak_mem_mb,
                              inference_ms, oom}}}
    Merges long-context sweep JSONs + per-experiment eval JSONs.
    For each (exp, seq) keeps the record with the highest F1.
    """
    records: dict[tuple, dict] = {}

    # ── long-context sweep files ─────────────────────────────────────────────
    for fp in sorted(glob.glob(os.path.join(BDIR, "long_context_sweep_*.json"))):
        with open(fp) as f:
            items = json.load(f)
        if isinstance(items, dict):
            items = items.get("results", [])
        for r in items:
            seq = r.get("seq", r.get("seq_length", 0))
            exp = r.get("exp_name", "")
            if not seq or not exp:
                continue
            rec = {
                "f1":           r.get("f1"),
                "accuracy":     r.get("accuracy"),
                "train_time_s": r.get("train_time_s"),
                "peak_mem_mb":  r.get("peak_mem_mb"),
                "inference_ms": r.get("inference_ms"),
                "oom":          r.get("oom", False),
            }
            key = (exp, seq)
            if key not in records or (rec["f1"] or 0) > (records[key]["f1"] or 0):
                records[key] = rec

    # ── per-experiment eval JSONs ─────────────────────────────────────────────
    for exp_dir in os.listdir(BDIR):
        dp = os.path.join(BDIR, exp_dir)
        if not os.path.isdir(dp):
            continue
        # normalise directory name to a canonical exp name
        base = exp_dir.split("_seq")[0]
        for fp in glob.glob(os.path.join(dp, "eval_*.json")):
            try:
                with open(fp) as f:
                    d = json.load(f)
                meta = d["experiment_metadata"]
                perf = d["performance_metrics"]
                seq  = meta["dataset_info"].get("max_seq_len", 0)
                if seq < 256:      # skip dynamic-padding era (~166 token) runs
                    continue
                rec = {
                    "f1":           perf["eval"].get("eval_f1"),
                    "accuracy":     perf["eval"].get("eval_accuracy"),
                    "train_time_s": perf.get("training_time_seconds"),
                    "peak_mem_mb":  (perf.get("peak_memory_mb") or
                                     meta.get("environment", {}).get("peak_memory_mb")),
                    "inference_ms": perf.get("inference_latency_ms"),
                    "oom":          False,
                }
                key = (base, seq)
                if key not in records or (rec["f1"] or 0) > (records[key]["f1"] or 0):
                    records[key] = rec
            except Exception:
                pass

    # ── also pull exp_14 from LRA sweep (only available source) ─────────────
    for fp in sorted(glob.glob(os.path.join(BDIR, "lra_sweep_*.json"))):
        with open(fp) as f:
            items = json.load(f)
        if isinstance(items, dict):
            items = items.get("results", [])
        for r in items:
            if r.get("exp_name") != "exp_14_token_drop_deepseek":
                continue
            seq = r.get("seq", 0)
            if not seq:
                continue
            rec = {
                "f1":           r.get("f1"),
                "accuracy":     r.get("accuracy"),
                "train_time_s": r.get("train_time_s"),
                "peak_mem_mb":  r.get("peak_mem_mb"),
                "inference_ms": r.get("inference_ms"),
                "oom":          r.get("oom", False),
                "_track":       "LRA-Text",
            }
            key = ("exp_14_token_drop_deepseek", seq)
            if key not in records or (rec["f1"] or 0) > (records[key]["f1"] or 0):
                records[key] = rec

    result: dict[str, dict[int, dict]] = defaultdict(dict)
    for (exp, seq), rec in records.items():
        result[exp][seq] = rec
    return result


# ── shared helpers ─────────────────────────────────────────────────────────────

def _save(fig, name):
    for ext in (".pdf", ".png"):
        p = os.path.join(OUT, name + ext)
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext == ".png" else 150)
    plt.close(fig)
    print(f"  saved {name}.pdf/.png")


def _plot_lines(ax, xs_ys_by_exp, exps, x_label, y_label,
                xscale="linear", yscale="linear", oom_xs=None):
    """Draw one line per experiment, return artists for legend."""
    for ename in exps:
        pts = xs_ys_by_exp.get(ename, [])
        if not pts:
            continue
        pts = sorted(pts)
        xs, ys = zip(*pts)
        s = EXP_STYLE.get(ename, {})
        ax.plot(xs, ys,
                label=s.get("label", ename),
                color=s.get("color", "gray"),
                linestyle=s.get("ls", "-"),
                marker=s.get("marker", "o"),
                linewidth=s.get("lw", 1.4),
                markersize=5)
    if oom_xs:
        for (exp, seq, y_approx) in oom_xs:
            s = EXP_STYLE.get(exp, {})
            ax.scatter([seq], [y_approx], marker="x", s=80,
                       color=s.get("color", "gray"), zorder=5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if xscale == "log2":
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"))
    else:
        ax.set_xscale(xscale)
    ax.set_yscale(yscale)


# ── Figure 1: log-log kernel time ─────────────────────────────────────────────

def fig_loglog(cx):
    pts = {}
    for ename in SELECTED_KERNEL:
        seqmap = cx.get(ename, {})
        pts[ename] = [(s, t) for s, t in seqmap.items() if t is not None]

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    _plot_lines(ax, pts, SELECTED_KERNEL,
                "Sequence length (tokens)", "Attention forward time (ms)",
                xscale="log2", yscale="log")
    ax.set_title("Log–log attention-kernel time")
    ax.legend(framealpha=0.85, edgecolor="none")
    fig.tight_layout()
    _save(fig, "sel_loglog")


# ── Figure 2 & 3: linear-scale kernel time ───────────────────────────────────

def fig_linear(cx):
    short_seqs = [128, 256, 512]
    long_seqs  = [1024, 2048, 4096]

    for seqs, suffix, title in [
        (short_seqs, "sel_linear_short", "Linear scale — short sequences (128–512 tokens)"),
        (long_seqs,  "sel_linear_long",  "Linear scale — long sequences (1024–4096 tokens)"),
    ]:
        pts = {}
        for ename in SELECTED_KERNEL:
            seqmap = cx.get(ename, {})
            pts[ename] = [(s, seqmap[s]) for s in seqs if seqmap.get(s) is not None]

        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        _plot_lines(ax, pts, SELECTED_KERNEL,
                    "Sequence length (tokens)", "Attention forward time (ms)")
        ax.set_xticks(seqs)
        ax.set_xticklabels([f"{s:,}" for s in seqs])
        ax.set_title(title)
        ax.legend(framealpha=0.85, edgecolor="none")
        fig.tight_layout()
        _save(fig, suffix)


# ── Figures 4-7: efficiency metrics ───────────────────────────────────────────

def fig_efficiency(imdb):
    metrics = [
        ("f1",           "IMDb F1",                  "sel_f1",        False, (0.0, 1.05)),
        ("inference_ms", "Inference latency (ms/seq)","sel_latency",   True,  None),
        ("peak_mem_mb",  "Peak memory (MB)",          "sel_memory",    False, None),
        ("train_time_s", "Training time (s)",         "sel_traintime", True,  None),
    ]

    for key, ylabel, fname, log_y, ylim in metrics:
        pts = {}
        for ename in SELECTED_EFFICIENCY:
            seqmap = imdb.get(ename, {})
            collected = []
            for seq, r in seqmap.items():
                if r.get("oom"):
                    continue
                v = r.get(key)
                if v is not None:
                    collected.append((seq, v))
            pts[ename] = collected

        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        _plot_lines(ax, pts, SELECTED_EFFICIENCY,
                    "Sequence length (tokens)", ylabel,
                    xscale="log2", yscale="log" if log_y else "linear")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(f"IMDb {ylabel} vs sequence length")
        ax.legend(framealpha=0.85, edgecolor="none")
        fig.tight_layout()
        _save(fig, fname)


# ── Appendix LaTeX table ──────────────────────────────────────────────────────

def _fmt(v, digits=3):
    if v is None:
        return "---"
    if isinstance(v, bool):
        return "OOM" if v else "---"
    return f"{v:.{digits}f}"

def _fmt_int(v):
    if v is None:
        return "---"
    return f"{int(v):,}"


def build_appendix_table(cx, imdb):
    """
    Produce report/appendix_omitted.tex with two tables:
      A.1  Kernel benchmark times for omitted experiments
      A.2  Task-level IMDb results for omitted experiments
    """
    omitted_kernel = [e for e in sorted(cx.keys()) if e not in SELECTED_KERNEL]
    omitted_task   = [e for e in sorted(ALL_EXPS_META.keys())
                      if e not in SELECTED_EFFICIENCY and e in imdb]

    lines = []
    lines.append(r"\section*{Appendix: Full Results for Omitted Experiments}")
    lines.append(r"\addcontentsline{toc}{section}{Appendix: Full Results for Omitted Experiments}")
    lines.append("")
    lines.append(r"This appendix reports all benchmark data for experiments not shown in the main figures.")
    lines.append(r"Experiments highlighted in the main text are: Dense Baseline (0), Bigger Bird Unified (5),")
    lines.append(r"Dynamic Context (13, highest overall score), and Token Dropping (8, second highest overall score).")
    lines.append("")

    # ── A.1: kernel times ────────────────────────────────────────────────────
    kernel_seqs = [128, 256, 512, 1024, 2048, 4096]
    lines.append(r"\subsection*{A.1\quad Attention-Kernel Forward Time (ms)}")
    lines.append(r"OOM denotes an out-of-memory failure; `---' denotes not benchmarked at that length.")
    lines.append("")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Attention-kernel forward time (ms) for omitted experiments, 128--4096 tokens.}")
    lines.append(r"  \label{tab:app-kernel}")
    col_spec = "@{}l" + "r" * len(kernel_seqs) + "@{}"
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")
    header_seqs = " & ".join(f"{s:,}" for s in kernel_seqs)
    lines.append(rf"    Experiment & {header_seqs} \\")
    lines.append(r"    \midrule")
    for ename in omitted_kernel:
        seqmap = cx[ename]
        cells = []
        for s in kernel_seqs:
            v = seqmap.get(s)
            cells.append("OOM" if (v is None and s in seqmap) else
                         f"{v:.3f}" if v is not None else "---")
        row = " & ".join(cells)
        name = ALL_EXPS_META.get(ename, ename.replace("_", " "))
        lines.append(rf"    {name} & {row} \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # ── A.2: task-level IMDb results ─────────────────────────────────────────
    lines.append(r"\subsection*{A.2\quad IMDb Task-Level Results (All Sequence Lengths)}")
    lines.append(r"F1 = 0.000 with accuracy $\approx$ 0.53 indicates majority-class collapse.")
    lines.append(r"OOM rows are included where recorded. `---' denotes metric not recorded.")
    lines.append("")
    lines.append(r"\begin{longtable}{@{}llrrrrrr@{}}")
    lines.append(r"  \caption{IMDb task-level results for all experiments not in the main figures.}")
    lines.append(r"  \label{tab:app-imdb} \\")
    lines.append(r"  \toprule")
    lines.append(r"  Experiment & Seq & F1 & Acc & Train (s) & Memory (MB) & Latency (ms) \\")
    lines.append(r"  \midrule")
    lines.append(r"  \endfirsthead")
    lines.append(r"  \multicolumn{7}{c}{\tablename\ \thetable{} — continued} \\")
    lines.append(r"  \toprule")
    lines.append(r"  Experiment & Seq & F1 & Acc & Train (s) & Memory (MB) & Latency (ms) \\")
    lines.append(r"  \midrule")
    lines.append(r"  \endhead")
    lines.append(r"  \bottomrule")
    lines.append(r"  \endlastfoot")

    # collect all task-level data including ALL exps (for appendix we want omitted ones)
    # plus ALL seqs
    all_task_exps = sorted(
        set(list(imdb.keys()) + list(ALL_EXPS_META.keys())),
        key=lambda e: int(e.split("_")[1]) if len(e.split("_")) > 1 and e.split("_")[1].isdigit() else 99
    )

    for ename in all_task_exps:
        if ename in SELECTED_EFFICIENCY:
            continue
        seqmap = imdb.get(ename, {})
        if not seqmap:
            # Exp 15 and others with no data
            name = ALL_EXPS_META.get(ename, ename)
            lines.append(rf"  {name} & \multicolumn{{6}}{{l}}{{No task-level run recorded}} \\")
            continue
        name = ALL_EXPS_META.get(ename, ename)
        first = True
        for seq in sorted(seqmap.keys()):
            r = seqmap[seq]
            display_name = name if first else ""
            first = False
            oom = r.get("oom", False)
            if oom:
                lines.append(rf"  {display_name} & {seq:,} & \multicolumn{{5}}{{l}}{{OOM}} \\")
                continue
            f1  = _fmt(r.get("f1"))
            acc = _fmt(r.get("accuracy"))
            tr  = _fmt(r.get("train_time_s"), 1)
            mem = _fmt(r.get("peak_mem_mb"), 0)
            lat = _fmt(r.get("inference_ms"), 2)
            lines.append(rf"  {display_name} & {seq:,} & {f1} & {acc} & {tr} & {mem} & {lat} \\")
        lines.append(r"  \addlinespace[4pt]")

    lines.append(r"\end{longtable}")

    out_path = os.path.join(ROOT, "report", "appendix_omitted.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved appendix_omitted.tex")
    return out_path


# ── Figure 8: BigBird family bar chart ───────────────────────────────────────

def fig_bigbird_family():
    """F1 comparison for BigBird family at 768 and 8000 tokens."""
    data_768 = {
        "Dense\nbaseline":    {"f1": 0.931, "color": "#1a1a1a"},
        "BigBird":            {"f1": 0.928, "color": "#4e79a7"},
        "BB: globals":        {"f1": 0.926, "color": "#76b7b2"},
        "BB: top-k MMR":      {"f1": 0.927, "color": "#59a14f"},
        "BB: top-k MMR v2":   {"f1": 0.929, "color": "#edc948"},
        "BB: MMR+globals":    {"f1": 0.929, "color": "#b07aa1"},
    }
    data_8000 = {
        "BigBird":                    {"f1": 0.925, "color": "#4e79a7"},
        "BB: MMR+globals":            {"f1": 0.938, "color": "#59a14f"},
        "BB: MMR+globals\n+random":   {"f1": 0.928, "color": "#f28e2b"},
        "BB: MMR+globals\n+teleports":{"f1": 0.940, "color": "#e15759"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for ax, data, title in [
        (axes[0], data_768,  "768 tokens"),
        (axes[1], data_8000, "8,000 tokens"),
    ]:
        names  = list(data.keys())
        f1s    = [data[n]["f1"] for n in names]
        colors = [data[n]["color"] for n in names]
        xs     = np.arange(len(names))
        bars   = ax.bar(xs, f1s, color=colors, width=0.6, edgecolor="white")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylim(0.88, 0.95)
        ax.set_ylabel("IMDb F1" if ax is axes[0] else "")
        ax.set_title(f"BigBird family @ {title}", fontsize=9)
        for bar, f1 in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, f1 + 0.0004,
                    f"{f1:.3f}", ha="center", va="bottom", fontsize=6.8)

    fig.suptitle("BigBird-family IMDb F1 comparison", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig_bigbird_family")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    cx   = load_complexity()
    imdb = load_imdb_task()

    print(f"  complexity: {len(cx)} experiments")
    print(f"  task-level: {len(imdb)} experiments")

    print("\nGenerating figures …")
    fig_loglog(cx)
    fig_linear(cx)
    fig_efficiency(imdb)

    fig_bigbird_family()

    print("\nGenerating appendix table …")
    build_appendix_table(cx, imdb)

    print(f"\nDone. Figures in report/figures/, appendix in report/appendix_omitted.tex")
    print("\nTo include in report.tex add before \\end{document}:")
    print("  \\appendix")
    print("  \\input{appendix_omitted}")


if __name__ == "__main__":
    main()
