#!/usr/bin/env python3
"""
Generate publication-quality figures for the Bigger Bird report.
All 15 experiments (0-15) are represented wherever data exists.

Usage:
  python viz/report_figures.py

Outputs to report/figures/ (created if absent).
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

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7.5,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
})

# ── colours & labels ──────────────────────────────────────────────────────────
EXP_META = {
    "exp_0_baseline":         {"label": "0  Dense Baseline",         "color": "#000000", "ls": "-",  "marker": "s"},
    "exp_1_deepseek_topk":    {"label": "1  DeepSeek Top-k",         "color": "#1f77b4", "ls": "-",  "marker": "o"},
    "exp_2_lightning_hybrid": {"label": "2  Lightning Hybrid",       "color": "#ff7f0e", "ls": "--", "marker": "^"},
    "exp_3_dynamic_globals":  {"label": "3  Dynamic Globals",        "color": "#2ca02c", "ls": "--", "marker": "v"},
    "exp_4_pbs_attn":         {"label": "4  PBS Attn",               "color": "#d62728", "ls": "-",  "marker": "D"},
    "exp_5_bigger_bird":      {"label": "5  Bigger Bird (unified)",  "color": "#9467bd", "ls": ":",  "marker": "P"},
    "exp_6_deepseek_pbs":     {"label": "6  DeepSeek+PBS",           "color": "#8c564b", "ls": "--", "marker": "X"},
    "exp_7_layer_adaptive":   {"label": "7  Layer-Adaptive",         "color": "#e377c2", "ls": "-",  "marker": "o"},
    "exp_8_token_drop":       {"label": "8  Token Drop",             "color": "#7f7f7f", "ls": "-",  "marker": "o"},
    "exp_9_attn_specul":      {"label": "9  Attn Speculation",       "color": "#bcbd22", "ls": ":",  "marker": "*"},
    "exp_10_gqa_sparse":      {"label": "10 GQA+Sparse",             "color": "#17becf", "ls": "-",  "marker": "o"},
    "exp_11_nsa":             {"label": "11 NSA",                    "color": "#1a9850", "ls": ":",  "marker": "h"},
    "exp_12_s2_hhst":         {"label": "12 S2-HHST",                "color": "#d73027", "ls": ":",  "marker": "H"},
    "exp_13_dynamic_context": {"label": "13 Dynamic Context",        "color": "#f781bf", "ls": "-",  "marker": "o"},
    "exp_14_token_drop_deepseek": {"label": "14 TokenDrop+DeepSeek", "color": "#a65628", "ls": "--", "marker": "^"},
}

def _short(label):
    parts = label.split(None, 2)
    return parts[0] + (" " + parts[2] if len(parts) > 2 else "")

SHORT_LABEL = {k: _short(v["label"]) for k, v in EXP_META.items()}


# ── data loaders ──────────────────────────────────────────────────────────────

def load_complexity():
    p = os.path.join(BDIR, "complexity_results.json")
    with open(p) as f:
        d = json.load(f)
    rows = defaultdict(dict)   # rows[exp_name][seq] = time_ms or None
    for r in d["results"]:
        v = None if r.get("oom") else r.get("time_ms")
        rows[r["exp_name"]][r["seq_length"]] = v
    return rows


def load_long_context_sweep():
    """Return dict[exp_name][seq] = record from all long_context_sweep_*.json files."""
    records = {}
    for fp in sorted(glob.glob(os.path.join(BDIR, "long_context_sweep_*.json"))):
        with open(fp) as f:
            d = json.load(f)
        items = d if isinstance(d, list) else d.get("results", [])
        for r in items:
            seq = r.get("seq", r.get("seq_length", 0))
            if not seq:
                continue
            exp = r.get("exp_name", "")
            key = (exp, seq)
            if key not in records:
                records[key] = r
    result = defaultdict(dict)
    for (exp, seq), r in records.items():
        result[exp][seq] = r
    return result


def load_efficiency_results():
    """Return list of dicts from benchmarks/efficiency_results.json."""
    with open(os.path.join(BDIR, "efficiency_results.json")) as f:
        d = json.load(f)
    return d.get("results", [])


def load_task_eval_jsons():
    """Load all eval_*.json files from per-experiment benchmark dirs."""
    records = defaultdict(dict)
    for exp_dir in os.listdir(BDIR):
        dp = os.path.join(BDIR, exp_dir)
        if not os.path.isdir(dp):
            continue
        for fp in glob.glob(os.path.join(dp, "eval_*.json")):
            try:
                with open(fp) as f:
                    d = json.load(f)
                meta = d["experiment_metadata"]
                perf = d["performance_metrics"]
                seq = meta["dataset_info"].get("max_seq_len", 0)
                # skip dynamic-padding era runs (seq ~166)
                if seq < 250:
                    continue
                exp = meta.get("name", exp_dir)
                # normalise name: strip trailing _seq* variants
                base = exp_dir.split("_seq")[0]
                rec = {
                    "seq": seq,
                    "f1": perf["eval"].get("eval_f1"),
                    "accuracy": perf["eval"].get("eval_accuracy"),
                    "train_time_s": perf.get("training_time_seconds"),
                    "peak_mem_mb": perf.get("peak_memory_mb") or
                                   meta.get("environment", {}).get("peak_memory_mb"),
                    "inference_ms": perf.get("inference_latency_ms"),
                }
                # keep latest / best f1 per (exp, seq)
                old = records[base].get(seq)
                if old is None or (rec["f1"] or 0) > (old["f1"] or 0):
                    records[base][seq] = rec
            except Exception:
                pass
    return records


# ── Figure 1: log-log kernel time ─────────────────────────────────────────────

def fig_loglog_kernel(cx):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    # exps that have 128–32768 data
    long_exps = ["exp_0_baseline", "exp_1_deepseek_topk",
                 "exp_7_layer_adaptive", "exp_10_gqa_sparse"]
    # all kernel-benchmarked exps
    all_kernel_exps = sorted(cx.keys())

    for ename in all_kernel_exps:
        m = EXP_META.get(ename, {})
        seq_times = sorted((s, t) for s, t in cx[ename].items() if t is not None)
        if not seq_times:
            continue
        xs, ys = zip(*seq_times)
        lw = 2.0 if ename in long_exps else 1.0
        ax.plot(xs, ys,
                label=m.get("label", ename),
                color=m.get("color", "gray"),
                linestyle=m.get("ls", "-"),
                marker=m.get("marker", "o"),
                linewidth=lw,
                markersize=4 if ename in long_exps else 3,
                alpha=1.0 if ename in long_exps else 0.55)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Attention forward time (ms)")
    ax.set_title("Log–log attention-kernel time — all benchmarked experiments")

    # OOM annotation for dense at 32768
    ax.annotate("Dense OOM\n@32k", xy=(32768, 95), fontsize=7, color="black",
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="black"),
                xytext=(20000, 140))

    leg = ax.legend(ncol=2, loc="upper left",
                    framealpha=0.85, edgecolor="none",
                    handlelength=2.2, columnspacing=1.0)
    fig.tight_layout()
    out = os.path.join(OUT, "fig_loglog_kernel.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


# ── Figure 2: bar chart — kernel time at 512 / 2048 / 4096 ───────────────────

def fig_kernel_bars(cx):
    seq_lengths = [512, 2048, 4096]
    exps = sorted(cx.keys(), key=lambda e: int(e.split("_")[1]) if e.split("_")[1].isdigit() else 99)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.2), sharey=False)
    for ax, seq in zip(axes, seq_lengths):
        names, vals, colors = [], [], []
        for e in exps:
            t = cx[e].get(seq)
            if t is not None:
                names.append(EXP_META.get(e, {}).get("label", e)[:2].strip())
                vals.append(t)
                colors.append(EXP_META.get(e, {}).get("color", "gray"))
        xs = np.arange(len(names))
        bars = ax.bar(xs, vals, color=colors, width=0.7, edgecolor="white", linewidth=0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"seq = {seq:,}", fontsize=9)
        ax.set_ylabel("Time (ms)" if seq == 512 else "")

    fig.suptitle("Attention-kernel time by experiment (all benchmarked exps)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUT, "fig_kernel_bars.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


# ── Figure 3: F1 vs sequence length (task level) ─────────────────────────────

def fig_f1_vs_seq(lc, task_evals):
    """Combine long-context sweep + per-exp eval JSONs."""
    # build: data[exp_name][seq] = f1
    data = defaultdict(dict)

    # from long-context sweeps
    for exp, seq_map in lc.items():
        for seq, r in seq_map.items():
            f1 = r.get("f1")
            if f1 is not None and not r.get("oom"):
                data[exp][seq] = f1

    # from per-exp eval JSONs (fills in exp_9 thru exp_14 etc.)
    for base, seq_map in task_evals.items():
        for seq, r in seq_map.items():
            f1 = r.get("f1")
            if f1 is not None:
                existing = data[base].get(seq)
                if existing is None:
                    data[base][seq] = f1

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for ename in sorted(data, key=lambda e: int(e.split("_")[1]) if len(e.split("_")) > 1 and e.split("_")[1].isdigit() else 99):
        m = EXP_META.get(ename, {})
        pts = sorted(data[ename].items())
        if not pts:
            continue
        xs, ys = zip(*pts)
        if len(xs) == 1:
            ax.scatter(xs, ys, color=m.get("color","gray"), marker=m.get("marker","o"),
                       s=40, label=m.get("label", ename), zorder=4)
        else:
            ax.plot(xs, ys,
                    label=m.get("label", ename),
                    color=m.get("color","gray"),
                    linestyle=m.get("ls","-"),
                    marker=m.get("marker","o"),
                    markersize=4, linewidth=1.4)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("IMDb F1")
    ax.set_title("IMDb F1 by experiment across sequence lengths")
    ax.axhline(y=0.0, color="lightcoral", lw=0.8, ls=":")
    ax.legend(ncol=2, loc="lower right", framealpha=0.85,
              edgecolor="none", fontsize=7)
    fig.tight_layout()
    out = os.path.join(OUT, "fig_f1_vs_seq.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


# ── Figure 4: peak memory vs seq length ───────────────────────────────────────

def fig_memory_vs_seq(lc, task_evals):
    data = defaultdict(dict)
    for exp, seq_map in lc.items():
        for seq, r in seq_map.items():
            v = r.get("peak_mem_mb")
            if v and not r.get("oom"):
                data[exp][seq] = v / 1024  # → GB
    for base, seq_map in task_evals.items():
        for seq, r in seq_map.items():
            v = r.get("peak_mem_mb")
            if v is not None and data[base].get(seq) is None:
                data[base][seq] = v / 1024

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for ename in sorted(data, key=lambda e: int(e.split("_")[1]) if len(e.split("_")) > 1 and e.split("_")[1].isdigit() else 99):
        m = EXP_META.get(ename, {})
        pts = sorted(data[ename].items())
        if not pts:
            continue
        xs, ys = zip(*pts)
        kw = dict(color=m.get("color","gray"), linestyle=m.get("ls","-"),
                  marker=m.get("marker","o"), markersize=4, linewidth=1.4,
                  label=m.get("label", ename))
        if len(xs) == 1:
            ax.scatter(xs, ys, color=m.get("color","gray"),
                       marker=m.get("marker","o"), s=40,
                       label=m.get("label", ename), zorder=4)
        else:
            ax.plot(xs, ys, **kw)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Peak device memory (GB)")
    ax.set_title("Peak memory by experiment across sequence lengths")
    ax.legend(ncol=2, loc="upper left", framealpha=0.85, edgecolor="none", fontsize=7)
    fig.tight_layout()
    out = os.path.join(OUT, "fig_memory_vs_seq.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


# ── Figure 5: inference latency vs seq length ─────────────────────────────────

def fig_latency_vs_seq(lc, task_evals):
    data = defaultdict(dict)
    for exp, seq_map in lc.items():
        for seq, r in seq_map.items():
            v = r.get("inference_ms")
            if v and not r.get("oom"):
                data[exp][seq] = v
    for base, seq_map in task_evals.items():
        for seq, r in seq_map.items():
            v = r.get("inference_ms")
            if v is not None and data[base].get(seq) is None:
                data[base][seq] = v

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for ename in sorted(data, key=lambda e: int(e.split("_")[1]) if len(e.split("_")) > 1 and e.split("_")[1].isdigit() else 99):
        m = EXP_META.get(ename, {})
        pts = sorted(data[ename].items())
        if not pts:
            continue
        xs, ys = zip(*pts)
        kw = dict(color=m.get("color","gray"), linestyle=m.get("ls","-"),
                  marker=m.get("marker","o"), markersize=4, linewidth=1.4,
                  label=m.get("label", ename))
        if len(xs) == 1:
            ax.scatter(xs, ys, color=m.get("color","gray"),
                       marker=m.get("marker","o"), s=40,
                       label=m.get("label", ename), zorder=4)
        else:
            ax.plot(xs, ys, **kw)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Inference latency (ms/sequence)")
    ax.set_title("Inference latency by experiment across sequence lengths")
    ax.legend(ncol=2, loc="upper left", framealpha=0.85, edgecolor="none", fontsize=7)
    fig.tight_layout()
    out = os.path.join(OUT, "fig_latency_vs_seq.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


# ── Figure 6: quality–efficiency scatter at 1024 tokens ──────────────────────

def fig_quality_efficiency(lc, task_evals, eff_results):
    """F1 vs inference latency at 1024 tokens; bubble = log(train time)."""
    SEQ = 1024
    recs = {}  # exp -> {f1, lat, mem, train}

    # from efficiency_results.json
    for r in eff_results:
        if r.get("seq_length") == SEQ and not r.get("oom"):
            recs[r["exp_name"]] = {
                "f1":   r.get("f1"),
                "lat":  r.get("inference_latency_ms"),
                "mem":  r.get("peak_memory_mb"),
                "train":r.get("train_time_s"),
            }

    # supplement from sweep / eval JSONs
    for exp, seq_map in lc.items():
        if exp not in recs and SEQ in seq_map and not seq_map[SEQ].get("oom"):
            r = seq_map[SEQ]
            recs[exp] = {"f1": r.get("f1"), "lat": r.get("inference_ms"),
                         "mem": r.get("peak_mem_mb"), "train": r.get("train_time_s")}

    for base, seq_map in task_evals.items():
        if base not in recs and SEQ in seq_map:
            r = seq_map[SEQ]
            recs[base] = {"f1": r.get("f1"), "lat": r.get("inference_ms"),
                          "mem": r.get("peak_mem_mb"), "train": r.get("train_time_s")}

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for ename, d in sorted(recs.items()):
        f1  = d.get("f1")
        lat = d.get("lat")
        if f1 is None or lat is None:
            continue
        m   = EXP_META.get(ename, {})
        train = d.get("train") or 100
        size  = max(20, min(400, 15 * np.log(train + 1)))
        ax.scatter(lat, f1, s=size,
                   color=m.get("color","gray"),
                   marker=m.get("marker","o"),
                   edgecolors="white", linewidths=0.5, zorder=4,
                   label=m.get("label", ename))
        # label dot with exp number
        ax.annotate(m.get("label","?")[:2].strip(),
                    (lat, f1), fontsize=6.5, ha="left", va="bottom",
                    xytext=(3, 2), textcoords="offset points")

    ax.set_xlabel("Inference latency at seq 1024 (ms/sequence)")
    ax.set_ylabel("IMDb F1 at seq 1024")
    ax.set_title("Quality vs latency at 1024 tokens\n(bubble size ∝ log training time)")
    ax.legend(ncol=1, loc="lower right", framealpha=0.8, edgecolor="none",
              fontsize=6.5, markerscale=0.9)
    fig.tight_layout()
    out = os.path.join(OUT, "fig_quality_efficiency_1024.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


# ── Figure 7: OOM / completion heatmap ───────────────────────────────────────

def fig_oom_heatmap(lc, eff_results):
    """Heatmap: green = completed, red = OOM, white = not run."""
    seq_order = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    exp_order  = [f"exp_{i}" for i in range(16) if i != 15]
    # Map short exp names to full
    exp_full = {
        "exp_0": "exp_0_baseline",
        "exp_1": "exp_1_deepseek_topk",
        "exp_2": "exp_2_lightning_hybrid",
        "exp_3": "exp_3_dynamic_globals",
        "exp_4": "exp_4_pbs_attn",
        "exp_5": "exp_5_bigger_bird",
        "exp_6": "exp_6_deepseek_pbs",
        "exp_7": "exp_7_layer_adaptive",
        "exp_8": "exp_8_token_drop",
        "exp_9": "exp_9_attn_specul",
        "exp_10": "exp_10_gqa_sparse",
        "exp_11": "exp_11_nsa",
        "exp_12": "exp_12_s2_hhst",
        "exp_13": "exp_13_dynamic_context",
        "exp_14": "exp_14_token_drop_deepseek",
    }

    # build status matrix: 1=ok, 0=oom, NaN=not run
    status = {}
    # from long-context sweep
    for exp_full_name, seq_map in lc.items():
        for short, full in exp_full.items():
            if full == exp_full_name:
                for seq, r in seq_map.items():
                    status[(short, seq)] = 0 if r.get("oom") else 1

    # from efficiency_results
    for r in eff_results:
        exp_name = r["exp_name"]
        for short, full in exp_full.items():
            if full == exp_name:
                seq = r["seq_length"]
                if (short, seq) not in status:
                    status[(short, seq)] = 0 if r.get("oom") else 1

    # Hard-code known OOM for dense at 65536
    status[("exp_0", 65536)] = 0
    status[("exp_1", 65536)] = 0

    # known single-point exp_11, exp_12 (512 tokens only)
    status[("exp_11", 512)] = 1
    status[("exp_12", 256)] = 1

    matrix = []
    for e in exp_order:
        row = []
        for s in seq_order:
            v = status.get((e, s), float("nan"))
            row.append(v)
        matrix.append(row)
    matrix = np.array(matrix, dtype=float)

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#ef4444", "#22c55e"])  # red=OOM, green=ok

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    # only plot non-NaN cells
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(seq_order)))
    ax.set_xticklabels([f"{s:,}" for s in seq_order], rotation=45, ha="right")
    ax.set_yticks(range(len(exp_order)))
    ax.set_yticklabels([f"Exp {e.replace('exp_','')}" for e in exp_order])
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_title("Run completion matrix (green = completed, red = OOM, white = not run)")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#22c55e", label="Completed"),
                       Patch(facecolor="#ef4444", label="OOM"),
                       Patch(facecolor="white", edgecolor="gray", label="Not run")]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9,
              fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(OUT, "fig_oom_heatmap.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


# ── Figure 8: BigBird family comparison ───────────────────────────────────────

def fig_bigbird_family():
    """Bar chart comparing BigBird family at 768 and 8000 tokens."""
    # Data from report tables
    data_768 = {
        "Dense\nbaseline":       {"f1": 0.931, "train": 216.7},
        "BigBird":               {"f1": 0.928, "train": 2766.6},
        "BB: globals":           {"f1": 0.926, "train": 3055.8},
        "BB: top-k MMR":         {"f1": 0.927, "train": 2902.7},
        "BB: top-k MMR v2":      {"f1": 0.929, "train": 2929.3},
        "BB: MMR+globals":       {"f1": 0.929, "train": 3184.5},
    }
    data_8000 = {
        "BigBird":               {"f1": 0.925, "train": 21504.5},
        "BB: MMR+globals":       {"f1": 0.938, "train": 7484.3},
        "BB: MMR+globals+random":{"f1": 0.928, "train": 5832.9},
        "BB: MMR+globals+teleports": {"f1": 0.940, "train": 4348.7},
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    colors_768  = ["#000000","#4e79a7","#76b7b2","#59a14f","#edc948","#b07aa1"]
    colors_8000 = ["#4e79a7","#59a14f","#f28e2b","#e15759"]

    for ax, data, colors, title in [
        (axes[0], data_768,  colors_768,  "768 tokens"),
        (axes[1], data_8000, colors_8000, "8 000 tokens"),
    ]:
        names = list(data.keys())
        f1s   = [data[n]["f1"]   for n in names]
        xs    = np.arange(len(names))
        bars  = ax.bar(xs, f1s, color=colors, width=0.6, edgecolor="white")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7.5)
        ax.set_ylim(0.88, 0.95)
        ax.set_ylabel("IMDb F1" if ax is axes[0] else "")
        ax.set_title(f"BigBird family @ {title}")

        # annotate bars
        for bar, f1 in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, f1 + 0.0005,
                    f"{f1:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("BigBird-family IMDb F1 comparison", fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUT, "fig_bigbird_family.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    cx         = load_complexity()
    lc         = load_long_context_sweep()
    eff        = load_efficiency_results()
    task_evals = load_task_eval_jsons()

    print(f"  complexity: {len(cx)} experiments")
    print(f"  long-context sweep: {len(lc)} experiments")
    print(f"  efficiency_results: {len(eff)} records")
    print(f"  task eval JSONs: {len(task_evals)} experiment dirs")

    print("\nGenerating figures …")
    fig_loglog_kernel(cx)
    fig_kernel_bars(cx)
    fig_f1_vs_seq(lc, task_evals)
    fig_memory_vs_seq(lc, task_evals)
    fig_latency_vs_seq(lc, task_evals)
    fig_quality_efficiency(lc, task_evals, eff)
    fig_oom_heatmap(lc, eff)
    fig_bigbird_family()

    print(f"\nAll figures saved to {OUT}/")
    print("Include in LaTeX with:")
    print("  \\includegraphics[width=\\linewidth]{figures/fig_loglog_kernel}")


if __name__ == "__main__":
    main()
