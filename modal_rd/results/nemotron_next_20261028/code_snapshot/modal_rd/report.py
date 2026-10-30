"""Pull results from the Modal volume and build presentation charts + summary.

    python modal_rd/report.py            # modal volume get -> modal_rd/results, charts -> modal_rd/figures
"""
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
BEST = "b6_scale_best"


def pull():
    # Volume path results/<tag>.json lands in modal_rd/results/<tag>.json.
    subprocess.run(["modal", "volume", "get", "--force", "bb-vol", "results/", str(HERE)], check=True,
                   stdout=subprocess.DEVNULL)


def load(tag):
    p = RES / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def label(rep):
    if rep["arm"] != "vs":
        return {"dense": "Dense FlashAttention", "exp19": "exp19 Bigger Bird (prev.)"}.get(rep["arm"], rep["arm"])
    a = rep["attn"]
    s = f"sink{a.get('sink', 64)}+win{a.get('window', 4096)}+vert{a.get('vertical', 2048)}"
    if a.get("exact_tail"):
        s += f"+tail{a['exact_tail']}"
    if a.get("route_queries") == "spread":
        s += " (spread)"
    if a.get("route_pool") == "max":
        s += " (max-pool)"
    if a.get("n_route", 64) != 64:
        s += f" (route q={a['n_route']})"
    if a.get("vertical_scope"):
        s += f" (scope {a['vertical_scope']})"
    return s


SCALE_ARMS = (("b6_scale_dense", "Dense FlashAttention", "tab:gray"),
              ("b6_scale_best", "Bigger Bird v3, window 4K", "tab:cyan"),
              ("b9_scale_final", "Bigger Bird v3, window 16K (final)", "tab:blue"))


def scaling(lines):
    reps = [(load(t), n, c) for t, n, c in SCALE_ARMS]
    reps = [x for x in reps if x[0]]
    if len(reps) < 2:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for rep, name, c in reps:
        xs = [r["mean_tokens"] / 1000 for r in rep["runs"]]
        ax1.plot(xs, [r["mean_prefill_s"] for r in rep["runs"]], "o-", label=name, color=c)
        ax2.plot(xs, [100 * r["exact"] / r["n"] for r in rep["runs"]], "o-", label=name, color=c)
    ax1.set(xlabel="prompt tokens (K)", ylabel="prefill time (s), one H100", title="Prefill latency vs context")
    ax2.set(xlabel="prompt tokens (K)", ylabel="exact-number accuracy (%)", ylim=(-5, 105),
            title="NIAH retrieval (6 prompts / length, depth 0.5)")
    for ax in (ax1, ax2):
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "scaling.png", dpi=160)
    d = reps[0][0]
    head = " | ".join(f"{n} prefill s / exact" for _, n, _ in reps[1:])
    lines.append(f"\n## Context scaling (6 prompts / length, depth 0.5, all prompts warmed, KV cache)\n\n"
                 f"| tokens | dense prefill s / exact | {head} |\n|---:|---:|" + "---:|" * (len(reps) - 1))
    for i, rd in enumerate(d["runs"]):
        cells = [f"{rd['mean_prefill_s']:.2f} / {rd['exact']}/{rd['n']}"]
        for rep, _, _ in reps[1:]:
            rb = rep["runs"][i] if i < len(rep["runs"]) else None
            cells.append(f"{rb['mean_prefill_s']:.2f} ({rd['mean_prefill_s'] / rb['mean_prefill_s']:.2f}x) / "
                         f"{rb['exact']}/{rb['n']}" if rb else "-")
        lines.append(f"| {rd['mean_tokens']:.0f} | " + " | ".join(cells) + " |")


def loglog(lines):
    """log-log prefill vs tokens with fitted exponents over >=32K tokens (where attention dominates)."""
    import numpy as np
    reps = [(load(t), n, c) for t, n, c in SCALE_ARMS]
    reps = [x for x in reps if x[0]]
    if len(reps) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    lines.append("\n## Empirical scaling exponent (prefill ~ tokens^k, fitted on 32K-127K)\n\n| arm | k |\n|---|---:|")
    for rep, name, c in reps:
        x = np.array([r["mean_tokens"] for r in rep["runs"]])
        y = np.array([r["mean_prefill_s"] for r in rep["runs"]])
        fit = x >= 30000
        k, b = np.polyfit(np.log(x[fit]), np.log(y[fit]), 1)
        ax.loglog(x / 1000, y, "o-", color=c, label=f"{name}  (slope {k:.2f})")
        lines.append(f"| {name} | {k:.2f} |")
    ax.set(xlabel="prompt tokens (K, log)", ylabel="prefill time (s, log), one H100",
           title="Prefill scaling: slope 2 = quadratic, 1 = linear")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "loglog.png", dpi=160)


def attn_bench(lines):
    rep = load("b12_attn_bench_final")
    if not rep:
        return
    rows = rep["rows"]
    x = [r["tokens"] / 1024 for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.loglog(x, [r["dense_s"] for r in rows], "o-", color="tab:gray", label="Dense FlashAttention (PyTorch SDPA)")
    ax.loglog(x, [r["sparse_s"] for r in rows], "o-", color="tab:blue", label="Bigger Bird v3 (routing + sparse + exact tail)")
    for r, xi in zip(rows, x):
        if r["speedup"] > 1:
            ax.annotate(f"{r['speedup']:.1f}x", (xi, r["sparse_s"]), textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=8, color="tab:blue")
    ax.set(xlabel="tokens (K, log)", ylabel="one attention layer, seconds (log)",
           title="Attention cost per layer, up to 1M tokens (synthetic, H100)")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "attn_bench.png", dpi=160)
    lines.append("\n## Attention-only cost per layer (synthetic Llama-8B shapes, BF16, H100; not a quality result)\n\n"
                 "| tokens | dense | sparse | speedup |\n|---:|---:|---:|---:|")
    lines += [f"| {r['tokens']} | {r['dense_s'] * 1000:.0f} ms | {r['sparse_s'] * 1000:.0f} ms | {r['speedup']:.2f}x |" for r in rows]


TASKS = (("niah_single_2", "single needle, essay haystack"), ("niah_single_3", "single needle, UUID value"),
         ("niah_multikey_1", "4 keys, retrieve 1"), ("niah_multikey_3", "UUID keys in a haystack of UUID distractors"),
         ("niah_multivalue", "4 values for one key"), ("niah_multiquery", "4 keys queried at once"))


def task_runs(task):
    """{context_bucket: (dense_run_list, sparse_run_list)} merging every batch that ran this task."""
    out = {}
    for prefix in ("b12", "b11", "b13"):
        d, b = load(f"{prefix}_{task}_dense"), load(f"{prefix}_{task}_final")
        if prefix == "b13" and task == "niah_single_2":
            d, b = load("b13_dense"), load("b13_final")
        if not (d and b):
            continue
        for rd, rb in zip(d["runs"], b["runs"]):
            key = round(rd["mean_tokens"] / 32768) * 32
            out.setdefault(key, ([], []))
            out[key][0].append(rd)
            out[key][1].append(rb)
    return out


def tasks(lines):
    table = []
    for task, desc in TASKS:
        for ctx, (ds, bs) in sorted(task_runs(task).items()):
            agg = lambda runs: (sum(r["exact"] for r in runs), sum(r["n"] for r in runs),
                                sum(r["partial"] * r["n"] for r in runs) / sum(r["n"] for r in runs))
            table.append((task, desc, ctx, agg(ds), agg(bs)))
    if not table:
        return
    lines.append("\n## Harder RULER NIAH variants (12-24 prompts per cell; none used for tuning)\n\n"
                 "| task | description | context | dense exact | sparse exact | dense recall | sparse recall |\n"
                 "|---|---|---:|---:|---:|---:|---:|")
    for task, desc, ctx, (de, dn, dp), (be, bn, bp) in table:
        lines.append(f"| {task} | {desc} | ~{ctx}K | {de}/{dn} | {be}/{bn} | {dp:.2f} | {bp:.2f} |")
    show = [r for r in table if r[2] in (32, 64) and r[0] in
            ("niah_single_2", "niah_multikey_1", "niah_multivalue", "niah_multiquery")]
    if not show:
        return
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = range(len(show))
    ax.bar([i - .2 for i in x], [100 * r[3][2] for r in show], .4, color="tab:gray", label="Dense FlashAttention")
    ax.bar([i + .2 for i in x], [100 * r[4][2] for r in show], .4, color="tab:blue", label="Bigger Bird v3 (final)")
    ax.set(xticks=list(x), xticklabels=[f"{r[0].replace('niah_', '')}\n~{r[2]}K" for r in show],
           ylabel="answer recall (%)", ylim=(0, 105), title="Harder needle tasks, not used for tuning (12 prompts each)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "tasks.png", dpi=160)


CONFIRM = (("b6_confirm_dense", "b6_confirm_best", "b9_confirm_final"), ("b9_confirm_dense_d01", "b9_confirm_final_d01"))


def confirmation(lines):
    reps = [load(t) for group in CONFIRM for t in group]
    if not any(reps):
        return
    lines.append("\n## Confirmation (30 prompts per context; prompts 0-5 were used for tuning at ~127K)\n\n"
                 "| context | depth | arm | exact all-30 | exact held-out 6-29 | last digit | mean prefill s |\n|---|---:|---|---:|---:|---:|---:|")
    for rep in filter(None, reps):
        for r in rep["runs"]:
            held = [e for e in r["examples"] if e["idx"] >= 6]
            lines.append(f"| ~{r['mean_tokens'] / 1000:.0f}K | {r['depth']} | {label(rep)} | {r['exact']}/{r['n']} | "
                         f"{sum(e['exact'] for e in held)}/{len(held)} | {r['last_digit']}/{r['n']} | {r['mean_prefill_s']:.2f} |")


def exact_at(tag, depth=0.5, nbytes=512000):
    rep = load(tag)
    for r in (rep or {}).get("runs", []):
        if r["depth"] == depth and r["bytes"] == nbytes:
            return r["exact"]
    return None


DEPTH_ARMS = (("Dense FlashAttention", "tab:gray", "b6_depth_dense", "b6_scale_dense"),
              ("Bigger Bird v3, window 4K", "tab:cyan", "b6_depth_best", "b6_scale_best"),
              ("Bigger Bird v3, window 16K (final)", "tab:blue", "b9_depth_final", "b9_scale_final"))


def depth(lines):
    arms = []
    for name, color, tag, mid in DEPTH_ARMS:
        rep = load(tag)
        if rep:
            vals = {r["depth"]: r["exact"] for r in rep["runs"] if r["bytes"] == 512000 and r["n"] == 6}
            if exact_at(mid) is not None:
                vals[0.5] = exact_at(mid)
            arms.append((name, color, vals))
    if len(arms) < 2:
        return
    depths = sorted(set.intersection(*(set(v) for _, _, v in arms)))
    lines.append("\n## Needle depth @~127K (6 prompts / depth)\n\n| depth | " + " | ".join(n for n, _, _ in arms) +
                 " |\n|---:|" + "---:|" * len(arms))
    lines += [f"| {k} | " + " | ".join(f"{v[k]}/6" for _, _, v in arms) + " |" for k in depths]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = .8 / len(arms)
    for j, (name, color, vals) in enumerate(arms):
        ax.bar([i + (j - (len(arms) - 1) / 2) * w for i in range(len(depths))], [vals[k] for k in depths], w,
               label=name, color=color)
    x = range(len(depths))
    ax.set(xticks=list(x), xticklabels=[str(k) for k in depths], xlabel="needle depth (fraction of prompt)",
           ylabel="exact / 6", ylim=(0, 6.5), title="Needle depth sensitivity @~127K")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "depth.png", dpi=160)


def ablations(lines):
    vert = [(64, "b6_abl_v64_t256"), (256, "b6_abl_v256_t256"), (512, "b5_last_v512_t256"), (1024, "b4_v1k_w4k_t256"),
            (2048, "b3_v2k_w4k_t256"), (3072, "b4_v3k_w4k_t256")]
    tail = [(0, "b6_abl_v512_t0"), (64, "b6_abl_v512_t64"), (256, "b5_last_v512_t256")]
    vert = [(v, exact_at(t)) for v, t in vert if exact_at(t) is not None]
    tail = [(v, exact_at(t)) for v, t in tail if exact_at(t) is not None]
    if not (vert and tail):
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot([v for v, _ in vert], [e for _, e in vert], "o-", color="tab:blue")
    a1.set(xscale="log", xlabel="routed columns per head (window 4096, exact tail 256)", ylabel="exact / 6",
           ylim=(-.3, 6.3), title="Routed-column budget: a sweet spot")
    a2.bar([str(v) for v, _ in tail], [e for _, e in tail], color="tab:blue")
    a2.set(xlabel="exact tail queries (vertical 512, window 4096)", ylabel="exact / 6", ylim=(0, 6.5),
           title="Exact question tail")
    for ax in (a1, a2):
        ax.grid(alpha=.3)
        ax.axhline(6, color="tab:gray", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(FIG / "ablations.png", dpi=160)
    lines.append("\n## Ablations @~127K, depth 0.5, first 6 prompts\n\n| routed columns | exact |\n|---:|---:|")
    lines += [f"| {v} | {e}/6 |" for v, e in vert]
    lines.append("\n| exact tail | exact |\n|---:|---:|")
    lines += [f"| {v} | {e}/6 |" for v, e in tail]


def sweep(lines):
    rows = []
    for p in sorted(RES.rglob("b[1-6]_*.json")):
        rep = json.loads(p.read_text())
        if "runs" not in rep:
            continue
        for r in rep["runs"]:
            if r["bytes"] == 512000 and r["depth"] == 0.5 and r["n"] == 6:
                rows.append((p.stem, label(rep), r["exact"], r["mean_prefill_s"]))
    if not rows:
        return
    lines.append("\n## All arms @~127K, first 6 prompts (tuning set)\n\n| run | arm | exact | prefill s |\n|---|---|---:|---:|")
    lines += [f"| {t} | {l} | {e}/6 | {s:.2f} |" for t, l, e, s in rows]


def recall(lines):
    rep = load("b1_recall_127k")
    if not rep:
        return
    layers = rep["prompts"][0]["layers"]
    show = ["exp19like_l128_q4096", "stream_sink64_l4096", "vs_s64_l4096_v2048", "oracle_top2048"]
    names = {"exp19like_l128_q4096": "exp19-like (local128 + last-query top4096)",
             "stream_sink64_l4096": "sink64 + local4096", "vs_s64_l4096_v2048": "sink64 + local4096 + vertical2048",
             "oracle_top2048": "oracle top-2048 per query"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in show:
        ax.plot([l[name]["cos_mid"] for l in layers], label=names[name])
    ax.set(xlabel="layer", ylabel="cosine vs dense attention output", title="Mid-prompt attention fidelity @~127K")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "recall_layers.png", dpi=160)


def main():
    pull()
    FIG.mkdir(exist_ok=True)
    lines = ["# Bigger Bird v3 Modal R&D results"]
    for f in (confirmation, tasks, scaling, loglog, attn_bench, depth, ablations, sweep, recall):
        f(lines)
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
