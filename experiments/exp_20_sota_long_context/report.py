"""Summarise exp 20 results: paired accuracy, speed and memory per task, domain and prompt length.

  python -m experiments.exp_20_sota_long_context.report --tag a100_120k [--out experiments/exp_20_sota_long_context/RESULTS.md]

Only examples that every arm finished without OOM are compared. "agree" is how often an arm picks the same letter
as dense; McNemar p is the exact two-sided test on the discordant pairs against dense.
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from experiments.exp_20_sota_long_context.run import RESULTS

LEN_BINS = [(0, 16448, "≤16K (bb = dense)"), (16448, 32768, "16-32K"), (32768, 65536, "32-64K"),
            (65536, 98304, "64-96K"), (98304, 10**9, "96K+")]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(tag):
    d = RESULTS / tag
    meta = json.loads((d / "meta.json").read_text())
    by_item = defaultdict(dict)
    for path in sorted(d.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            by_item[(r["task"], r["id"])][r["arm"]] = r
    return meta, by_item


def pct(x):
    return f"{100 * x:.1f}"


def acc_table(title, items, arms, key):
    """items: list of {arm: rec}; key(rec) -> row label."""
    rows = defaultdict(list)
    for it in items:
        rows[key(it[arms[0]])].append(it)
        rows["**All**"].append(it)
    head = ["", "n"] + [f"{a} acc %" for a in arms] + [f"Δ {a} pp" for a in arms[1:]] + \
           [f"agree {a} %" for a in arms[1:]] + [f"McNemar p ({a})" for a in arms[1:]]
    out = [f"### {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    order = {lab: i for i, (_, _, lab) in enumerate(LEN_BINS)}
    for label, its in sorted(rows.items(), key=lambda kv: (kv[0] == "**All**", order.get(kv[0], 0), kv[0])):
        n = len(its)
        acc = {a: sum(it[a]["correct"] for it in its) / n for a in arms}
        cells = [label, str(n)] + [pct(acc[a]) for a in arms]
        cells += [f"{100 * (acc[a] - acc[arms[0]]):+.1f}" for a in arms[1:]]
        cells += [pct(sum(it[a]["pred"] == it[arms[0]]["pred"] for it in its) / n) for a in arms[1:]]
        for a in arms[1:]:
            b = sum(it[arms[0]]["correct"] and not it[a]["correct"] for it in its)
            c = sum(it[a]["correct"] and not it[arms[0]]["correct"] for it in its)
            cells.append(f"{mcnemar(b, c):.2f} ({c}↑/{b}↓)")
        out.append("| " + " | ".join(cells) + " |")
    return out + [""]


def eff_table(items, arms):
    head = ["prompt length", "n", "mean tokens"] + [f"{a} prefill s" for a in arms] + \
           [f"{a} speedup" for a in arms[1:]] + [f"{a} decode ms/tok" for a in arms] + [f"{a} peak GB" for a in arms]
    out = ["### Speed and memory by prompt length", "",
           "Prefill = one CUDA-timed forward over the prompt; peak = max allocated during prefill + decode; "
           "speedup = dense prefill / arm prefill (ratio of means).", "",
           "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for lo, hi, label in LEN_BINS + [(0, 10**9, "**All**")]:
        its = [it for it in items if lo < it[arms[0]]["tokens"] <= hi]
        if not its:
            continue
        mean = lambda a, k: statistics.fmean(it[a][k] for it in its)
        cells = [label, str(len(its)), f"{mean(arms[0], 'tokens'):.0f}"]
        cells += [f"{mean(a, 'prefill_s'):.2f}" for a in arms]
        cells += [f"{mean(arms[0], 'prefill_s') / mean(a, 'prefill_s'):.2f}x" for a in arms[1:]]
        cells += [f"{mean(a, 'decode_ms_per_token'):.1f}" if its[0][a].get("decode_ms_per_token") else "—" for a in arms]
        cells += [f"{mean(a, 'peak_gb'):.1f} (max {max(it[a]['peak_gb'] for it in its):.1f})" for a in arms]
        out.append("| " + " | ".join(cells) + " |")
    return out + [""]


def fidelity(items, arms):
    """How far each sparse arm's answer distribution moves from dense (more sensitive than accuracy)."""
    out = ["### Answer-distribution fidelity vs dense", "",
           "TV = total-variation distance between the arms' A-D distributions; Δp(gold) = arm minus dense "
           "probability on the gold letter.", "",
           "| arm | mean TV | median TV | mean Δp(gold) | mean dense p(gold) |", "|---|---|---|---|---|"]
    gi = "ABCD".index
    for a in arms[1:]:
        tv = [0.5 * sum(abs(x - y) for x, y in zip(it[a]["probs"], it[arms[0]]["probs"])) for it in items]
        dg = [it[a]["probs"][gi(it[a]["gold"])] - it[arms[0]]["probs"][gi(it[a]["gold"])] for it in items]
        pg = [it[arms[0]]["probs"][gi(it[a]["gold"])] for it in items]
        out.append(f"| {a} | {statistics.fmean(tv):.4f} | {statistics.median(tv):.4f} | {statistics.fmean(dg):+.4f} "
                   f"| {statistics.fmean(pg):.3f} |")
    return out + [""]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--cap", default="", help="first N examples per task in run order, e.g. codedebug=100,lbv2=100")
    a = p.parse_args()
    meta, by_item = load(a.tag)
    caps = {t: int(n) for t, n in (c.split("=") for c in filter(None, a.cap.split(",")))}
    seen = defaultdict(int)
    for key in list(by_item):  # dict order = file order = the run's seeded shuffle
        seen[key[0]] += 1
        if key[0] in caps and seen[key[0]] > caps[key[0]]:
            del by_item[key]
    arms = list(meta["arms"])
    complete = [it for it in by_item.values() if all(x in it and not it[x].get("oom") for x in arms)]
    oom = defaultdict(int)
    for it in by_item.values():
        for x, r in it.items():
            oom[x] += bool(r.get("oom"))
    lines = [f"# Exp 20 results: `{a.tag}`", "",
             f"GPU {meta['gpu']}, torch {meta['torch']}, max prompt {meta['max_tokens']} tokens (middle-truncated), "
             f"MLP chunk {meta['mlp_chunk']}, {meta['decode_tokens']} decode tokens"
             + (f", capped at {a.cap}" if a.cap else "") + ". Arms: "
             + "; ".join(f"`{k}` = {json.dumps(v) if v else 'HF SDPA dense'}" for k, v in meta["arms"].items()), "",
             f"{len(complete)} examples complete in every arm; OOM per arm: {dict(oom) or 'none'}.", ""]
    by_task = defaultdict(list)
    for it in complete:
        by_task[it[arms[0]]["task"]].append(it)
    lines += acc_table("Accuracy by task", complete, arms, lambda r: r["task"])
    if by_task.get("lbv2"):
        lines += acc_table("LongBench v2 by domain", by_task["lbv2"], arms, lambda r: r["group"])
        lines += acc_table("LongBench v2 by difficulty", by_task["lbv2"], arms, lambda r: r["difficulty"])
    lines += acc_table("Accuracy by prompt length (after truncation)", complete, arms,
                       lambda r: next(lab for lo, hi, lab in LEN_BINS if lo < r["tokens"] <= hi))
    lines += fidelity(complete, arms)
    lines += eff_table(complete, arms)
    text = "\n".join(lines)
    print(text)
    (RESULTS / a.tag / "RESULTS.md").write_text(text + "\n")
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
