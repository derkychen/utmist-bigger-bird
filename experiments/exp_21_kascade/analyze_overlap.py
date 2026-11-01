"""Phase 0 analysis: cross-layer similarity of v3 routed columns + Kascade anchor DP.

For layers i (anchor) and j (reuser), averaged over prompts:
  same_head[i, j]  mean_h |R_i,h ∩ R_j,h| / V           (reuse with head h -> head h)
  best_head[i, j]  mean_h max_h' |R_i,h' ∩ R_j,h| / V   (oracle head matching; upper bound)
  pooled[i, j]     |U_i ∩ U_j| / |U_j|, U = union over heads
Chance level for same_head is V / (n - sink - window).
Needle coverage per layer = fraction of heads whose V columns contain the whole number span.
Under reuse, layer j inherits its anchor's coverage.

Anchors are chosen by dynamic programming on same_head (layer 0 always an anchor; every
layer reuses the nearest anchor at or before it), maximising the summed similarity.

  python -m experiments.exp_21_kascade.analyze_overlap --tag p0_16k
Also saves the mean head-by-head overlap tensor used for Kascade head remapping.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.exp_21_kascade.common import OUT as EXP_OUT

OUT = EXP_OUT / "phase0"


def masks(routes, n):
    L, H, V = routes.shape
    m = torch.zeros(L, H, n, dtype=torch.float32)
    m.scatter_(2, torch.from_numpy(routes).long(), 1.0)
    return m


def pair_metrics(m, V):
    L, H, n = m.shape
    inter = torch.einsum("ahn,bgn->abhg", m, m)                    # [L, L, H, H] |R_a,h ∩ R_b,g|
    same = inter.diagonal(dim1=2, dim2=3).mean(-1) / V             # [L, L]
    best = inter.amax(2).mean(-1) / V                              # anchor head chosen per reuser head
    u = m.amax(1)                                                  # [L, n]
    pooled = (u @ u.T) / u.sum(-1)[None, :]                        # |U_i ∩ U_j| / |U_j|
    return same.numpy(), best.numpy(), pooled.numpy(), (inter / V).numpy()


def coverage(routes, number):
    lo, hi = number
    return ((routes >= lo) & (routes < hi)).sum(-1).__ge__(hi - lo).mean(-1)   # [L]


def choose_anchors(S, k):
    """Max sum_j S[anchor(j), j] with k anchors, anchor(j) = last anchor <= j, layer 0 an anchor."""
    L = S.shape[0]
    seg = np.full((L, L + 1), -np.inf)                             # seg[a, e] = sum_{l in [a, e)} S[a, l]
    for a in range(L):
        seg[a, a + 1:] = np.cumsum(S[a, a:])
    best = np.full((k + 1, L + 1), -np.inf)
    arg = np.zeros((k + 1, L + 1), dtype=int)
    best[1, 1:] = seg[0, 1:]
    for c in range(2, k + 1):
        for e in range(c, L + 1):
            vals = [best[c - 1, a] + seg[a, e] for a in range(c - 1, e)]
            arg[c, e] = int(np.argmax(vals)) + c - 1
            best[c, e] = max(vals)
    anchors, e = [], L
    for c in range(k, 1, -1):
        anchors.append(int(arg[c, e]))
        e = arg[c, e]
    return [0] + sorted(anchors), float(best[k, L] / L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--anchors", default="1,2,4,8,16")
    a = p.parse_args()
    d = OUT / a.tag
    summary = json.loads((d / "summary.json").read_text())
    sink, window = summary["attn"]["sink"], summary["attn"]["window"]
    same, best, pooled, cov, chance, inter = [], [], [], [], [], []
    for run in summary["runs"]:
        z = np.load(d / f"{run['name']}.npz")
        routes, n = z["routes"], int(z["tokens"])
        V = routes.shape[-1]
        s, b, u, x = pair_metrics(masks(routes, n), V)
        same.append(s); best.append(b); pooled.append(u); inter.append(x)
        cov.append(coverage(routes, z["number"]))
        chance.append(V / (n - sink - window))
    same, best, pooled, cov, inter = (np.mean(x, 0) for x in (same, best, pooled, cov, inter))
    L = same.shape[0]
    adj = lambda M: float(np.mean([M[i, i + 1] for i in range(L - 1)]))
    report = {"tag": a.tag, "prompts": len(summary["runs"]), "exact": sum(r["exact"] for r in summary["runs"]),
              "chance_same_head": float(np.mean(chance)),
              "adjacent_mean": {"same_head": adj(same), "best_head": adj(best), "pooled": adj(pooled)},
              "adjacent_same_head": [float(same[i, i + 1]) for i in range(L - 1)],
              "needle_coverage_per_layer": cov.tolist(), "anchors": {}}
    for k in map(int, a.anchors.split(",")):
        anchors, score = choose_anchors(same, k)
        owner = [max(x for x in anchors if x <= j) for j in range(L)]
        report["anchors"][k] = {"layers": anchors, "mean_same_head_overlap": float(score),
                                "mean_needle_coverage_under_reuse": float(np.mean([cov[o] for o in owner]))}
    report["anchors"][L] = {"layers": list(range(L)), "mean_same_head_overlap": 1.0,
                            "mean_needle_coverage_under_reuse": float(cov.mean())}
    (d / "analysis.json").write_text(json.dumps(report, indent=1))
    # inter[a, b, h, g] = mean |R_a,h ∩ R_b,g| / V, used by model.head_map for head remapping
    np.savez_compressed(d / "matrices.npz", same_head=same, best_head=best, pooled=pooled, coverage=cov,
                        inter=inter.astype(np.float32))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, M, title in zip(axes, (same, best, pooled), ("same head", "best head (oracle)", "pooled over heads")):
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"{title}: adjacent {adj(M):.2f}")
        ax.set_xlabel("reusing layer j"); ax.set_ylabel("anchor layer i")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(f"v3 routed-column overlap across layers ({a.tag}, chance {np.mean(chance):.3f})")
    fig.savefig(d / "overlap.png", dpi=120, bbox_inches="tight")
    print(json.dumps({k: v for k, v in report.items() if k not in ("needle_coverage_per_layer", "adjacent_same_head")}, indent=1))


if __name__ == "__main__":
    main()
