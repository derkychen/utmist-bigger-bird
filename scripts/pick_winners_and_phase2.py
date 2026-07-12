#!/usr/bin/env python3
"""Pick exp 5-10 winners from sweep JSON and run Phase 2 @ 2048 for top 1-2."""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(ROOT, "benchmarks", "long_context_sweep_results.json")
PY = os.path.join(ROOT, "venv", "bin", "python")
EXP1_2048_TIME = 105.0  # reference from overnight sweep
EXP1_2048_F1 = 0.80
BASELINE_F1 = 0.887


def main():
    if not os.path.isfile(SWEEP):
        print(f"Missing {SWEEP}")
        sys.exit(1)
    with open(SWEEP) as f:
        data = json.load(f)
    results = [r for r in data["results"] if r.get("exp") in (5, 6, 7, 10) and r.get("seq") == 2048 and not r.get("oom")]

    scored = []
    for r in results:
        f1 = r.get("f1") or 0
        t = r.get("train_time_s") or 1e9
        scored.append((r["exp"], r["exp_name"], f1, t))

    print("Phase 1 @ 2048:")
    for row in scored:
        print(f"  exp {row[0]} ({row[1]}): F1={row[2]:.3f} time={row[3]:.1f}s")

    winners = []
    for exp, name, f1, t in scored:
        if f1 >= BASELINE_F1 - 0.02 or (f1 >= 0.85 and t <= EXP1_2048_TIME * 0.85):
            winners.append(exp)

    if not winners and scored:
        winners = [max(scored, key=lambda x: x[2])[0]]
    winners = sorted(set(winners))[:2]
    print(f"\nPhase 2 winners: {winners}")

    out = os.path.join(ROOT, "benchmarks", "exp_5_10_winners.json")
    with open(out, "w") as f:
        json.dump({"winners": winners, "scored_2048": scored}, f, indent=2)

    for exp in winners:
        print(f"\n========== Phase 2 exp {exp} ==========")
        subprocess.run(
            [
                PY, "run_experiment.py", "--exp", str(exp), "--size", "big",
                "--fixed-length", "--seq", "2048", "--grad-checkpoint",
                "--train-samples", "6000", "--eval-samples", "1000", "--epochs", "3",
            ],
            cwd=ROOT,
            check=False,
        )


if __name__ == "__main__":
    main()
