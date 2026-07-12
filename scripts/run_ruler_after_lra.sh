#!/usr/bin/env bash
# Wait for the in-flight LRA all-13 sweep to finish, then run the RULER eval track.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="${ROOT}/benchmarks/ruler_after_lra.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Iseconds) Waiting for LRA sweep to release GPU ==="
while pgrep -f "run_lra_experiment.py|eval\.lra\.(run|sweep)" >/dev/null 2>&1; do
  RUN=$(pgrep -af "run_lra_experiment.py|eval\.lra\.(run|sweep)" | grep -v grep | head -1 || true)
  echo "  still running: ${RUN:-unknown}"
  sleep 120
done
echo "=== $(date -Iseconds) LRA idle — starting RULER sweep ==="

# Phase 1: smoke test (validates pipeline on GPU before the big sweep)
.venv/bin/python -m eval.ruler.run --task niah --exp 0 --seq 1024 --depth 0.5 --size ruler-smoke

# Phase 2: full report sweep — all 13 exps × niah + mq_niah × seq × depth
.venv/bin/python -m eval.ruler.sweep \
  --tasks niah,mq_niah \
  --exps 0,1,2,3,4,5,6,7,8,9,10,11,12 \
  --seqs 2048,4096,8192 \
  --depths 0.1,0.5,0.9 \
  --size ruler-report \
  --skip-existing

cp benchmarks/ruler_sweep_results.json benchmarks/ruler_report_results.json
.venv/bin/python viz/ruler_viz.py
python3 scripts/build_dashboard.py

echo "=== $(date -Iseconds) RULER TRACK COMPLETE ==="
