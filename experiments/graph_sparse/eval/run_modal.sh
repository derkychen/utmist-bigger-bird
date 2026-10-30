#!/usr/bin/env bash
# Graph-sparse evaluation on Modal H100 80GB.
# Runs staged ablations: each stage runs in a separate container.
#
# Usage:
#   bash experiments/graph_sparse/eval/run_modal.sh [STAGE] [TAG] [BYTES] [N]
#
# Examples:
#   bash experiments/graph_sparse/eval/run_modal.sh 0 graph_s0 512000 10   # baselines
#   bash experiments/graph_sparse/eval/run_modal.sh 1 graph_s1 512000 10   # static graph
#   bash experiments/graph_sparse/eval/run_modal.sh all graph_all 512000 10 # all stages
#
# Each run resumes: if a container hits the Modal timeout, rerun with the same TAG.
# Then pull results and build report:
#   modal volume get bb-vol results/graph_eval/$TAG ~/scratch/results/graph_eval/
#   python -m experiments.graph_sparse.eval.report --tag $TAG --plots
set -euo pipefail
cd "$(dirname "$0")/../../.."

STAGE=${1:-0}
TAG=${2:-graph_s0}
BYTES=${3:-512000}
N=${4:-10}

mkdir -p modal_rd/logs

if [ "$STAGE" = "all" ]; then
  for S in 0 1 2 3 4; do
    echo "Launching stage $S..."
    modal run --detach modal_rd/app.py --module experiments.graph_sparse.eval.run_eval \
      --args "--stage $S --task niah --bytes $BYTES --n $N --tag ${TAG}" \
      > "modal_rd/logs/graph_${TAG}_s${S}.log" 2>&1 &
    sleep 3
  done
  wait
  echo "All stages launched; follow with: tail -f modal_rd/logs/graph_${TAG}_s*.log"
else
  echo "Launching stage $STAGE..."
  modal run --detach modal_rd/app.py --module experiments.graph_sparse.eval.run_eval \
    --args "--stage $STAGE --task niah --bytes $BYTES --n $N --tag ${TAG}" \
    > "modal_rd/logs/graph_${TAG}_s${STAGE}.log" 2>&1
  echo "Launched; follow with: tail -f modal_rd/logs/graph_${TAG}_s${STAGE}.log"
fi
