#!/usr/bin/env bash
# Multi-hop NIAH evaluation on Modal H100 80GB.
# Runs dense + bb_v3 + best graph arms on 2-hop and 3-hop benchmarks.
#
# Usage:
#   bash experiments/graph_sparse/eval/run_multihop_modal.sh [TAG] [BYTES] [N]
set -euo pipefail
cd "$(dirname "$0")/../../.."

TAG=${1:-graph_multihop}
BYTES=${2:-512000}
N=${3:-10}

mkdir -p modal_rd/logs

for HOPS in 2 3; do
  echo "Launching multihop_${HOPS}hop..."
  modal run --detach modal_rd/app.py --module experiments.graph_sparse.eval.run_eval \
    --args "--arms dense bb_v3 graph_static_b8 graph_walk_random_l5 --task multihop_${HOPS} --bytes $BYTES --n $N --tag ${TAG}" \
    > "modal_rd/logs/graph_${TAG}_mh${HOPS}.log" 2>&1 &
  sleep 3
done
wait
echo "Launched; follow with: tail -f modal_rd/logs/graph_${TAG}_mh*.log"
