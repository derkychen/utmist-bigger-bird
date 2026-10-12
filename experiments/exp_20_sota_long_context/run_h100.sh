#!/usr/bin/env bash
# Exp 20 full run on Modal H100 80GB: every task, every item, no truncation up to 128K (--max-tokens 130000).
# Three containers in parallel (one per task group), all writing to /vol/results/exp20/$TAG/<task>.jsonl.
# Each run resumes: if a container hits the 6 h Modal timeout, rerun this script with the same TAG.
#   bash experiments/exp_20_sota_long_context/run_h100.sh [TAG]
# Then pull and report locally:
#   modal volume get bb-vol results/exp20/$TAG ~/scratch/results/exp20/
#   SCRATCH=~/scratch python -m experiments.exp_20_sota_long_context.report --tag $TAG \
#       --out experiments/exp_20_sota_long_context/RESULTS.md
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=${1:-h100_130k}
mkdir -p modal_rd/logs   # excluded from the Modal image, so logging here does not abort the build
for TASKS in lcqa32k,lcqa64k,lcqa128k codedebug lbv2; do
  modal run --detach modal_rd/app.py --module experiments.exp_20_sota_long_context.run \
    --args "--tasks $TASKS --max-tokens 130000 --tag $TAG" > "modal_rd/logs/exp20_${TAG}_${TASKS//,/_}.log" 2>&1 &
  sleep 3
done
wait
echo "launched; follow with: tail -f modal_rd/logs/exp20_${TAG}_*.log"
