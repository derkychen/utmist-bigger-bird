#!/usr/bin/env bash
# Exp 20 on one A100 40GB: pilot (all LongCodeQA + 100 each of Code.Debug / LongBench v2), then the full sets.
# Resumable: rerunning skips (example, arm) pairs already in $SCRATCH/results/exp20/$TAG/*.jsonl.
#   SCRATCH=~/scratch bash experiments/exp_20_sota_long_context/run_a100.sh [TAG]
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=${1:-a100_120k}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SCRATCH/triton-cache}
RUN="python -m experiments.exp_20_sota_long_context.run --max-tokens 120000 --tag $TAG"
$RUN --tasks lcqa32k,lcqa64k,lcqa128k
$RUN --tasks codedebug,lbv2 --limit 100
python -m experiments.exp_20_sota_long_context.report --tag "$TAG" > /dev/null
$RUN --tasks codedebug,lbv2
python -m experiments.exp_20_sota_long_context.report --tag "$TAG" > /dev/null
