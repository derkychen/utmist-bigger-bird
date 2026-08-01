#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for DeepSeek-R1-Distill-Llama-8B.
#
# Designed for Compute Canada `fir` H100 nodes (MIG-partitioned 40GB slices).
# Runs on a single 40GB MIG slice. To swap to a bigger distill variant or the
# 70B-FP8 flagship, edit MODEL_ID / TENSOR_PARALLEL below — nothing else.
#
# Usage (inside an salloc'd GPU node):
#   bash r1/serve.sh                 # default port 8000
#   PORT=8001 bash r1/serve.sh       # custom port
#   MODEL_ID=... TENSOR_PARALLEL=4 bash r1/serve.sh   # override model/TP
set -euo pipefail

# --- One-line config: change these to swap base model -----------------------
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
MODEL_PATH="${MODEL_PATH:-/scratch/$USER/models/DeepSeek-R1-Distill-Llama-8B}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
# ----------------------------------------------------------------------------

# Fall back to HF hub ID if the local snapshot is missing.
if [ ! -d "$MODEL_PATH" ]; then
    echo "r1/serve.sh: local snapshot $MODEL_PATH not found, using HF hub id: $MODEL_ID"
    EFFECTIVE_MODEL="$MODEL_ID"
else
    EFFECTIVE_MODEL="$MODEL_PATH"
fi

# Modules + venv (idempotent — safe to re-source on a fresh shell).
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate

export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "r1/serve.sh: serving $EFFECTIVE_MODEL  (TP=$TENSOR_PARALLEL)  on $HOST:$PORT"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$EFFECTIVE_MODEL" \
    --served-model-name r1-distill-llama-8b \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --port "$PORT" \
    --host "$HOST" \
    --trust-remote-code
