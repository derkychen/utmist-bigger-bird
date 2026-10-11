#!/bin/bash
# Trillium RULER generative eval (zero-shot) for one experiment over several lengths.
#
#   cd $SCRATCH/bigger-bird-r1
#   sbatch scripts/ruler_generative_trillium.sbatch EXP SEQS [MAX_EXAMPLES] [ATTN_KWARGS_JSON]
#
#   sbatch --partition=debug --time=1:00:00 scripts/ruler_generative_trillium.sbatch 13 4096 5
#   sbatch --time=24:00:00 scripts/ruler_generative_trillium.sbatch 13 65536,131072 30 '{"use_triton": true}'
#
# Trillium's sbatch forces --export=NONE, so settings are positional arguments.
#SBATCH --job-name=ruler-gen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=ruler_gen_%j.out

set -euo pipefail
EXP="${1:?usage: sbatch $0 EXP SEQS [MAX_EXAMPLES] [ATTN_KWARGS_JSON]}"
SEQS="${2:-4096}"
MAX_EXAMPLES="${3:-30}"
ATTN_KWARGS="${4:-}"
TASK="${TASK:-niah}"
DEPTH="${DEPTH:-0.5}"

module load StdEnv/2023 gcc arrow cuda/12.6 python/3.11.5
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR="${SCRATCH}/.cache/triton"  # $HOME is read-only on compute nodes

nvidia-smi --query-gpu=name,memory.total --format=csv
for seq in ${SEQS//,/ }; do
  echo "=== $(date -Iseconds) exp ${EXP} seq ${seq} kwargs ${ATTN_KWARGS:-default} ==="
  args=(--task "$TASK" --exp "$EXP" --seq "$seq" --depth "$DEPTH"
        --eval-samples 128 --max-examples "$MAX_EXAMPLES")
  if [[ -n "$ATTN_KWARGS" ]]; then
    args+=(--attn-kwargs "$ATTN_KWARGS")
  fi
  python -m eval.ruler_llama.run_generative "${args[@]}"
done
