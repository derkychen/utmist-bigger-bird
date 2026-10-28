#!/bin/bash
# Exp 21 vs exp 19 vs dense on RULER NIAH at ~123K actual tokens (BF16, generative, exact scoring).
#
#   cd $SCRATCH/bigger-bird-r1
#   sbatch experiments/exp_21_bigger_bird_dense_early/run_128k_trillium.sbatch [EXPS] [TASKS] [MAX_EXAMPLES]
#
#   sbatch experiments/exp_21_bigger_bird_dense_early/run_128k_trillium.sbatch                 # 21,19,0 x multikey_1,single_1 x 30
#   sbatch --partition=debug --time=1:00:00 experiments/exp_21_bigger_bird_dense_early/run_128k_trillium.sbatch 21 niah_multikey_1 2
#
# RULER --seq counts characters, and characters per token differ by task, so each task gets its own --seq
# (measured with the checkpoint tokenizer). The prompt must stay under the 131072-position limit;
# run_generative raises instead of truncating the question away.
# Trillium's sbatch forces --export=NONE, so settings are positional arguments.
#SBATCH --job-name=exp21-128k
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=exp21_128k_%j.out

set -euo pipefail
EXPS="${1:-21,19,0}"
TASKS="${2:-niah_multikey_1,niah_single_1}"
MAX_EXAMPLES="${3:-30}"
declare -A SEQ=(
  [niah_multikey_1]=614400  # 4.94 chars/token -> ~124K tokens (extrapolated, not yet tokenized)
  [niah_single_1]=491520    # 4.01 chars/token -> 122.5K-122.6K tokens (measured)
)

module load StdEnv/2023 gcc arrow cuda/12.6 python/3.11.5
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR="${SCRATCH}/.cache/triton"  # $HOME is read-only on compute nodes

nvidia-smi --query-gpu=name,memory.total --format=csv
for task in ${TASKS//,/ }; do
  seq="${SEQ[$task]:?no --seq set for $task; add it to SEQ}"
  for exp in ${EXPS//,/ }; do
    echo "=== $(date -Iseconds) exp ${exp} task ${task} seq ${seq} ==="
    python -m eval.ruler_llama.run_generative --task "$task" --exp "$exp" --seq "$seq" --depth 0.5 \
      --eval-samples 128 --max-examples "$MAX_EXAMPLES"
  done
done
