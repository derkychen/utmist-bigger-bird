#!/bin/bash
# Trillium GPU sbatch: two decisive checks before any further sweeping.
#
#   sbatch scripts/run_gate_trillium.sbatch
#
# 1. NoLiMa gate. Every NoLiMa dense baseline so far sits at chance, so the whole
#    track is uninterpretable. The prompt now carries an explicit
#    "Options: 0=<name> 1=<name> ..." legend; this run decides whether that made
#    the task learnable. Chance is 0.10. Anything near it means the recipe is at
#    fault, not the data, and no further NoLiMa runs are worth submitting.
#
# 2. Seed study. Every cell in the report is a single seed, and repeated configs
#    disagree by a median of 8.0 accuracy points — wider than every mechanism's
#    reported win. Three seeds on the current leader (exp 7, layer-adaptive) and
#    the dense baseline in the same cell say whether that lead is real.
#
# Both stages are short: individual runs are 4-9 minutes at these presets.
#
#SBATCH --job-name=biggerbird-gate
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=biggerbird_gate_%j.out
#SBATCH --error=biggerbird_gate_%j.err
#SBATCH --mail-type=FAIL
## Uncomment and set if your default account is wrong:
##SBATCH --account=def-YOURACCOUNT

set -euo pipefail

module load StdEnv/2023
module load gcc
module load arrow
module load cuda/12.6
module load python/3.11.5

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
SEEDS="${SEEDS:-42 43 44}"
STAGE="${STAGE:-all}"

cd "$REPO_DIR"
export COMPUTE_RESOURCE="${COMPUTE_RESOURCE:-trillium-h100}"
export CLUSTER_NAME="${CLUSTER_NAME:-trillium}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "ERROR: missing venv at ${VENV_DIR}" >&2
  echo "On the GPU login node run once: bash scripts/setup_trillium_env.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${SCRATCH}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH}/.cache/torch}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME"

echo "=== $(date -Iseconds) BiggerBird gate ==="
echo "host=$(hostname)  job=${SLURM_JOB_ID:-local}  stage=${STAGE}  seeds=${SEEDS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_nolima_gate() {
  echo ""
  echo "=== $(date -Iseconds) STAGE 1: NoLiMa gate (chance = 0.10) ==="
  # No --skip-existing: the NoLiMa results already on disk predate the prompt fix
  # and must not be allowed to satisfy this check.
  python -m eval.nolima.run \
    --task onehop --exp 0 --seq 2048 --size nolima-report --grad-checkpoint
}

run_seed_study() {
  echo ""
  echo "=== $(date -Iseconds) STAGE 2: seed study on lra/listops/1024 ==="
  for s in $SEEDS; do
    for e in 0 7; do
      echo "--- $(date -Iseconds) exp ${e}, seed ${s} ---"
      python -m eval.lra.run \
        --task listops --exp "$e" --seq 1024 --size lra-full \
        --seed "$s" --grad-checkpoint
    done
  done
}

case "$STAGE" in
  all)    run_nolima_gate; run_seed_study ;;
  nolima) run_nolima_gate ;;
  seeds)  run_seed_study ;;
  *) echo "Unknown STAGE=${STAGE} (use all|nolima|seeds)" >&2; exit 1 ;;
esac

echo ""
echo "=== $(date -Iseconds) VERDICT ==="
python scripts/summarize_gate.py || true

echo ""
echo "=== $(date -Iseconds) GATE COMPLETE ==="
echo "Commit the new results:  git add benchmarks/ && git commit -m 'NoLiMa gate + listops seed study'"
