#!/bin/bash
# Trillium GPU sbatch: LRA / RULER / NoLiMa eval sweeps
#
# Submit from the GPU login node, from a writable scratch copy of the repo:
#   ssh …@trillium-gpu.scinet.utoronto.ca
#   cd $SCRATCH/utmist-bigger-bird   # or wherever the repo lives on scratch
#   sbatch scripts/run_evals_trillium.sbatch
#
# Optional overrides (export before sbatch, or: sbatch --export=ALL,TRACK=ruler …):
#   TRACK=all|lra|ruler|nolima   (default: all = lra then ruler then nolima)
#   REPO_DIR=/path/to/repo       (default: $SLURM_SUBMIT_DIR)
#   VENV_DIR=/path/to/.venv      (default: $REPO_DIR/.venv)
#   SIZE_LRA=lra-report
#   SIZE_RULER=ruler-report
#   SIZE_NOLIMA=nolima-report
#   EXPS=0,1,2,...,14
#
#SBATCH --job-name=biggerbird-evals
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --output=biggerbird_evals_%j.out
#SBATCH --error=biggerbird_evals_%j.err
#SBATCH --mail-type=FAIL
## Uncomment and set if your default account is wrong:
##SBATCH --account=def-YOURACCOUNT

set -euo pipefail

# --- modules (load inside the job; do not rely on .bashrc) ---
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

# --- paths (compute nodes: $HOME read-only; write under $SCRATCH) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
TRACK="${TRACK:-all}"
SIZE_LRA="${SIZE_LRA:-lra-report}"
SIZE_RULER="${SIZE_RULER:-ruler-report}"
SIZE_NOLIMA="${SIZE_NOLIMA:-nolima-report}"
EXPS="${EXPS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}"

cd "$REPO_DIR"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export PYTHONUNBUFFERED=1
# Keep HF / torch caches on scratch (writable; pre-download on a login node)
export HF_HOME="${HF_HOME:-${SCRATCH}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH}/.cache/torch}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME"

echo "=== $(date -Iseconds) BiggerBird evals on Trillium ==="
echo "host=$(hostname)  job=${SLURM_JOB_ID:-local}  track=${TRACK}"
echo "repo=${REPO_DIR}"
echo "python=$(command -v python)  $(python -V 2>&1)"
nvidia-smi || true

run_lra() {
  echo "=== $(date -Iseconds) LRA sweep (${SIZE_LRA}) ==="
  python -m eval.lra.sweep \
    --tasks listops,text,image,pathfinder \
    --exps "$EXPS" \
    --size "$SIZE_LRA" \
    --grad-checkpoint \
    --skip-existing
  python viz/lra_viz.py || true
}

run_ruler() {
  echo "=== $(date -Iseconds) RULER smoke ==="
  python -m eval.ruler.run \
    --task niah_single_1 --exp 0 --seq 1024 --depth 0.5 --size ruler-smoke

  echo "=== $(date -Iseconds) RULER sweep (${SIZE_RULER}) ==="
  python -m eval.ruler.sweep \
    --exps "$EXPS" \
    --seqs 2048,4096,8192 \
    --depths 0.1,0.5,0.9 \
    --size "$SIZE_RULER" \
    --grad-checkpoint \
    --skip-existing

  cp -f benchmarks/ruler_sweep_results.json benchmarks/ruler_report_results.json
  python viz/ruler_viz.py || true
}

run_nolima() {
  echo "=== $(date -Iseconds) NoLiMa sweep (${SIZE_NOLIMA}) ==="
  python -m eval.nolima.sweep \
    --tasks onehop,twohop,hard \
    --exps "$EXPS" \
    --size "$SIZE_NOLIMA" \
    --grad-checkpoint \
    --skip-existing
}

case "$TRACK" in
  all)
    run_lra
    run_ruler
    run_nolima
    ;;
  lra)    run_lra ;;
  ruler)  run_ruler ;;
  nolima) run_nolima ;;
  *)
    echo "Unknown TRACK=${TRACK} (use all|lra|ruler|nolima)" >&2
    exit 1
    ;;
esac

python scripts/build_dashboard.py || true
echo "=== $(date -Iseconds) EVALS COMPLETE (track=${TRACK}) ==="
