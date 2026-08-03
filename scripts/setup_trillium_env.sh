#!/usr/bin/env bash
# One-time setup on a Trillium *login* node (has internet).
# Do NOT run this inside an sbatch job — compute nodes have no network.
#
#   ssh …@trillium-gpu.scinet.utoronto.ca   # GPU login preferred
#   cd /scratch/rneilalu/utmist-bigger-bird
#   bash scripts/setup_trillium_env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Needed if you logged in with: bash --noprofile --norc
if ! type module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile 2>/dev/null || true
fi
if ! type module >/dev/null 2>&1; then
  echo "ERROR: 'module' command not found. Start a normal login shell first:" >&2
  echo "  ssh -i ~/.ssh/id_rsa rneilalu@trillium.scinet.utoronto.ca" >&2
  exit 1
fi

# IMPORTANT: load arrow *before* activating the venv (Alliance pyarrow policy)
module load StdEnv/2023
module load gcc
module load arrow
module load cuda/12.6
module load python/3.11.5

if [[ ! -f .venv/bin/activate ]]; then
  echo "=== (re)creating .venv ==="
  rm -rf .venv
  python -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
# Prefer Alliance wheels; no pyarrow pin — use the arrow module
REQ="${ROOT}/requirements-trillium.txt"
if [[ -f "$REQ" ]]; then
  python -m pip install -r "$REQ"
else
  python -m pip install \
    accelerate datasets transformers torch numpy \
    scikit-learn scipy matplotlib pandas tqdm \
    sentencepiece tiktoken safetensors tokenizers \
    protobuf rich typer filelock fsspec huggingface_hub \
    pyyaml requests packaging psutil sympy regex
fi

echo "=== sanity check ==="
python - <<'PY'
import torch
import pyarrow
print("torch", torch.__version__)
print("pyarrow", pyarrow.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo
echo "Done. Submit with:"
echo "  sbatch --time=12:00:00 --export=ALL,TRACK=lra scripts/run_evals_trillium.sbatch"
