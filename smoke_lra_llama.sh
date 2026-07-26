#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=65536M
#SBATCH --time=1:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/smoke_lra_llama_%j.out
#SBATCH --error=/scratch/%u/smoke_lra_llama_%j.err

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/$USER/utmist-bigger-bird

echo "=== Smoke test: LRA-Llama listops, exp 0, seq 512 ==="
python -m eval.lra_llama.run --task listops --exp 0 --seq 512 --size lra-smoke 2>&1 || echo "FAILED"

echo ""
echo "=== Smoke test: RULER-Llama niah, exp 0, seq 512, depth 0.5 ==="
python -m eval.ruler_llama.run --task niah --exp 0 --seq 512 --depth 0.5 --size ruler-smoke 2>&1 || echo "FAILED"

echo ""
echo "=== Smoke test: LRA-Llama text, exp 1, seq 512 ==="
python -m eval.lra_llama.run --task text --exp 1 --seq 512 --size lra-smoke 2>&1 || echo "FAILED"

echo ""
echo "=== ALL SMOKE TESTS DONE ==="
