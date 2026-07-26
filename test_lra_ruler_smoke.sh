#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=/scratch/%u/lra_ruler_smoke_%j.out
#SBATCH --error=/scratch/%u/lra_ruler_smoke_%j.err

set -euo pipefail

echo "=== Job $SLURM_JOB_ID on $SLURM_NODELIST ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false

cd /home/$USER/utmist-bigger-bird

# ── LRA smoke tests (listops + text) ──
# Uses from-scratch BART encoder (small, fast)
echo ""
echo "============================================"
echo "=== LRA smoke: listops, exp 0, seq 512 ==="
echo "============================================"
python -m eval.lra.run --task listops --exp 0 --seq 512 --size lra-smoke 2>&1 || echo "FAILED: lra listops exp0"

echo ""
echo "============================================"
echo "=== LRA smoke: listops, exp 1, seq 512 ==="
echo "============================================"
python -m eval.lra.run --task listops --exp 1 --seq 512 --size lra-smoke 2>&1 || echo "FAILED: lra listops exp1"

echo ""
echo "============================================"
echo "=== LRA smoke: text, exp 0, seq 1024 ==="
echo "============================================"
python -m eval.lra.run --task text --exp 0 --seq 1024 --size lra-smoke 2>&1 || echo "FAILED: lra text exp0"

echo ""
echo "============================================"
echo "=== LRA smoke: text, exp 1, seq 1024 ==="
echo "============================================"
python -m eval.lra.run --task text --exp 1 --seq 1024 --size lra-smoke 2>&1 || echo "FAILED: lra text exp1"

# ── RULER smoke tests (niah + mq_niah) ──
echo ""
echo "============================================"
echo "=== RULER smoke: niah, exp 0, seq 1024, depth 0.5 ==="
echo "============================================"
python -m eval.ruler.run --task niah --exp 0 --seq 1024 --depth 0.5 --size ruler-smoke 2>&1 || echo "FAILED: ruler niah exp0"

echo ""
echo "============================================"
echo "=== RULER smoke: niah, exp 1, seq 1024, depth 0.5 ==="
echo "============================================"
python -m eval.ruler.run --task niah --exp 1 --seq 1024 --depth 0.5 --size ruler-smoke 2>&1 || echo "FAILED: ruler niah exp1"

echo ""
echo "============================================"
echo "=== RULER smoke: mq_niah, exp 0, seq 1024, depth 0.5 ==="
echo "============================================"
python -m eval.ruler.run --task mq_niah --exp 0 --seq 1024 --depth 0.5 --size ruler-smoke 2>&1 || echo "FAILED: ruler mq_niah exp0"

echo ""
echo "============================================"
echo "=== RULER smoke: mq_niah, exp 1, seq 1024, depth 0.5 ==="
echo "============================================"
python -m eval.ruler.run --task mq_niah --exp 1 --seq 1024 --depth 0.5 --size ruler-smoke 2>&1 || echo "FAILED: ruler mq_niah exp1"

echo ""
echo "=== All LRA+RULER smoke tests completed ==="
