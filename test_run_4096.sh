#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=65536M
#SBATCH --time=6:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/run4096_%j.out
#SBATCH --error=/scratch/%u/run4096_%j.err

set -euo pipefail

echo "=== Job $SLURM_JOB_ID on $SLURM_NODELIST ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/$USER/utmist-bigger-bird

# ── Part 1: IMDb experiments at 4096 seq len (small data) ──
for EXP in 0 1 8 13 14 15; do
    echo ""
    echo "============================================"
    echo "=== IMDb exp_$EXP (small-long, 4096) ==="
    echo "============================================"
    python run_llama_tests.py --exp $EXP --size small-long 2>&1 || echo "FAILED: imdb exp_$EXP"
done

echo ""
echo "=== All IMDb 4096 experiments completed ==="

# ── Part 2: LRA evals at 4096 seq len (smoke data, batch=1 for 4096) ──
# listops + text tasks, exps 0,1,8,13,14,15
# Use --batch 1 --accum 8 to avoid OOM at 4096 seq len on from-scratch BART
for EXP in 0 1 8 13 14 15; do
    echo ""
    echo "============================================"
    echo "=== LRA listops, exp $EXP, seq 4096 ==="
    echo "============================================"
    python -m eval.lra.run --task listops --exp $EXP --seq 4096 --size lra-smoke --batch 1 --accum 8 --grad-checkpoint 2>&1 || echo "FAILED: lra listops exp_$EXP"
done

for EXP in 0 1 8 13 14 15; do
    echo ""
    echo "============================================"
    echo "=== LRA text, exp $EXP, seq 4096 ==="
    echo "============================================"
    python -m eval.lra.run --task text --exp $EXP --seq 4096 --size lra-smoke --batch 1 --accum 8 --grad-checkpoint 2>&1 || echo "FAILED: lra text exp_$EXP"
done

echo ""
echo "=== All LRA 4096 experiments completed ==="

# ── Part 3: RULER evals at 4096 seq len (smoke data, batch=1) ──
# niah + mq_niah tasks, exps 0,1,8,13,14,15
for EXP in 0 1 8 13 14 15; do
    echo ""
    echo "============================================"
    echo "=== RULER niah, exp $EXP, seq 4096, depth 0.5 ==="
    echo "============================================"
    python -m eval.ruler.run --task niah --exp $EXP --seq 4096 --depth 0.5 --size ruler-smoke --batch 1 --accum 8 --grad-checkpoint 2>&1 || echo "FAILED: ruler niah exp_$EXP"
done

for EXP in 0 1 8 13 14 15; do
    echo ""
    echo "============================================"
    echo "=== RULER mq_niah, exp $EXP, seq 4096, depth 0.5 ==="
    echo "============================================"
    python -m eval.ruler.run --task mq_niah --exp $EXP --seq 4096 --depth 0.5 --size ruler-smoke --batch 1 --accum 8 --grad-checkpoint 2>&1 || echo "FAILED: ruler mq_niah exp_$EXP"
done

echo ""
echo "=== All RULER 4096 experiments completed ==="

# ── Rebuild dashboard ──
echo ""
echo "=== Rebuilding dashboard ==="
python3 scripts/build_dashboard.py 2>&1 || echo "WARNING: dashboard build failed"

echo ""
echo "=== ALL 4096 RUNS COMPLETE ==="
