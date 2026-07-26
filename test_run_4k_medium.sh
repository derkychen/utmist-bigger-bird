#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=65536M
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/run4k_medium_%j.out
#SBATCH --error=/scratch/%u/run4k_medium_%j.err

set -euo pipefail

echo "=== Job $SLURM_JOB_ID on $SLURM_NODELIST ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/$USER/utmist-bigger-bird

EXPS="0 1 8 13 14 15"
SEQ=4096

# ── Part 1: IMDb experiments at 4096 seq len (medium data) ──
# exp_15 (BiggerBird) OOMs at 4096 even with bs=1 due to gather attention
# Skip it for IMDb 4096 — it works at shorter seq lens
for EXP in 0 1 8 13 14; do
    echo ""
    echo "============================================"
    echo "=== IMDb exp_$EXP (medium-long, 4096) ==="
    echo "============================================"
    python run_llama_tests.py --exp $EXP --size medium-long 2>&1 || echo "FAILED: imdb exp_$EXP"
done

echo ""
echo "=== All IMDb 4096 medium experiments completed ==="

# ── Part 2: LRA evals on R1-Llama-8B at 4096 seq len ──
# Use batch=1 accum=16 to fit 4096 seq len in 40GB MIG
for EXP in 0 1 8 13 14 15; do
    echo ""
    echo "============================================"
    echo "=== LRA-Llama listops, exp $EXP, seq $SEQ ==="
    echo "============================================"
    python -m eval.lra_llama.run --task listops --exp $EXP --seq $SEQ --size lra-smoke --batch 1 --accum 16 2>&1 || echo "FAILED: lra-llama listops exp_$EXP"
done

for EXP in 0 1 8 13 14 15; do
    echo ""
    echo "============================================"
    echo "=== LRA-Llama text, exp $EXP, seq $SEQ ==="
    echo "============================================"
    python -m eval.lra_llama.run --task text --exp $EXP --seq $SEQ --size lra-smoke --batch 1 --accum 16 2>&1 || echo "FAILED: lra-llama text exp_$EXP"
done

echo ""
echo "=== All LRA-Llama 4096 experiments completed ==="

# ── Part 3: RULER evals on R1-Llama-8B at 4096 seq len ──
for EXP in 0 1 8 13 14 15; do
    echo ""
    echo "============================================"
    echo "=== RULER-Llama niah, exp $EXP, seq $SEQ, depth 0.5 ==="
    echo "============================================"
    python -m eval.ruler_llama.run --task niah --exp $EXP --seq $SEQ --depth 0.5 --size ruler-smoke --batch 1 --accum 16 2>&1 || echo "FAILED: ruler-llama niah exp_$EXP"
done

for EXP in 0 1 8 13 14 15; do
    echo ""
    echo "============================================"
    echo "=== RULER-Llama mq_niah, exp $EXP, seq $SEQ, depth 0.5 ==="
    echo "============================================"
    python -m eval.ruler_llama.run --task mq_niah --exp $EXP --seq $SEQ --depth 0.5 --size ruler-smoke --batch 1 --accum 16 2>&1 || echo "FAILED: ruler-llama mq_niah exp_$EXP"
done

echo ""
echo "=== All RULER-Llama 4096 experiments completed ==="

# ── Rebuild dashboard ──
echo ""
echo "=== Rebuilding dashboard ==="
python3 scripts/build_dashboard.py 2>&1 || echo "WARNING: dashboard build failed"

echo ""
echo "=== ALL 4K MEDIUM RUNS COMPLETE ==="
