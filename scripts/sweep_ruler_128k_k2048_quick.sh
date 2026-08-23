#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --output=/scratch/%u/sweep_ruler_128k_k2048_quick_%j.out
#SBATCH --exclude=fc11013

# Exp 10 k=2048 at 128K with only 5 examples (15 timed out in 12h)
set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

echo "========================================"
echo "Exp 10 k=2048 at 128K (5 examples quick)"
echo "$(date -Iseconds)"
echo "========================================"

echo "=== RULER niah | exp 10 | seq 131072 | $(date -Iseconds) ==="
python -m eval.ruler_llama.run_generative \
    --task niah --exp 10 --seq 131072 --depth 0.5 \
    --eval-samples 128 --max-examples 5 \
    --attn-kwargs '{"top_k": 2048, "low_rank_dim": 128, "use_triton": false}' 2>&1 | tail -8

echo "=== COMPLETE: $(date -Iseconds) ==="
