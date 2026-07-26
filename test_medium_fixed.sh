#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=/scratch/%u/medium_fixed_%j.out
#SBATCH --error=/scratch/%u/medium_fixed_%j.err

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
cd /home/$USER/utmist-bigger-bird

for EXP in 0 1 8 13 14 15; do
    echo ""
    echo "============================================"
    echo "=== Testing exp_$EXP (medium) with all fixes ==="
    echo "============================================"
    python run_llama_tests.py --exp $EXP --size medium 2>&1 || echo "FAILED: exp_$EXP"
done
echo ""
echo "=== All medium-fixed experiments completed ==="
