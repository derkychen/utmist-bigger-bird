#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=/scratch/%u/exp1_fix_%j.out
#SBATCH --error=/scratch/%u/exp1_fix_%j.err

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
cd /home/$USER/utmist-bigger-bird

echo "=== Testing exp_1 (small) with top_k=128, low_rank_dim=64, ratio=2 ==="
python run_llama_tests.py --exp 1 --size small 2>&1
echo "=== DONE ==="
