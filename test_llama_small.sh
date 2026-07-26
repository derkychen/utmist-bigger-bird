#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=/scratch/%u/llama_small_%j.out
#SBATCH --error=/scratch/%u/llama_small_%j.err

set -euo pipefail

echo "=== Job $SLURM_JOB_ID on $SLURM_NODELIST ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate

export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false

cd /home/$USER/utmist-bigger-bird

EXPERIMENTS="0 1 8 13 14 15"

for EXP in $EXPERIMENTS; do
    echo ""
    echo "============================================"
    echo "=== Testing exp_$EXP (small) ==="
    echo "============================================"
    python run_llama_tests.py --exp $EXP --size small 2>&1 || echo "FAILED: exp_$EXP"
done

echo ""
echo "=== All small experiments completed ==="
