#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --account=def-guerzhoy

#SBATCH --output=/scratch/%u/exp18_profile_%j.out

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

python -m experiments.exp_18_confidence_gated.profile \
  --seqs 2048,4096,8192,16384 \
  --top-k 512 \
  --low-rank-dim 64 \
  --window-size 256 \
  --gate-threshold 0.5 \
  --warmup 1 \
  --iters 2
