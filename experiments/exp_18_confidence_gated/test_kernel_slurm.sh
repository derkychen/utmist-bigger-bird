#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/exp18_kernel_test_%j.out

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

python -m experiments.exp_18_confidence_gated.test_fast_kernel \
  --seqs 2048,4096,8192,16384,32768,65536 \
  --window 256 \
  --top-k 512 \
  --warmup 3 \
  --iters 10
