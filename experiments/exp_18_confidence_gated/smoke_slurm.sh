#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --account=def-guerzhoy

#SBATCH --output=/scratch/%u/exp18_smoke_%j.out

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

python -m eval.ruler_llama.run_generative \
  --task niah --exp 18 --seq 4096 --depth 0.5 \
  --eval-samples 8 --max-examples 10

python -m eval.ruler_llama.run_generative \
  --task niah --exp 18 --seq 8192 --depth 0.5 \
  --eval-samples 8 --max-examples 10
