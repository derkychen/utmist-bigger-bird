#!/bin/bash
#SBATCH --account=def-guerzhoy
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --output=/scratch/thomas7/topk_128k_resubmit_%j.out

source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/thomas7/utmist-bigger-bird

echo "=== TopK 128K resubmit (15 examples) ==="
echo "=== RULER niah | exp 1 | seq 131072 | resubmit | $(date -Iseconds) ==="

python -m eval.ruler_llama.run_generative \
  --task niah \
  --exp 1 \
  --seq 131072 \
  --depth 0.5 \
  --eval-samples 15 \
  --max-examples 15 \
  --attn-kwargs '{"top_k": 512, "low_rank_dim": 128, "use_triton": false}'

echo "=== Done | $(date -Iseconds) ==="
