#!/bin/bash
#SBATCH --account=def-guerzhoy
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --output=/scratch/thomas7/sweep_exp11_%j.out

source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/thomas7/utmist-bigger-bird

for SEQ in 4096 8192 16384 32768 65536; do
  echo "=== RULER niah | exp 11 | seq $SEQ | sweep11 | $(date -Iseconds) ==="
  python -m eval.ruler_llama.run_generative \
    --task niah \
    --exp 11 \
    --seq $SEQ \
    --depth 0.5 \
    --eval-samples 128 \
    --max-examples 30
  echo "---"
done

echo "=== Done | $(date -Iseconds) ==="
