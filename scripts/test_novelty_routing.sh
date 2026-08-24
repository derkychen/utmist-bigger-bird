#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/exp18_novelty_test_%j.out

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

# Test scale-invariant hybrid routing at small scale
# Budget = ratio * seq_len (no hard-coded top_k)
# 2K @ 5% = 102 tokens, 4K @ 5% = 204, 8K @ 5% = 410
# 2K @ 10% = 205, 4K @ 10% = 410, 8K @ 10% = 819

BASE='{"top_k": 2048, "low_rank_dim": 128, "window_size": 256, "gate_threshold": 0.5, "peak_threshold": -1.0, "linear_weight": 0.5, "use_triton": false, "always_global": true, "num_route_queries": 4, "adaptive_low_rank": true'

for SEQ in 2048 4096 8192; do
  for RATIO in 0.05 0.10; do
    echo "=== Hybrid routing at ${SEQ}, ratio=${RATIO} ==="
    python -m eval.ruler_llama.run_generative \
      --task niah --exp 18 --seq $SEQ --depth 0.5 \
      --eval-samples 128 --max-examples 10 \
      --attn-kwargs "${BASE}, \"routing_mode\": \"hybrid\", \"novelty_ratio\": ${RATIO}, \"novelty_window\": 64}}" || true
  done
done
