#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/exp18_test_256k_%j.out

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

echo "=== Exp 18 v3 test at 256K (10 examples, ~8h) ==="
python -m eval.ruler_llama.run_generative \
  --task niah --exp 18 --seq 262144 --depth 0.5 \
  --eval-samples 128 --max-examples 10 \
  --attn-kwargs '{"top_k": 2048, "low_rank_dim": 128, "window_size": 256, "gate_threshold": 0.5, "peak_threshold": -1.0, "linear_weight": 0.5, "use_triton": false, "always_global": true, "num_route_queries": 4, "adaptive_low_rank": true}'
