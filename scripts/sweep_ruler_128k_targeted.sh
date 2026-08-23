#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_128k_targeted_%j.out
#SBATCH --exclude=fc11013

# Targeted 128K experiments:
# 1. exp 17 with low-rank coarse fix + fine_k=512 (test if coarse scoring was the issue)
# 2. exp 17 with fine_k=2048 (test if fine budget is the bottleneck)
# 3. exp 1 with k=1024 (fill gap between k=512 and k=2048)
set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

run_ruler() {
    local exp=$1 seq=$2 kwargs_json=$3 max_ex=$4
    echo "=== RULER niah | exp $exp | seq $seq | $(date -Iseconds) ==="
    local cmd="python -m eval.ruler_llama.run_generative \
        --task niah --exp $exp --seq $seq --depth 0.5 \
        --eval-samples 128 --max-examples $max_ex"
    if [ -n "$kwargs_json" ]; then
        cmd="$cmd --attn-kwargs '$kwargs_json'"
        echo "  kwargs: $kwargs_json  max_examples: $max_ex"
    fi
    if eval "$cmd" 2>&1 | tail -8; then
        echo ""
    else
        echo "  FAILED: exp $exp seq $seq"
        echo ""
    fi
}

echo "========================================"
echo "Targeted 128K experiments"
echo "$(date -Iseconds)"
echo "========================================"

# 1. exp 17 with low-rank coarse fix, topk_blocks=512, fine_k=512 (10 examples)
run_ruler 17 131072 '{"block_size": 128, "topk_blocks": 512, "fine_k": 512, "window_size": 256, "use_triton": false}' 10

# 2. exp 17 with fine_k=2048, topk_blocks=512 (5 examples — fine_k=2048 uses more memory)
run_ruler 17 131072 '{"block_size": 128, "topk_blocks": 512, "fine_k": 2048, "window_size": 256, "use_triton": false}' 5

# 3. exp 1 with k=1024 at 128K (10 examples — fill the gap)
run_ruler 1 131072 '{"top_k": 1024, "low_rank_dim": 128, "use_triton": false}' 10

echo "=== TARGETED SWEEP COMPLETE: $(date -Iseconds) ==="
