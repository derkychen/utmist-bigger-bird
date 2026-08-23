#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_exp17_128k_%j.out
#SBATCH --exclude=fc11013

# Exp 17 at 128K with large block budgets
# At 128K with block_size=128, there are 1024 blocks
# topk_blocks=512 = 50% coverage (worked at 64K)
# topk_blocks=256 = 25% coverage
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
echo "Exp 17 at 128K with large block budgets"
echo "$(date -Iseconds)"
echo "========================================"

# 128K: 1024 blocks. 50% coverage = 512 blocks. Use 10 examples (128K is slow).
run_ruler 17 131072 '{"block_size": 128, "topk_blocks": 512, "fine_k": 512, "window_size": 256, "use_triton": false}' 10

# 128K: 25% coverage = 256 blocks
run_ruler 17 131072 '{"block_size": 128, "topk_blocks": 256, "fine_k": 512, "window_size": 256, "use_triton": false}' 10

echo "=== EXP 17 128K SWEEP COMPLETE: $(date -Iseconds) ==="
