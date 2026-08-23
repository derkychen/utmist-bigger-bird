#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_exp17_maxpool_%j.out
#SBATCH --exclude=fc11013

# Exp 17 with max-pool block scoring fix
set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

MAX_EXAMPLES=30
EVAL_SAMPLES=128

run_ruler() {
    local exp=$1 seq=$2 kwargs_json=$3
    echo "=== RULER niah | exp $exp | seq $seq | $(date -Iseconds) ==="
    local cmd="python -m eval.ruler_llama.run_generative \
        --task niah --exp $exp --seq $seq --depth 0.5 \
        --eval-samples $EVAL_SAMPLES --max-examples $MAX_EXAMPLES"
    if [ -n "$kwargs_json" ]; then
        cmd="$cmd --attn-kwargs '$kwargs_json'"
        echo "  kwargs: $kwargs_json"
    fi
    if eval "$cmd" 2>&1 | tail -8; then
        echo ""
    else
        echo "  FAILED: exp $exp seq $seq"
        echo ""
    fi
}

echo "========================================"
echo "Exp 17 max-pool block scoring sweep"
echo "$(date -Iseconds)"
echo "========================================"

# Default config with max-pool fix
KW='{"block_size": 128, "topk_blocks": 32, "fine_k": 512, "window_size": 256, "use_triton": false}'

# Test at 16K (where mean-pool got 56.7%)
run_ruler 17 16384 "$KW"

# Continue through longer contexts
run_ruler 17 32768 "$KW"
run_ruler 17 65536 "$KW"

# Also try with larger topk_blocks for better coverage
KW2='{"block_size": 128, "topk_blocks": 64, "fine_k": 512, "window_size": 256, "use_triton": false}'
run_ruler 17 16384 "$KW2"
run_ruler 17 32768 "$KW2"

echo "=== EXP 17 MAXPOOL SWEEP COMPLETE: $(date -Iseconds) ==="
