#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_128k_more_%j.out
#SBATCH --exclude=fc11013

# 128K runs that timed out or weren't submitted yet
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
echo "RULER 128K additional runs"
echo "$(date -Iseconds)"
echo "========================================"

# Exp 10 at 128K with k=512 (timed out in previous job)
run_ruler 10 131072 '{"top_k": 512, "low_rank_dim": 128, "use_triton": false}'

# Exp 10 at 128K with k=1024 (larger budget)
run_ruler 10 131072 '{"top_k": 1024, "low_rank_dim": 128, "use_triton": false}'

echo "=== 128K MORE SWEEP COMPLETE: $(date -Iseconds) ==="
