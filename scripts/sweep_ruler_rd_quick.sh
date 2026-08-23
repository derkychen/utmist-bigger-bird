#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_rd_quick_%j.out
#SBATCH --exclude=fc11013

# Quick R&D sweep: 4K only for new experiments to get fast signal

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
echo "RULER R&D QUICK sweep (4K only)"
echo "$(date -Iseconds)"
echo "========================================"

# Exp 16 (Free NSA) at 4K
run_ruler 16 4096 '{"block_size": 64, "topk_blocks": 16, "window_size": 256, "use_triton": false}'

# Exp 17 (Coarse-to-fine) at 4K
run_ruler 17 4096 '{"block_size": 128, "topk_blocks": 32, "fine_k": 512, "window_size": 256, "use_triton": false}'

# Exp 15 (BiggerBird proper) at 4K
run_ruler 15 4096 '{"fragment_size": 128, "max_k": 1024, "min_k": 128, "globals_per_head": 32, "teleports_per_head": 8, "use_teleports": false, "use_triton": false}'

# Exp 4 with larger budget at 4K
run_ruler 4 4096 '{"block_size": 128, "num_blocks": 8, "use_triton": false}'

# Exp 10 with larger budget at 4K
run_ruler 10 4096 '{"top_k": 512, "low_rank_dim": 128, "use_triton": false}'

echo "=== QUICK SWEEP COMPLETE: $(date -Iseconds) ==="
