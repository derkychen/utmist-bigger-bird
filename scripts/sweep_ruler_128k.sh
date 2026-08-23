#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_128k_%j.out
#SBATCH --exclude=fc11013

# 128K sweep: test all working methods at 128K with larger budgets
# - Exp 1 with k=2048 (k=1024 got 26.7%, k=512 got 13.3%)
# - Exp 13 with k=1024 (content-based routing, fixed)
# - Exp 15 with k=1024 (BiggerBird fixed)
# - Exp 10 with k=1024 (for comparison with k=512)

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
echo "RULER 128K sweep with larger budgets"
echo "$(date -Iseconds)"
echo "========================================"

# Exp 1 at 128K with k=2048 (k=1024 got 26.7%)
run_ruler 1 131072 '{"top_k": 2048, "low_rank_dim": 128, "use_triton": false}'

# Exp 13 at 128K with k=512 and k=1024
run_ruler 13 131072 '{"drop_after_layer": 3, "target_budget": 512, "chunk_size": 8192, "use_triton": false}'
run_ruler 13 131072 '{"drop_after_layer": 3, "target_budget": 1024, "chunk_size": 8192, "use_triton": false}'

# Exp 15 at 128K with k=512
run_ruler 15 131072 '{"fragment_size": 128, "max_k": 512, "min_k": 128, "globals_per_head": 8, "teleports_per_head": 4, "use_teleports": false, "use_triton": false}'

# Exp 15 at 128K with k=1024
run_ruler 15 131072 '{"fragment_size": 128, "max_k": 1024, "min_k": 128, "globals_per_head": 8, "teleports_per_head": 4, "use_teleports": false, "use_triton": false}'

# Exp 1 at 128K with k=4096 (push the budget further)
run_ruler 1 131072 '{"top_k": 4096, "low_rank_dim": 128, "use_triton": false}'

echo "=== 128K SWEEP COMPLETE: $(date -Iseconds) ==="
