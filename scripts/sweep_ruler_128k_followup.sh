#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_128k_followup_%j.out
#SBATCH --exclude=fc11013

# Follow-up 128K runs:
# 1. exp 15 at 128K with 15 examples (30 timed out in dynctx job)
# 2. exp 10 at 128K with k=2048 (k=1024 only got 13.3%)
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
echo "128K follow-up: exp 15 + exp 10 k=2048"
echo "$(date -Iseconds)"
echo "========================================"

# Exp 15 at 128K with 15 examples
run_ruler 15 131072 '{"fragment_size": 128, "max_k": 512, "min_k": 128, "globals_per_head": 8, "teleports_per_head": 4, "use_teleports": false, "use_triton": false}' 15

# Exp 10 at 128K with k=2048 (15 examples)
run_ruler 10 131072 '{"top_k": 2048, "low_rank_dim": 128, "use_triton": false}' 15

echo "=== COMPLETE: $(date -Iseconds) ==="
