#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_exp17_bigbudget_%j.out
#SBATCH --exclude=fc11013

# Exp 17 with larger block budgets to test coverage hypothesis
# topk_blocks=128 at 32K (50% coverage) and 64K (25% coverage)
# topk_blocks=256 at 64K (50% coverage)
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
echo "Exp 17 larger block budgets"
echo "$(date -Iseconds)"
echo "========================================"

# topk_blocks=128 at 32K (50% coverage) — should recover 100%
run_ruler 17 32768 '{"block_size": 128, "topk_blocks": 128, "fine_k": 512, "window_size": 256, "use_triton": false}' 30

# topk_blocks=128 at 64K (25% coverage) — test if 25% is enough
run_ruler 17 65536 '{"block_size": 128, "topk_blocks": 128, "fine_k": 512, "window_size": 256, "use_triton": false}' 30

# topk_blocks=256 at 64K (50% coverage) — should recover if 50% is the threshold
run_ruler 17 65536 '{"block_size": 128, "topk_blocks": 256, "fine_k": 512, "window_size": 256, "use_triton": false}' 15

echo "=== EXP 17 BIG BUDGET SWEEP COMPLETE: $(date -Iseconds) ==="
