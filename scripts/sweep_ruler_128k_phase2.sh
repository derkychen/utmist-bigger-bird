#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_128k_phase2_%j.out
#SBATCH --exclude=fc11013

# Phase 2 experiments:
# 1. exp 1 k=1536 at 128K (narrow down the phase transition)
# 2. exp 1 k=2048 at 128K depth=0.0 (test depth sensitivity)
# 3. exp 1 k=2048 at 128K depth=1.0 (test depth sensitivity)
# 4. exp 17 fine_k=2048 at 128K with topk_blocks=256 (fewer blocks, same fine budget)
set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

run_ruler() {
    local exp=$1 seq=$2 depth=$3 kwargs_json=$4 max_ex=$5
    echo "=== RULER niah | exp $exp | seq $seq | depth $depth | $(date -Iseconds) ==="
    local cmd="python -m eval.ruler_llama.run_generative \
        --task niah --exp $exp --seq $seq --depth $depth \
        --eval-samples 128 --max-examples $max_ex"
    if [ -n "$kwargs_json" ]; then
        cmd="$cmd --attn-kwargs '$kwargs_json'"
        echo "  kwargs: $kwargs_json  max_examples: $max_ex"
    fi
    if eval "$cmd" 2>&1 | tail -8; then
        echo ""
    else
        echo "  FAILED: exp $exp seq $seq depth $depth"
        echo ""
    fi
}

echo "========================================"
echo "Phase 2: phase transition + depth sensitivity"
echo "$(date -Iseconds)"
echo "========================================"

# 1. exp 1 k=1536 at 128K depth=0.5 (narrow the phase transition: 1024→0%, 2048→80%)
run_ruler 1 131072 0.5 '{"top_k": 1536, "low_rank_dim": 128, "use_triton": false}' 10

# 2. exp 1 k=2048 at 128K depth=0.0 (needle at start)
run_ruler 1 131072 0.0 '{"top_k": 2048, "low_rank_dim": 128, "use_triton": false}' 10

# 3. exp 1 k=2048 at 128K depth=1.0 (needle at end)
run_ruler 1 131072 1.0 '{"top_k": 2048, "low_rank_dim": 128, "use_triton": false}' 10

# 4. exp 17 fine_k=2048 with topk_blocks=256 at 128K (fewer blocks, same fine budget)
run_ruler 17 131072 0.5 '{"block_size": 128, "topk_blocks": 256, "fine_k": 2048, "window_size": 256, "use_triton": false}' 5

echo "=== PHASE 2 SWEEP COMPLETE: $(date -Iseconds) ==="
