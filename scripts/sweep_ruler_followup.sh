#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_followup_%j.out
#SBATCH --exclude=fc11013

# Follow-up sweep: test the promising experiments at longer sequences
# - Exp 10 (GQA sparse) with k=512 at 8K-128K (got 100% at 4K)
# - Exp 17 (Coarse-to-fine) with smaller block_size at 4K-64K
# - Exp 1 (DeepSeek top-k) with k=1024 at 64K (for comparison with full sweep's 128K)

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

MAX_EXAMPLES=30
EVAL_SAMPLES=128

declare -A oom_state

run_ruler() {
    local exp=$1 seq=$2 kwargs_json=$3
    if [[ "${oom_state[$exp]:-0}" == "1" ]]; then
        echo "  SKIP exp $exp seq $seq (already OOM'd at shorter seq)"
        return 0
    fi
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
        echo "  OOM/FAILED: exp $exp seq $seq"
        oom_state[$exp]=1
        echo ""
    fi
}

echo "========================================"
echo "RULER follow-up sweep"
echo "$(date -Iseconds)"
echo "========================================"

# --- Phase 1: Exp 10 (GQA sparse) with k=512 at 8K-128K ---
echo ""
echo "--- Phase 1: Exp 10 (GQA sparse, k=512) at 8K-128K ---"
for seq in 8192 16384 32768 65536 131072; do
    run_ruler 10 $seq '{"top_k": 512, "low_rank_dim": 128, "use_triton": false}'
done

# --- Phase 2: Exp 17 (Coarse-to-fine) with smaller block_size ---
echo ""
echo "--- Phase 2: Exp 17 with block_size=32 (finer blocks) ---"
for seq in 4096 8192 16384 32768 65536; do
    run_ruler 17 $seq '{"block_size": 32, "topk_blocks": 64, "fine_k": 512, "window_size": 256, "use_triton": false}'
done

# --- Phase 3: Exp 17 with block_size=64 ---
echo ""
echo "--- Phase 3: Exp 17 with block_size=64 ---"
for seq in 4096 8192 16384 32768 65536; do
    run_ruler 17 $seq '{"block_size": 64, "topk_blocks": 32, "fine_k": 512, "window_size": 256, "use_triton": false}'
done

# --- Phase 4: Exp 1 with k=1024 at 64K (for comparison) ---
echo ""
echo "--- Phase 4: Exp 1 with k=1024 at 64K ---"
run_ruler 1 65536 '{"top_k": 1024, "low_rank_dim": 128, "use_triton": false}'

echo ""
echo "=== FOLLOW-UP SWEEP COMPLETE: $(date -Iseconds) ==="
