#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_combined_%j.out

# RULER niah sweep — COMBINED default + tuned parameters
# Group 1: High-coverage methods with default params (should work)
# Group 2-4: Low-coverage methods with TUNED params (increased budgets)
#
# Usage: sbatch --export=GROUP=1 scripts/sweep_ruler_combined.sh

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

GROUP="${GROUP:-1}"
MAX_EXAMPLES=30
EVAL_SAMPLES=128
SEQS="${SEQS:-4096 8192 16384 32768 65536 131072}"

# Declare tuned kwargs per experiment
declare -A TUNED_KWARGS

# Group 1: High-coverage (default params, no tuning needed)
# exp 0: Dense (100% coverage)
# exp 2: Lightning Hybrid (100% coverage, linear attention)
# exp 12: S2-HHST (100% coverage, strided)
# exp 13: DynCtx (100% at 4K, degrades — but already tuned)

# Group 2: Top-k methods (TUNED: k=512, lr=128, larger window)
TUNED_KWARGS[1]='{"top_k": 512, "low_rank_dim": 128, "use_triton": false}'
TUNED_KWARGS[7]='{"k_early": 512, "k_mid": 256, "k_late": 128, "low_rank_dim": 128, "use_triton": false}'
TUNED_KWARGS[10]='{"top_k": 512, "low_rank_dim": 128, "use_triton": false}'
TUNED_KWARGS[14]='{"drop_after_layer": 3, "drop_ratio": 0.3, "top_k": 512, "low_rank_dim": 128, "use_triton": false}'

# Group 3: Block/gate/anchor methods (TUNED: more blocks/globals/anchors, larger window)
TUNED_KWARGS[3]='{"window_size": 256, "num_globals": 64, "use_triton": false}'
TUNED_KWARGS[4]='{"block_size": 128, "num_blocks": 8, "use_triton": false}'
TUNED_KWARGS[5]='{"window_size": 256, "local_k": 128, "num_globals": 64, "num_teleports": 32, "use_triton": false}'
TUNED_KWARGS[6]='{"top_k": 512, "low_rank_dim": 128, "block_size": 64, "num_blocks": 8, "use_triton": false}'
TUNED_KWARGS[9]='{"window_size": 256, "num_anchors": 64, "verify_every": 4, "use_triton": false}'

# Group 4: NSA + BiggerBird + TokenDrop (TUNED)
TUNED_KWARGS[8]='{"drop_after_layer": 3, "drop_ratio": 0.3, "use_triton": false}'
TUNED_KWARGS[11]='{"block_size": 64, "stride": 64, "topk_blocks": 16, "window_size": 256, "use_triton": false}'
TUNED_KWARGS[15]='{"fragment_size": 128, "max_k": 1024, "min_k": 128, "globals_per_head": 32, "teleports_per_head": 8, "use_teleports": false, "use_triton": false}'

# Set experiments based on group
case $GROUP in
    1) EXPS="0 2 12 13" ;;
    2) EXPS="1 7 10 14" ;;
    3) EXPS="3 4 5 6 9" ;;
    4) EXPS="8 11 15" ;;
    *) EXPS="$GROUP" ;;  # Single experiment
esac

declare -A oom_state

run_ruler() {
    local exp=$1 seq=$2
    if [[ "${oom_state[$exp]:-0}" == "1" ]]; then
        echo "  SKIP exp $exp seq $seq (already OOM'd at shorter seq)"
        return 0
    fi
    echo "=== RULER niah | exp $exp | seq $seq | group $GROUP | $(date -Iseconds) ==="

    local kwargs_json="${TUNED_KWARGS[$exp]:-}"
    local cmd="python -m eval.ruler_llama.run_generative \
        --task niah --exp $exp --seq $seq --depth 0.5 \
        --eval-samples $EVAL_SAMPLES --max-examples $MAX_EXAMPLES"

    if [ -n "$kwargs_json" ]; then
        cmd="$cmd --attn-kwargs '$kwargs_json'"
        echo "  Tuned kwargs: $kwargs_json"
    fi

    if eval "$cmd" 2>&1 | tail -10; then
        echo ""
    else
        echo "  OOM/FAILED: exp $exp seq $seq — marking as OOM"
        oom_state[$exp]=1
        echo ""
    fi
}

echo "========================================"
echo "RULER niah COMBINED sweep (group $GROUP)"
echo "Exps: $EXPS"
echo "Seqs: $SEQS"
echo "========================================"

for exp in $EXPS; do
    for seq in $SEQS; do
        run_ruler "$exp" "$seq"
    done
done

echo "=== COMBINED SWEEP GROUP $GROUP COMPLETE: $(date -Iseconds) ==="
