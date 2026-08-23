#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_tuned_%j.out

# RULER niah sweep with TUNED causal sparse attention parameters
# Increased sparsity budgets for better coverage at long context
#
# Key changes from default:
#   - Top-k methods: k=512, low_rank_dim=128 (full dim), local_window=512
#   - Block methods: more blocks, larger blocks
#   - Anchor methods: more anchors, larger window
#   - Gate methods: more globals, larger window

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

# Tuned experiments with custom kwargs
# Format: "exp:kwargs"
# Using python -c to override EXP_REGISTRY kwargs at runtime

declare -A TUNED_KWARGS

# Top-k methods: increase k, low_rank_dim, local_window
TUNED_KWARGS[1]='{"top_k": 512, "low_rank_dim": 128, "use_triton": false}'
TUNED_KWARGS[7]='{"k_early": 512, "k_mid": 256, "k_late": 128, "low_rank_dim": 128, "use_triton": false}'
TUNED_KWARGS[10]='{"top_k": 512, "low_rank_dim": 128, "use_triton": false}'

# Block methods: more blocks, larger blocks
TUNED_KWARGS[4]='{"block_size": 128, "num_blocks": 8, "use_triton": false}'
TUNED_KWARGS[6]='{"top_k": 512, "low_rank_dim": 128, "block_size": 64, "num_blocks": 8, "use_triton": false}'

# Gate methods: more globals, larger window
TUNED_KWARGS[3]='{"window_size": 256, "num_globals": 64, "use_triton": false}'
TUNED_KWARGS[5]='{"window_size": 256, "local_k": 128, "num_globals": 64, "num_teleports": 32, "use_triton": false}'

# Anchor methods: more anchors, larger window
TUNED_KWARGS[9]='{"window_size": 256, "num_anchors": 64, "verify_every": 4, "use_triton": false}'

# NSA: more blocks selected
TUNED_KWARGS[11]='{"block_size": 64, "stride": 64, "topk_blocks": 16, "window_size": 256, "use_triton": false}'

# BiggerBird: larger budget
TUNED_KWARGS[15]='{"fragment_size": 128, "max_k": 1024, "min_k": 128, "globals_per_head": 32, "teleports_per_head": 8, "use_teleports": false, "use_triton": false}'

# Default (no tuning needed — already 100% coverage)
# exp 0 (Dense), exp 2 (Lightning), exp 12 (S2-HHST), exp 13 (DynCtx)

EXPS="${EXPS:-0 2 12 13 1 7 10 4 6 3 5 9 11 15}"
SEQS="${SEQS:-4096 8192 16384 32768 65536 131072}"
MAX_EXAMPLES=30
EVAL_SAMPLES=128

declare -A oom_state

run_ruler_tuned() {
    local exp=$1 seq=$2
    if [[ "${oom_state[$exp]:-0}" == "1" ]]; then
        echo "  SKIP exp $exp seq $seq (already OOM'd at shorter seq)"
        return 0
    fi
    echo "=== RULER niah | exp $exp | seq $seq | TUNED | $(date -Iseconds) ==="

    local kwargs_json="${TUNED_KWARGS[$exp]:-}"
    local extra_args=""
    if [ -n "$kwargs_json" ]; then
        extra_args="--attn-kwargs '$kwargs_json'"
        echo "  Using tuned kwargs: $kwargs_json"
    fi

    local cmd="python -m eval.ruler_llama.run_generative \
        --task niah --exp $exp --seq $seq --depth 0.5 \
        --eval-samples $EVAL_SAMPLES --max-examples $MAX_EXAMPLES"

    if [ -n "$kwargs_json" ]; then
        cmd="$cmd --attn-kwargs '$kwargs_json'"
    fi

    if eval "$cmd" 2>&1 | tail -8; then
        echo ""
    else
        echo "  OOM/FAILED: exp $exp seq $seq — marking as OOM"
        oom_state[$exp]=1
        echo ""
    fi
}

echo "========================================"
echo "RULER niah TUNED sweep"
echo "Exps: $EXPS"
echo "Seqs: $SEQS"
echo "Tuned parameters for better coverage"
echo "========================================"

for exp in $EXPS; do
    for seq in $SEQS; do
        run_ruler_tuned "$exp" "$seq"
    done
done

echo "=== TUNED SWEEP COMPLETE: $(date -Iseconds) ==="
