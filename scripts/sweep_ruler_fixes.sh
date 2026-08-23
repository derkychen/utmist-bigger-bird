#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_fixes_%j.out
#SBATCH --exclude=fc11013

# Sweep for fixed experiments:
# - Exp 15 (BiggerBird): removed learned gate, use token-level top-k routing
# - Exp 17 (Coarse-to-fine): fixed numerical precision issue (fast path = exp 1)
# - Exp 6, 14: missing coverage (likely 0% but need to confirm)

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
        echo "  SKIP exp $exp seq $seq (already OOM'd)"
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
echo "RULER fixed experiments sweep"
echo "$(date -Iseconds)"
echo "========================================"

# --- Exp 15 (BiggerBird, fixed: token-level routing, no learned gate) ---
echo ""
echo "--- Exp 15 (BiggerBird fixed) ---"
for seq in 4096 8192 16384 32768 65536; do
    run_ruler 15 $seq '{"fragment_size": 128, "max_k": 512, "min_k": 128, "globals_per_head": 8, "teleports_per_head": 4, "use_teleports": false, "use_triton": false}'
done

# --- Exp 17 (Coarse-to-fine, fixed: fast path for short sequences) ---
echo ""
echo "--- Exp 17 (Coarse-to-fine fixed) ---"
for seq in 4096 8192 16384 32768 65536; do
    run_ruler 17 $seq '{"block_size": 128, "topk_blocks": 32, "fine_k": 512, "window_size": 256, "use_triton": false}'
done

# --- Exp 17 at 128K with larger budgets ---
echo ""
echo "--- Exp 17 at 128K with larger budgets ---"
run_ruler 17 131072 '{"block_size": 128, "topk_blocks": 64, "fine_k": 1024, "window_size": 512, "use_triton": false}'

# --- Exp 6 (DeepSeek + PBS): missing coverage ---
echo ""
echo "--- Exp 6 (DeepSeek + PBS) ---"
run_ruler 6 4096 '{"top_k": 512, "low_rank_dim": 128, "block_size": 128, "num_blocks": 8, "use_triton": false}'
run_ruler 6 8192 '{"top_k": 512, "low_rank_dim": 128, "block_size": 128, "num_blocks": 8, "use_triton": false}'

# --- Exp 14 (Token drop + DeepSeek): missing coverage ---
echo ""
echo "--- Exp 14 (Token drop + DeepSeek) ---"
run_ruler 14 4096 '{"top_k": 512, "low_rank_dim": 128, "drop_after_layer": 3, "drop_frac": 0.3, "use_triton": false}'
run_ruler 14 8192 '{"top_k": 512, "low_rank_dim": 128, "drop_after_layer": 3, "drop_frac": 0.3, "use_triton": false}'

echo ""
echo "=== FIXES SWEEP COMPLETE: $(date -Iseconds) ==="
