#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_rd_%j.out

# R&D sweep: new parameter-free experiments + tuned budgets for existing ones
#
# Priority order:
#   1. Exp 1 (DeepSeek top-k) with k=1024, k=2048 at 128K — push the winner
#   2. Exp 16 (Free NSA) — parameter-free NSA at all lengths
#   3. Exp 17 (Coarse-to-fine) — two-stage routing at all lengths
#   4. Exp 15 (BiggerBird proper) — already parameter-free in causal mode
#   5. Exp 4, 10 with larger budgets — parameter-free but under-budgeted
#
# All experiments use parameter-free routing (no learned gates/compression).
# All run zero-shot (no training), which is the correct RULER protocol.

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
        echo "  OOM/FAILED: exp $exp seq $seq — marking as OOM"
        oom_state[$exp]=1
        echo ""
    fi
}

echo "========================================"
echo "RULER niah R&D sweep"
echo "Focus: parameter-free sparse methods"
echo "$(date -Iseconds)"
echo "========================================"

# --- Phase 1: Exp 1 with larger k at 128K (push the winner) ---
echo ""
echo "--- Phase 1: Exp 1 (DeepSeek top-k) with larger k at 128K ---"
run_ruler 1 131072 '{"top_k": 1024, "low_rank_dim": 128, "use_triton": false}'
run_ruler 1 131072 '{"top_k": 2048, "low_rank_dim": 128, "use_triton": false}'
# Also test k=1024 at 64K for comparison
run_ruler 1 65536 '{"top_k": 1024, "low_rank_dim": 128, "use_triton": false}'

# --- Phase 2: Exp 16 (Free NSA) at all lengths ---
echo ""
echo "--- Phase 2: Exp 16 (Free NSA, parameter-free) ---"
for seq in 4096 8192 16384 32768 65536 131072; do
    run_ruler 16 $seq '{"block_size": 64, "topk_blocks": 16, "window_size": 256, "use_triton": false}'
done

# --- Phase 3: Exp 17 (Coarse-to-fine) at all lengths ---
echo ""
echo "--- Phase 3: Exp 17 (Coarse-to-fine, two-stage routing) ---"
for seq in 4096 8192 16384 32768 65536 131072; do
    run_ruler 17 $seq '{"block_size": 128, "topk_blocks": 32, "fine_k": 512, "window_size": 256, "use_triton": false}'
done

# --- Phase 4: Exp 15 (BiggerBird proper) with tuned params ---
echo ""
echo "--- Phase 4: Exp 15 (BiggerBird proper, parameter-free causal) ---"
for seq in 4096 8192 16384 32768 65536 131072; do
    run_ruler 15 $seq '{"fragment_size": 128, "max_k": 1024, "min_k": 128, "globals_per_head": 32, "teleports_per_head": 8, "use_teleports": false, "use_triton": false}'
done

# --- Phase 5: Exp 4, 10 with larger budgets ---
echo ""
echo "--- Phase 5: Exp 4, 10 with larger budgets ---"
for seq in 4096 8192 16384 32768 65536; do
    run_ruler 4 $seq '{"block_size": 128, "num_blocks": 8, "use_triton": false}'
    run_ruler 10 $seq '{"top_k": 512, "low_rank_dim": 128, "use_triton": false}'
done

# --- Phase 6: Exp 17 with larger fine_k at 128K ---
echo ""
echo "--- Phase 6: Exp 17 with larger fine_k at 128K ---"
run_ruler 17 131072 '{"block_size": 128, "topk_blocks": 64, "fine_k": 1024, "window_size": 512, "use_triton": false}'
run_ruler 17 65536 '{"block_size": 128, "topk_blocks": 64, "fine_k": 1024, "window_size": 512, "use_triton": false}'

echo ""
echo "=== R&D SWEEP COMPLETE: $(date -Iseconds) ==="
