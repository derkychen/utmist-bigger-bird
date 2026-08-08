#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_oom_%j.out

# OOM push: RULER niah + LRA Text at 16K, 32K, 64K, 128K
# Stops at first OOM per experiment per track
# 30 examples per run, depth=0.5, seed=42, eval-samples=128

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

EXPS="0 1 8 13 14 15"
LONG_SEQS="16384 32768 65536 131072"
MAX_EXAMPLES=30
EVAL_SAMPLES=128

# Track OOM state per experiment per track
declare -A ruler_oom
declare -A lra_oom

run_ruler() {
    local exp=$1 seq=$2
    local key="${exp}_${seq}"
    if [[ "${ruler_oom[$exp]:-0}" == "1" ]]; then
        echo "  SKIP RULER exp $exp seq $seq (already OOM'd at shorter seq)"
        return 0
    fi
    echo "=== RULER niah | exp $exp | seq $seq | $(date -Iseconds) ==="
    if python -m eval.ruler_llama.run_generative \
        --task niah --exp "$exp" --seq "$seq" --depth 0.5 \
        --eval-samples $EVAL_SAMPLES --max-examples $MAX_EXAMPLES \
        2>&1 | tail -5; then
        echo ""
    else
        echo "  OOM/FAILED: exp $exp seq $seq — marking as OOM"
        ruler_oom[$exp]=1
        echo ""
    fi
}

run_lra_text() {
    local exp=$1 seq=$2
    if [[ "${lra_oom[$exp]:-0}" == "1" ]]; then
        echo "  SKIP LRA text exp $exp seq $seq (already OOM'd at shorter seq)"
        return 0
    fi
    echo "=== LRA text | exp $exp | seq $seq | $(date -Iseconds) ==="
    if python -m eval.lra_llama.run_generative_text \
        --task text --exp "$exp" --seq "$seq" \
        --eval-samples $EVAL_SAMPLES --max-examples $MAX_EXAMPLES \
        2>&1 | tail -5; then
        echo ""
    else
        echo "  OOM/FAILED: exp $exp seq $seq — marking as OOM"
        lra_oom[$exp]=1
        echo ""
    fi
}

# Phase 5: RULER niah OOM push (16K-128K)
echo "========================================"
echo "PHASE 5: RULER niah OOM push (16K-128K)"
echo "========================================"
for exp in $EXPS; do
    for seq in $LONG_SEQS; do
        run_ruler "$exp" "$seq"
    done
done

# Phase 6: LRA Text OOM push (16K-128K)
echo "========================================"
echo "PHASE 6: LRA Text OOM push (16K-128K)"
echo "========================================"
for exp in $EXPS; do
    for seq in $LONG_SEQS; do
        run_lra_text "$exp" "$seq"
    done
done

# Regenerate dashboard
echo "========================================"
echo "Regenerating dashboard..."
echo "========================================"
python3 scripts/build_dashboard.py

echo "=== OOM PUSH COMPLETE: $(date -Iseconds) ==="
