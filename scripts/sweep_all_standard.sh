#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_standard_%j.out

# Standardized sweep: RULER niah + LRA Text for all 6 experiments
# 30 examples per run, depth=0.5, seed=42, eval-samples=128
# Sequence lengths: 512, 1024, 2048, 4096, 8192
# Stops at first OOM per experiment per track

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

EXPS="0 1 8 13 14 15"
SEQS="512 1024 2048 4096 8192"
MAX_EXAMPLES=30
EVAL_SAMPLES=128

run_ruler() {
    local exp=$1 seq=$2
    echo "=== RULER niah | exp $exp | seq $seq | $(date -Iseconds) ==="
    python -m eval.ruler_llama.run_generative \
        --task niah --exp "$exp" --seq "$seq" --depth 0.5 \
        --eval-samples $EVAL_SAMPLES --max-examples $MAX_EXAMPLES \
        2>&1 | tail -5
    echo ""
}

run_lra_text() {
    local exp=$1 seq=$2
    echo "=== LRA text | exp $exp | seq $seq | $(date -Iseconds) ==="
    python -m eval.lra_llama.run_generative_text \
        --task text --exp "$exp" --seq "$seq" \
        --eval-samples $EVAL_SAMPLES --max-examples $MAX_EXAMPLES \
        2>&1 | tail -5
    echo ""
}

# Phase 3: RULER niah sweep (512-8K)
echo "========================================"
echo "PHASE 3: RULER niah sweep (512-8K)"
echo "========================================"
for exp in $EXPS; do
    for seq in $SEQS; do
        run_ruler "$exp" "$seq" || echo "  FAILED/OOM: exp $exp seq $seq"
    done
done

# Phase 4: LRA Text sweep (512-8K)
echo "========================================"
echo "PHASE 4: LRA Text sweep (512-8K)"
echo "========================================"
for exp in $EXPS; do
    for seq in $SEQS; do
        run_lra_text "$exp" "$seq" || echo "  FAILED/OOM: exp $exp seq $seq"
    done
done

# Regenerate dashboard
echo "========================================"
echo "Regenerating dashboard..."
echo "========================================"
python3 scripts/build_dashboard.py

echo "=== SWEEP COMPLETE: $(date -Iseconds) ==="
