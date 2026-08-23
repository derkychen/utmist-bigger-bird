#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_fixed_exps_%j.out

# RULER niah for experiments 2-12 with causal fallback fix
set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

EXPS="7 10 2 3 4 5 6 9 11 12"
SEQS="4096 8192 16384 32768 65536 131072"
MAX_EXAMPLES=30
EVAL_SAMPLES=128

declare -A oom_state

run_ruler() {
    local exp=$1 seq=$2
    if [[ "${oom_state[$exp]:-0}" == "1" ]]; then
        echo "  SKIP exp $exp seq $seq (already OOM'd)"
        return 0
    fi
    echo "=== RULER niah | exp $exp | seq $seq | $(date -Iseconds) ==="
    if python -m eval.ruler_llama.run_generative \
        --task niah --exp "$exp" --seq "$seq" --depth 0.5 \
        --eval-samples $EVAL_SAMPLES --max-examples $MAX_EXAMPLES \
        2>&1 | tail -8; then
        echo ""
    else
        echo "  OOM/FAILED: exp $exp seq $seq — marking as OOM"
        oom_state[$exp]=1
        echo ""
    fi
}

echo "========================================"
echo "RULER niah FIXED exps 2-12 (causal fallback)"
echo "========================================"

for exp in $EXPS; do
    for seq in $SEQS; do
        run_ruler "$exp" "$seq"
    done
done

echo "========================================"
echo "Regenerating dashboard..."
echo "========================================"
python3 scripts/build_dashboard.py

echo "=== RULER FIXED EXPS COMPLETE: $(date -Iseconds) ==="
