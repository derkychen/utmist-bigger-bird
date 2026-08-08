#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_lra_text_%j.out

# LRA Text (IMDb sentiment) sweep: 512-8K standard + 16K-128K OOM push
# 30 examples per run, seed=42, eval-samples=128

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

EXPS="0 1 8 13 14 15"
SHORT_SEQS="512 1024 2048 4096 8192"
LONG_SEQS="16384 32768 65536 131072"
MAX_EXAMPLES=30
EVAL_SAMPLES=128

declare -A oom_status

run_lra_text() {
    local exp=$1 seq=$2
    if [[ "${oom_status[$exp]:-0}" == "1" ]]; then
        echo "  SKIP LRA text exp $exp seq $seq (already OOM'd)"
        return 0
    fi
    echo "=== LRA text | exp $exp | seq $seq | $(date -Iseconds) ==="
    if python -m eval.lra_llama.run_generative_text \
        --task text --exp "$exp" --seq "$seq" \
        --eval-samples $EVAL_SAMPLES --max-examples $MAX_EXAMPLES \
        2>&1 | tail -5; then
        echo ""
    else
        echo "  OOM/FAILED: exp $exp seq $seq"
        oom_status[$exp]=1
        echo ""
    fi
}

echo "========================================"
echo "LRA TEXT: Standard sweep (512-8K)"
echo "========================================"
for exp in $EXPS; do
    for seq in $SHORT_SEQS; do
        run_lra_text "$exp" "$seq"
    done
done

echo "========================================"
echo "LRA TEXT: OOM push (16K-128K)"
echo "========================================"
for exp in $EXPS; do
    for seq in $LONG_SEQS; do
        run_lra_text "$exp" "$seq"
    done
done

echo "========================================"
echo "Regenerating dashboard..."
echo "========================================"
python3 scripts/build_dashboard.py

echo "=== LRA TEXT SWEEP COMPLETE: $(date -Iseconds) ==="
