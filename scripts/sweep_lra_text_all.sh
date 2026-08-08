#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_lra_text_all_%j.out

# LRA Text generative eval for ALL 16 experiments
# Tests 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K
# 30 examples per run, seed=42, eval-samples=128

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 2>/dev/null || true
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/$USER/utmist-bigger-bird

EXPS="0 13 1 7 10 2 3 4 5 6 8 9 11 12 14 15"
SEQS="512 1024 2048 4096 8192 16384 32768 65536 131072"
MAX_EXAMPLES=30
EVAL_SAMPLES=128

declare -A oom_state

run_lra_text() {
    local exp=$1 seq=$2
    if [[ "${oom_state[$exp]:-0}" == "1" ]]; then
        echo "  SKIP exp $exp seq $seq (already OOM'd)"
        return 0
    fi
    echo "=== LRA text | exp $exp | seq $seq | $(date -Iseconds) ==="
    if python -m eval.lra_llama.run_generative_text \
        --task text --exp "$exp" --seq "$seq" \
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
echo "LRA Text sweep (512-128K, all 16 exps)"
echo "========================================"

for exp in $EXPS; do
    for seq in $SEQS; do
        run_lra_text "$exp" "$seq"
    done
done

echo "========================================"
echo "Regenerating dashboard..."
echo "========================================"
python3 scripts/build_dashboard.py

echo "=== LRA TEXT SWEEP COMPLETE: $(date -Iseconds) ==="
