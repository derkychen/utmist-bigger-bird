#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=/scratch/%u/sweep_ruler_dynctx_%j.out
#SBATCH --exclude=fc11013

# Exp 13 (Dynamic Context) with content-based routing (fixed from norm-based)
# Test at all sequence lengths with target_budget=512 (same as exp 1's k=512)

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
echo "Exp 13 (Dynamic Context) with content-based routing"
echo "$(date -Iseconds)"
echo "========================================"

# Exp 13 with target_budget=512 (same as exp 1's k=512)
for seq in 4096 8192 16384 32768 65536 131072; do
    run_ruler 13 $seq '{"drop_after_layer": 3, "target_budget": 512, "chunk_size": 8192, "use_triton": false}'
done

# Also test with target_budget=1024 at 64K and 128K
run_ruler 13 65536 '{"drop_after_layer": 3, "target_budget": 1024, "chunk_size": 8192, "use_triton": false}'
run_ruler 13 131072 '{"drop_after_layer": 3, "target_budget": 1024, "chunk_size": 8192, "use_triton": false}'

echo "=== DYNCTX SWEEP COMPLETE: $(date -Iseconds) ==="
