#!/bin/bash
# Trillium: cd $SCRATCH/bigger-bird-r1 && sbatch kernels/bench_routed_window_slurm.sh
#SBATCH --job-name=routed-window-bench
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=1:00:00
#SBATCH --account=def-guerzhoy
#SBATCH --output=routed_window_bench_%j.out

set -euo pipefail
module load StdEnv/2023 gcc arrow cuda/12.6 python/3.11.5
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="${SCRATCH}/.cache/triton"  # $HOME is read-only on compute nodes
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m pytest kernels/tests tests -q
python -m kernels.bench_routed_window --seqs 4096,32768 --top-k 512,2048 --warmup 2 --iters 5
python -m kernels.bench_routed_window --seqs 65536,131072 --top-k 512,2048,4096,8192 \
  --warmup 2 --iters 5 --skip-old
