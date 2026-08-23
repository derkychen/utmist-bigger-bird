#!/usr/bin/env python3
"""Exp 18 v3 — Multi-resolution sparse attention RULER test.

Run on a standalone H100 (no Slurm needed):
  python run_exp18_local.py --seq 4096 --max-examples 10      # 4K quick test
  python run_exp18_local.py --seq 262144 --max-examples 10    # 256K crossover test

Prerequisites:
  - DeepSeek-R1-Distill-Llama-8B model downloaded
  - Python env with torch, transformers, triton
  - Set MODEL_PATH below to where the model lives
"""
import argparse
import os
import sys

# === CONFIG: Change this to your model path ===
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/scratch/thomas7/models/DeepSeek-R1-Distill-Llama-8B",
)
# ==============================================

# Make sure we can import from the repo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Exp 18 RULER test (standalone H100)")
    parser.add_argument("--seq", type=int, default=4096, help="Context length")
    parser.add_argument("--max-examples", type=int, default=10, help="Number of examples")
    parser.add_argument("--depth", type=float, default=0.5, help="Needle depth")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Model path")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Override the exp_18 config to v3 settings
    import json
    v3_kwargs = json.dumps({
        "top_k": 2048,
        "low_rank_dim": 128,
        "window_size": 256,
        "gate_threshold": 0.5,
        "peak_threshold": -1.0,
        "linear_weight": 0.5,
        "use_triton": False,
        "always_global": True,
        "num_route_queries": 4,
        "adaptive_low_rank": True,
    })

    # Patch sys.argv to call the existing runner
    sys.argv = [
        "run_generative",
        "--task", "niah",
        "--exp", "18",
        "--seq", str(args.seq),
        "--depth", str(args.depth),
        "--eval-samples", "128",
        "--max-examples", str(args.max_examples),
        "--attn-kwargs", v3_kwargs,
    ]

    # Override model path in the runner
    import eval.ruler_llama.run_generative as rg
    rg.MODEL_PATH = args.model_path

    rg.main()

if __name__ == "__main__":
    main()
