# Experiment Commands Quick Reference

## List Available Configs
```bash
python run_experiment.py --list
```

## By Compute Size

### SMALL (Quick test / 8GB RAM / ~1GB mem usage)
```bash
# Quick sanity check - runs in ~5 min
python run_experiment.py --exp 0 --size small    # baseline (full attention)
python run_experiment.py --exp 1 --size small    # DeepSeek Top-K
python run_experiment.py --exp 2 --size small    # Lightning Hybrid
python run_experiment.py --exp 3 --size small    # Dynamic Globals
python run_experiment.py --exp 4 --size small    # PBS Attention
python run_experiment.py --exp 5 --size small    # Bigger Bird (unified)
python run_experiment.py --exp 6 --size small    # DeepSeek + PBS hybrid
python run_experiment.py --exp 7 --size small    # Layer-Adaptive Sparsity
python run_experiment.py --exp 8 --size small    # Token Dropping
python run_experiment.py --exp 9 --size small    # Attention Speculation
python run_experiment.py --exp 10 --size small   # GQA + Sparse Routing
python run_experiment.py --exp 11 --size small   # NSA (arXiv:2502.11089, Triton kernels)
python run_experiment.py --exp 12 --size small   # S2-Attention / HHST (arXiv:2407.17678)
```
**Config:** 500 samples, seq=256, batch=1, 2 epochs

### MEDIUM (Good GPU / 16GB VRAM / ~6GB mem usage)
```bash
# Decent training - runs in ~30 min
python run_experiment.py --exp 1 --size medium
python run_experiment.py --exp 2 --size medium
python run_experiment.py --exp 3 --size medium
python run_experiment.py --exp 4 --size medium
python run_experiment.py --exp 5 --size medium
python run_experiment.py --exp 6 --size medium
```
**Config:** 2000 samples, seq=512, batch=2, accum=4, 3 epochs

### BIG (Big GPU / 24GB+ VRAM / ~12GB mem usage)
```bash
# Full training - runs in ~2 hours
python run_experiment.py --exp 1 --size big
python run_experiment.py --exp 2 --size big
python run_experiment.py --exp 3 --size big
python run_experiment.py --exp 4 --size big
python run_experiment.py --exp 5 --size big
python run_experiment.py --exp 6 --size big
```
**Config:** 6000 samples, seq=768, batch=4, accum=8, 3 epochs

### XL (Full IMDb / Large GPU / ~20GB mem usage)
```bash
# Production run - full dataset
python run_experiment.py --exp 3 --size xl
```
**Config:** 25000 samples, seq=1024, batch=8, accum=16, 5 epochs

## With Custom Overrides

```bash
# Medium but with more epochs
python run_experiment.py --exp 3 --size medium --epochs 5

# Small batch size for low VRAM
python run_experiment.py --exp 3 --size big --batch 2 --accum 16

# Longer sequences with gradient checkpointing (saves memory)
python run_experiment.py --exp 3 --size medium --seq 768 --grad-checkpoint

# Save weights after training so you can re-run eval without retraining
python run_experiment.py --exp 1 --size big --save-weights

# Custom everything
python run_experiment.py --exp 3 --size small \
    --train-samples 1000 \
    --seq 512 \
    --batch 2 \
    --epochs 4 \
    --lr 2e-5
```

## Full Matrix: Experiment × Compute

### Original Methods (exp 1-4) + Baseline
| Exp | Description | Key Params |
|---|---|---|
| 0 | Baseline (full O(n²) dense) | - |
| 1 | DeepSeek Top-K | `top_k=64, low_rank_dim=16` |
| 2 | Lightning Hybrid | `block_size=128` |
| 3 | Dynamic Globals | `window_size=64, num_globals=16` |
| 4 | PBS Attention | `block_size=64, num_blocks=2` |

### New Hybrid / Advanced Ideas (exp 5-10)
| Exp | Description | Key Params | Why |
|---|---|---|---|
| 5 | Bigger Bird (unified) | `local_k=32 + globals=16 + teleports=8 = 56 keys/query` | Original 3-component proposal |
| 6 | DeepSeek + PBS hybrid | `top_k=64 within top-4 blocks, sorted indices` | Content-aware + coalesced GPU reads |
| 7 | Layer-Adaptive | `k_early=192, k_mid=64, k_late=32` | Different layers, different sparsity |
| 8 | Token Dropping | `drop 30% after layer 3` | Reduces n for ALL subsequent layers |
| 9 | Attention Speculation | `window=64 + 4 anchors, KL distill every 4 layers` | Fast path + verifier distillation |
| 10 | GQA + Sparse | `kv_groups=4, top_k=64` | Memory + softmax cost both reduced |

### Triton-Kernel Experiments (exp 11-12)
| Exp | Description | Key Params | Why |
|---|---|---|---|
| 11 | NSA (arXiv:2502.11089) | `block_size=32, stride=32, topk_blocks=4, window_size=128` | Compressed + selected + sliding-window branches with shared Triton kernels |
| 12 | S2-Attention / HHST (arXiv:2407.17678) | `shard_size=32, local_blocks=2, stride_blocks=16, use_sink=True` | Sharded strided attention with sink tokens |

Run ALL experiments (small smoke test):
```bash
for exp in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
  python run_experiment.py --exp $exp --size small
done
```

Run ALL new experiments at scale:
```bash
for exp in 5 6 7 8 9 10 11 12; do
  python run_experiment.py --exp $exp --size big
done
```

## Accelerated Kernels (training + torch.compile)

Two opt-in accelerators for training/eval:

- `BIGGER_BIRD_TRAIN_KERNELS=1` — use the autograd-capable Triton **gather** kernel
  on the training compute path (applies to exp 3, 5, 9; CUDA only, and only when the
  sparse branch is active, i.e. `--seq` larger than the experiment's key count).
  Off by default; inference already uses the forward-only fused kernels.
- `--compile` (or `BIGGER_BIRD_TORCH_COMPILE=1`) — wrap the model in `torch.compile`.
  Compiles cleanly; speedup is hardware/model dependent (roughly neutral on small GPUs).

```bash
# Training gather kernel + torch.compile (exp 5)
BIGGER_BIRD_TRAIN_KERNELS=1 python run_experiment.py --exp 5 --size medium --compile

# Training gather kernel only (drop --compile if it doesn't help on your GPU)
BIGGER_BIRD_TRAIN_KERNELS=1 python run_experiment.py --exp 5 --size small
```

> The training gather kernel does not apply attention dropout; this is safe because
> BART-base uses `attention_dropout=0.0`.

## Saving & Reloading Weights

Pass `--save-weights` to persist the model after training:
```bash
python run_experiment.py --exp 1 --size big --save-weights
```
Weights are saved to `benchmarks/<exp_name>/weights_<timestamp>/` and the path is recorded in the matching `eval_<timestamp>.json` under `experiment_metadata.weights_path`.

To reload for eval-only later:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

weights_dir = "benchmarks/exp_1_deepseek_topk/weights_20260516_165432"
model = AutoModelForSequenceClassification.from_pretrained(weights_dir)
tokenizer = AutoTokenizer.from_pretrained(weights_dir)
# Re-apply patches if needed (exp_1-4 only):
# from exp_1_deepseek_topk.model import patch_bart; patch_bart(model)
# For exp 11: from exp_11_nsa.model import PatchedModel  # wrap base_model with NSA patches
```

> **Note:** For patched experiments (exp 1–4) the *base* BART weights are saved (patches are
> re-applied at load time). For the baseline (exp 0) the full model is saved as-is.
>
> **Exp 11 (NSA):** Same as other patches — use `PatchedModel` from `exp_11_nsa.model` after loading weights.
>
> **Exp 12 (S2-HHST):** S2-Attention / HHST (arXiv:2407.17678) — use `PatchedModel` from `exp_12_s2_hhst.model` after loading weights.

## Long-Context Stress Tests

BART-base has a native limit of 1024 tokens. The `--long` preset extends position embeddings and uses fixed-length padding to stress the full context window.

```bash
# Single long-context run (seq=2048, 500 train samples, fixed padding)
python run_experiment.py --exp 5 --size long --grad-checkpoint

# Override sequence length beyond 2048 (extends position embeddings automatically)
python run_experiment.py --exp 5 --size long --seq 4096 --grad-checkpoint

# Baseline at 2048 (will likely OOM without grad-checkpoint on 16GB GPU)
python run_experiment.py --exp 0 --size long --seq 2048 --grad-checkpoint

# Full sweep: baseline vs sparse at 1024, 2048, 4096
python run_long_context_sweep.py --seqs 1024,2048,4096 --exps 0,3,5,8 --grad-checkpoint

# Sparse-only sweep (skip baseline OOMs)
python run_long_context_sweep.py --seqs 2048,4096,8192 --exps 3,5,8,10
```

**Key flags for long context:**
- `--size long` — 500 train samples, seq=2048, batch=1, fixed-length padding
- `--seq <N>` — override sequence length (auto-extends position embeddings if >1024)
- `--fixed-length` — pad every sample to max_length (full attention workload)
- `--grad-checkpoint` — essential for baseline at 2048+ to avoid OOM

## Visualize Results

```bash
# After running experiments
python viz/training_trajectory_viz.py  # Per-epoch curves
python viz/compare_experiments.py      # Cross-experiment comparison
python viz/scaling_laws_viz.py         # Scaling trends
```

## Memory-Saving Tips

For 8GB RAM machines:
```bash
# Always use small + gradient checkpointing
python run_experiment.py --exp 3 --size small --grad-checkpoint

# If still crashing, reduce sequence length
python run_experiment.py --exp 3 --size small --seq 128
```

For cloud GPU instances:
```bash
# SSH in first
ssh <user>@<server-ip>

# Then run big/xl experiments
python run_experiment.py --exp 3 --size xl
```
