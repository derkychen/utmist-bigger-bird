"""Runtime + parity benchmark for exp_2 Lightning Hybrid: Triton kernels vs PyTorch.

Compares the fused band + ELU-linear Triton path against the pure-PyTorch
fallback on the encoder self-attention module, across sequence lengths.

Run:
    python benchmarks/exp2_kernel_benchmark.py
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartAttention

from exp_2_lightning_hybrid.model import LightningHybridAttention
from shared.kernels import triton_available

EMBED_DIM = 768
NUM_HEADS = 12
BLOCK_SIZE = 128
BATCH = 2
SEQ_LENS = [256, 512, 1024, 2048]
WARMUP = 3
ITERS = 20


def _build_modules(device, dtype):
    """One base attention -> two LightningHybrid wrappers sharing identical weights."""
    cfg = BartConfig(d_model=EMBED_DIM, encoder_attention_heads=NUM_HEADS)
    base = BartAttention(
        embed_dim=EMBED_DIM, num_heads=NUM_HEADS, dropout=0.0, is_decoder=False, bias=True, config=cfg
    )
    triton_mod = LightningHybridAttention(base, block_size=BLOCK_SIZE, use_triton=True)
    torch_mod = LightningHybridAttention(copy.deepcopy(base), block_size=BLOCK_SIZE, use_triton=False)
    triton_mod.load_state_dict(torch_mod.state_dict())
    return (
        triton_mod.to(device=device, dtype=dtype).eval(),
        torch_mod.to(device=device, dtype=dtype).eval(),
    )


def _make_inputs(seq_len, device, dtype):
    torch.manual_seed(seq_len)
    hidden = torch.randn(BATCH, seq_len, EMBED_DIM, device=device, dtype=dtype)
    # Realistic padding: each sample keeps a random prefix length.
    keep = torch.randint(seq_len // 2, seq_len + 1, (BATCH,), device=device)
    pos = torch.arange(seq_len, device=device).unsqueeze(0)
    pad_2d = (pos < keep.unsqueeze(1)).to(dtype)
    # HF-style additive 4D mask: 0 where attend, large-negative where masked.
    ext = (1.0 - pad_2d)[:, None, None, :] * torch.finfo(dtype).min
    return hidden, ext, pad_2d.bool()


@torch.no_grad()
def _timed(mod, hidden, mask):
    for _ in range(WARMUP):
        mod(hidden, attention_mask=mask)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = mod(hidden, attention_mask=mask)[0]
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e3, out


def main():
    if not torch.cuda.is_available():
        print("CUDA not available - Triton kernels require a CUDA GPU. Aborting.")
        return
    if not triton_available():
        print("triton_available() is False - kernels cannot run on this machine. Aborting.")
        return

    device = torch.device("cuda")
    dtype = torch.float32
    print(f"Device: {torch.cuda.get_device_name(0)}  dtype: {dtype}")
    print(f"batch={BATCH} heads={NUM_HEADS} dim={EMBED_DIM} block_size={BLOCK_SIZE} "
          f"(global branch active when seq > {BLOCK_SIZE * 4})\n")

    triton_mod, torch_mod = _build_modules(device, dtype)

    header = f"{'seq':>6} | {'torch (ms)':>11} | {'triton (ms)':>12} | {'speedup':>8} | {'max|Δ|':>10} | branches"
    print(header)
    print("-" * len(header))

    for seq_len in SEQ_LENS:
        hidden, mask, valid = _make_inputs(seq_len, device, dtype)
        branches = "local+global" if seq_len > BLOCK_SIZE * 4 else "local"

        try:
            t_torch, out_torch = _timed(torch_mod, hidden, mask)
        except RuntimeError as e:
            torch.cuda.empty_cache()
            t_torch, out_torch = None, None
            if "out of memory" not in str(e).lower():
                raise

        t_triton, out_triton = _timed(triton_mod, hidden, mask)

        if t_torch is None:
            print(f"{seq_len:>6} | {'OOM':>11} | {t_triton:>12.3f} | "
                  f"{'  n/a':>7}  | {'   n/a':>10} | {branches}")
        else:
            # Compare only valid (non-padding) query rows; fully-masked padding
            # rows have undefined attention output and are pooled out downstream.
            vmask = valid.unsqueeze(-1)
            diff = torch.where(vmask, (out_torch - out_triton).abs(), torch.zeros_like(out_torch))
            max_diff = diff.max().item()
            speedup = t_torch / t_triton
            print(f"{seq_len:>6} | {t_torch:>11.3f} | {t_triton:>12.3f} | "
                  f"{speedup:>7.2f}x | {max_diff:>10.2e} | {branches}")
        sys.stdout.flush()
        del hidden, mask, out_torch, out_triton
        torch.cuda.empty_cache()

    print("\nParity note: max|Δ| is the largest abs difference between the Triton")
    print("and PyTorch outputs (should be ~1e-3 or smaller in fp32).")


if __name__ == "__main__":
    main()
