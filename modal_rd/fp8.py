"""Weight-only FP8 (E4M3) for the decoder's linear layers.

Each decoder layer's q/k/v/o and MLP projections keep their weights as FP8 with one scale per output row
(~7 GB of the 16 GB of Llama-8B weights are saved); activations stay BF16. The embedding table and lm_head
stay BF16.
  * one token (decode): `kernels/fp8_gemv.py` reads the FP8 weights directly, so batch-1 decode, which reads
    every weight once per token, moves half the bytes through the linear layers;
  * more tokens (prefill, appended turns): the weight is converted to BF16 for that call and multiplied with
    the usual BF16 GEMM, so prefill speed is about unchanged.
Lossy (every weight rounded to 3 mantissa bits); check accuracy before use. FP8-activation GEMMs
(`torch._scaled_mm`) were slower at batch 1: per-token activation quantization costs more than it saves.

  from modal_rd.fp8 import install_fp8
  install_fp8(model)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.fp8_gemv import fp8_dequant, fp8_gemv

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max


def quantize_rows(x):
    """[m, k] -> (FP8 [m, k], FP32 scales [m, 1]) with each row scaled to the FP8 range."""
    scale = x.abs().amax(-1, keepdim=True).float().clamp(min=1e-12) / FP8_MAX
    return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale


class FP8Linear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        q, s = quantize_rows(linear.weight.detach())                # one scale per output row
        self.register_buffer("weight_fp8", q)                        # [out, in]
        self.register_buffer("weight_scale", s.view(-1))             # [out] FP32
        self.bias = linear.bias
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.out_dtype = linear.weight.dtype

    def dequantized(self):
        if self.weight_fp8.is_cuda:
            return fp8_dequant(self.weight_fp8.view(torch.uint8), self.weight_scale, self.out_dtype)
        return (self.weight_fp8.float() * self.weight_scale[:, None]).to(self.out_dtype)

    def forward(self, x):
        if x.numel() == self.in_features and x.is_cuda:              # one token: FP8 GEMV
            y = fp8_gemv(x.reshape(-1), self.weight_fp8.view(torch.uint8), self.weight_scale)
            if self.bias is not None:
                y = y + self.bias
            return y.view(*x.shape[:-1], self.out_features)
        return F.linear(x, self.dequantized(), self.bias)


def install_fp8(model):
    """Replace every decoder layer's nn.Linear (attention and MLP projections) with FP8Linear; idempotent.

    Projections are swapped on the original attention module (`_rd_base` under modal_rd.attention wrappers)
    and the wrapper is pointed at the same modules, so arms installed later reuse the FP8 projections."""
    for layer in model.model.layers:
        for parent in (getattr(layer.self_attn, "_rd_base", layer.self_attn), layer.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, FP8Linear(child))
        base = getattr(layer.self_attn, "_rd_base", None)
        if base is not None:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(layer.self_attn, name, getattr(base, name))
    torch.cuda.empty_cache()
    return model
