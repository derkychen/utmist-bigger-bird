"""Keep the input embedding table on the CPU.

The table is [vocab, hidden] = 128256 x 4096 BF16 = 1.05 GB, but a forward pass only reads one row per
token, so prefill copies n rows (n x 8 KB) to the GPU and decode copies one. The lookup is the same
row gather, so the output is bit-identical. The weight is held as a plain attribute (not a Parameter),
so `model.parameters()` / `model.device` still report the GPU and HF `generate` is unaffected.
"""
import torch
import torch.nn as nn


class CPUEmbedding(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.weight_cpu = embedding.weight.detach().to("cpu")
        if torch.cuda.is_available():
            self.weight_cpu = self.weight_cpu.pin_memory()
        self.padding_idx = embedding.padding_idx

    def forward(self, input_ids):
        rows = self.weight_cpu[input_ids.cpu()]
        if input_ids.is_cuda:
            rows = rows.pin_memory()
        return rows.to(input_ids.device, non_blocking=True)


def install_cpu_embedding(model):
    """Move `model.model.embed_tokens` to pinned CPU memory; frees its GPU copy. Idempotent."""
    emb = model.model.embed_tokens
    if not isinstance(emb, CPUEmbedding):
        model.model.embed_tokens = CPUEmbedding(emb)
        del emb
        torch.cuda.empty_cache()
    return model
