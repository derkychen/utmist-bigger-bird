"""Sequence-chunked MLP: same outputs as the unchunked MLP, smaller activation peak.

Runs under pytest or `python scripts/run_tests_plain.py tests/test_mlp_chunking.py`.
"""
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

from scripts.bench_exp1_flash import chunk_mlps


class _Layer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = LlamaMLP(cfg)


class _Model(torch.nn.Module):
    """Just the `model.model.layers[i].mlp` path chunk_mlps walks."""
    def __init__(self, cfg, n_layers=2):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(_Layer(cfg) for _ in range(n_layers))


def _setup(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=64, intermediate_size=224)
    return _Model(cfg).eval()


def test_chunked_matches_unchunked():
    model = _setup()
    x = torch.randn(2, 37, 64)  # 37 is not a multiple of the chunk size
    ref = [layer.mlp(x) for layer in model.model.layers]
    chunk_mlps(model, 8)
    out = [layer.mlp(x) for layer in model.model.layers]
    # Same math; BLAS may block a different row count differently, so allow
    # fp32 rounding (~3e-7 observed) but nothing more.
    for r, o in zip(ref, out):
        assert (r - o).abs().max().item() < 1e-5


def test_short_input_skips_chunking():
    model = _setup()
    calls = []
    inner = model.model.layers[0].mlp.forward
    model.model.layers[0].mlp.forward = lambda x: calls.append(x.shape[1]) or inner(x)
    chunk_mlps(model, 8)
    model.model.layers[0].mlp(torch.randn(1, 5, 64))
    assert calls == [5]


def test_long_input_runs_in_chunks():
    model = _setup()
    calls = []
    inner = model.model.layers[0].mlp.forward
    model.model.layers[0].mlp.forward = lambda x: calls.append(x.shape[1]) or inner(x)
    chunk_mlps(model, 8)
    model.model.layers[0].mlp(torch.randn(1, 20, 64))
    assert calls == [8, 8, 4]


def test_rejects_nonpositive_chunk():
    try:
        chunk_mlps(_setup(), 0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
