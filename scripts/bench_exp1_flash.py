"""Exp 1 flash vs torch attention backend on RULER niah (4-bit, bf16).

Loads the model once, switches each layer's ``attn_backend`` in place, and
runs the same examples per backend. The first example per backend is reported
separately as warm-up. Note: the flash kernel compiles once per new sequence
length (~0.7-0.9 s on the 3080), and uncached generation creates a new length on
every decode step, so ``mean_seconds`` for flash includes compile time too.

    python -m scripts.bench_exp1_flash --seqs 4096,16384 --backends flash,torch
    python -m scripts.bench_exp1_flash --seqs 32768,49152,65536 --backends flash
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from eval.lra_llama.lra_llama_dataset import _ids_to_text
from eval.ruler.ruler_dataset import build_ruler_dataset
from eval.ruler_llama import run_generative as ruler


def chunk_mlps(model, chunk: int) -> None:
    """Run each layer's MLP over sequence chunks of ``chunk`` tokens.

    The MLP is per-token, so outputs are unchanged; only the peak activation
    memory drops (the 14336-wide intermediate is built one chunk at a time).
    """
    if chunk < 1:
        raise ValueError("chunk must be at least 1")
    for layer in model.model.layers:
        forward = layer.mlp.forward

        def chunked(x, _forward=forward):
            if x.shape[1] <= chunk:
                return _forward(x)
            return torch.cat([_forward(part) for part in x.split(chunk, dim=1)], dim=1)

        layer.mlp.forward = chunked


def set_backend(model, backend: str) -> None:
    for layer in model.model.layers:
        layer.self_attn.attn_backend = backend


def run_backend(model, tokenizer, dataset, n: int) -> dict:
    times, predictions, correct = [], [], 0
    torch.cuda.reset_peak_memory_stats()
    for i in range(n):
        row = dataset[i]
        start = time.perf_counter()
        generated = ruler.generate_answer(model, tokenizer, _ids_to_text(row["input_ids"]), device="cuda")
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        pred = ruler.parse_prediction(generated, task="niah")
        correct += int(pred == row["labels"])
        predictions.append({"label": int(row["labels"]), "pred": int(pred), "generated": generated[:80]})
        print(f"  [{i + 1}/{n}] label={row['labels']} pred={pred} {times[-1]:.1f}s", flush=True)
    rest = times[1:] or times
    return {"status": "ok", "accuracy": correct / n, "n": n, "warmup_seconds": times[0],
            "mean_seconds": sum(rest) / len(rest),
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "predictions": predictions}


def probe_compile(seq: int) -> dict:
    """Kernel-only timing: first call at a new length vs a repeat call."""
    from kernels.bigger_bird_flash import bigger_bird_flash
    out = {}
    for t in (seq, seq + 1):
        q, k, v = (torch.randn(32, t, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        idx = torch.randperm(t, device="cuda")[:512].repeat(32, 1)[:, None, :]
        timings = []
        for _ in range(2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            bigger_bird_flash(q, k, v, idx, front=0, window=256, num_heads=32, scale=1.0)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        out[str(t)] = {"first_call_s": timings[0], "repeat_call_s": timings[1]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--seqs", default="4096,16384")
    parser.add_argument("--backends", default="flash,torch")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--low-rank-dim", type=int, default=128)
    parser.add_argument("--query-chunk", type=int, default=16)
    parser.add_argument("--mlp-chunk", type=int, default=0,
                        help="run MLPs over this many tokens at a time (0 = off)")
    parser.add_argument("--probe-compile", action="store_true")
    parser.add_argument("--output-dir", default="benchmarks/exp_1_flash")
    args = parser.parse_args()

    config = {"top_k": args.top_k, "low_rank_dim": args.low_rank_dim,
              "query_chunk": args.query_chunk, "use_triton": False}
    module, cls, defaults = ruler.EXP_REGISTRY[1]
    ruler.EXP_REGISTRY[1] = (module, cls, {**defaults, **config})
    tokenizer = ruler.load_tokenizer(args.model_path)
    model = ruler.build_generative_model(1, model_path=args.model_path,
                                         torch_dtype=torch.bfloat16, load_in_4bit=True)
    if args.mlp_chunk:
        chunk_mlps(model, args.mlp_chunk)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seq in (int(s) for s in args.seqs.split(",")):
        dataset = build_ruler_dataset(task="niah", seq_len=seq, needle_depth=0.5,
                                      train_samples=10, eval_samples=args.max_examples,
                                      seed=42)["validation"]
        result = {"seq_len": seq, "config": {**config, "mlp_chunk": args.mlp_chunk, "quantization": "4bit",
                  "dtype": "bfloat16", "gpu": torch.cuda.get_device_name(0)}, "backends": {}}
        if args.probe_compile:
            result["compile_probe"] = probe_compile(seq)
        for backend in args.backends.split(","):
            print(f"=== seq {seq} backend {backend} ===", flush=True)
            set_backend(model, backend)
            try:
                result["backends"][backend] = run_backend(model, tokenizer, dataset, args.max_examples)
                result["backends"][backend]["effective_backend"] = model.model.layers[0].self_attn.last_backend
            except torch.cuda.OutOfMemoryError:
                result["backends"][backend] = {"status": "oom"}
                torch.cuda.empty_cache()
            print(f"--- {backend}: {json.dumps({k: v for k, v in result['backends'][backend].items() if k != 'predictions'})}", flush=True)
        path = out_dir / f"bench_seq{seq}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(result, indent=2))
        print(f"saved: {path}", flush=True)


if __name__ == "__main__":
    main()
