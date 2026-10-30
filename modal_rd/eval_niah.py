"""Paired NIAH exact-number eval with KV-cached generation on one H100.

Every arm: same prompts, BF16, greedy, max_new_tokens=10, KV cache on.
Prefill = first full-context forward (CUDA events). Sparse arms use sparse
prefill + exact cached decode. `--graph` decodes with a captured CUDA graph
(modal_rd/graph_decode.py: same greedy tokens, no per-token Python overhead).
`--warm all` runs every prompt once untimed first (Triton specialises on sequence
length), then times the same prompts.

  python -m modal_rd.eval_niah --arm dense --bytes 512000 --n 6
  python -m modal_rd.eval_niah --arm vs --attn '{"window":4096,"vertical":2048}' --bytes 512000 --n 6
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

from modal_rd import attention as rd
from modal_rd.cpu_embed import install_cpu_embedding
from modal_rd.graph_decode import graph_generate
from modal_rd.kv_int8 import make_int8_cache
from modal_rd.mlp_chunk import install_chunked_mlp
from modal_rd.niah import MODEL_PATH, load_tokenizer, prepare_task
from modal_rd.model_config import MAX_CONTEXT, model_metadata, tokenize_prompt

RESULTS = Path("/vol/results")


def parse_layers(spec):
    out = []
    for part in filter(None, spec.split(",")):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def build(arm, attn, dense_layers):
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 low_cpu_mem_usage=True).cuda().eval()
    if arm in ("vs", "dense_rd"):
        rd.install(model, arm, dense_layers=dense_layers, **attn)
    elif arm == "exp19":
        from experiments.exp_19_bigger_bird_flash.model_llama import BiggerBirdFlashAttention
        for layer in model.model.layers:
            layer.self_attn = BiggerBirdFlashAttention(layer.self_attn, **attn)
            layer.self_attn.is_causal = True
            layer.self_attn.eval()
    elif arm != "dense":
        raise ValueError(arm)
    return model


def eos_ids(model):
    """The stop tokens HF generate uses (generation_config only), so --graph returns the same ids."""
    e = model.generation_config.eos_token_id
    return {*([e] if isinstance(e, int) else e or [])}


def run_one(model, tokenizer, row, arm, max_new, kv_int8=False, graph=False):
    if row["tokens"] + max_new > MAX_CONTEXT:
        raise ValueError("Prompt plus generation exceeds the checkpoint context")
    events = []
    pre = model.register_forward_pre_hook(lambda m, a, kw: events.append([torch.cuda.Event(enable_timing=True), None, kw["input_ids"].shape[1]]) or events[-1][0].record(), with_kwargs=True)
    def post(m, a, kw, o):
        events[-1][1] = torch.cuda.Event(enable_timing=True)
        events[-1][1].record()
    post_h = model.register_forward_hook(post, with_kwargs=True)
    try:
        inputs = tokenize_prompt(tokenizer, row["text"], return_tensors="pt", truncation=False).to("cuda")
        assert inputs["input_ids"].shape[1] == row["tokens"]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            if graph:
                gen = graph_generate(model, inputs["input_ids"], max_new, eos_ids(model))[0]
                out = torch.cat([inputs["input_ids"], torch.tensor([gen], device="cuda")], 1)
            else:
                cache_kw = {"past_key_values": make_int8_cache(model.config)} if kv_int8 else {}
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, use_cache=True,
                                     pad_token_id=tokenizer.pad_token_id, **cache_kw)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    finally:
        pre.remove()
        post_h.remove()
    gen_ids = out[0, row["tokens"]:].tolist()
    gen = tokenizer.decode(gen_ids, skip_special_tokens=True)
    first = re.search(r"\d+(?:,\d{3})*", gen)
    number = first[0].replace(",", "") if first else None
    if "answers" in row:
        # RULER-style string match: every gold value must appear in the output.
        flat = re.sub(r"\s", "", gen)
        hits = [a in flat for a in row["answers"]]
        exact, partial = all(hits), sum(hits) / len(hits)
    else:
        exact = bool(number == row["answer"])
        partial = float(exact)
    routes = None
    attn0 = model.model.layers[-1].self_attn
    if getattr(attn0, "last_routes", None) is not None and row.get("spans"):
        lo, hi = row["spans"]["number"]
        hit = []
        for layer in model.model.layers:
            r = getattr(layer.self_attn, "last_routes", None)
            if r is not None:
                hit.append(((r >= lo) & (r < hi)).sum(-1).ge(hi - lo).float().mean().item())
        routes = {"mean_head_fraction_full_number_routed": sum(hit) / max(1, len(hit)), "slots": int(attn0.last_routes.shape[-1])}
    return {"idx": row["idx"], "tokens": row["tokens"], "token_sha256": row["token_sha256"], "answer": row["answer"],
            "answers": row.get("answers", [row["answer"]]), "generated": gen, "generated_ids": gen_ids,
            "exact": exact, "partial": partial,
            "last_digit": bool(number and int(number[-1]) == row["label"]),
            "prefill_s": events[0][0].elapsed_time(events[0][1]) / 1000, "wall_s": wall,
            "forwards": len(events), "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "routes": routes}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="dense", choices=["dense", "dense_rd", "vs", "exp19"])
    p.add_argument("--attn", default="{}")
    p.add_argument("--dense-layers", default="")
    p.add_argument("--kv-int8", action="store_true", help="store the KV cache as INT8 (per token per KV head)")
    p.add_argument("--graph", action="store_true", help="CUDA-graph decode (modal_rd/graph_decode.py; not with --kv-int8)")
    p.add_argument("--cpu-embed", action="store_true", help="keep the input embedding table on the CPU (exact, -1.05 GB)")
    p.add_argument("--mlp-chunk", type=int, default=0, help="run the MLP in slices of this many tokens (0 = off)")
    p.add_argument("--bytes", default="512000")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--depths", default="0.5")
    p.add_argument("--max-new", type=int, default=10)
    p.add_argument("--warm", default="all", choices=["all", "first", "none"])
    p.add_argument("--task", default="niah", help="RULER NIAH variant, e.g. niah_multikey_1, niah_multivalue")
    p.add_argument("--question-first", action="store_true", help="move the question to the start of the prompt")
    p.add_argument("--question-offset", type=int, default=0, help="with --question-first: characters of filler before the question")
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    if a.graph and a.kv_int8:
        p.error("--graph does not support --kv-int8")
    attn, dense_layers = json.loads(a.attn), parse_layers(a.dense_layers)
    tokenizer = load_tokenizer()
    model = build(a.arm, attn, dense_layers)
    if a.mlp_chunk:
        install_chunked_mlp(model, a.mlp_chunk)
    if a.cpu_embed:
        install_cpu_embedding(model)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{a.tag}.json"
    import transformers
    report = {"transformers": transformers.__version__, **model_metadata(), "arm": a.arm, "task": a.task, "question_first": a.question_first, "question_offset": a.question_offset, "attn": attn, "mlp_chunk": a.mlp_chunk, "cpu_embed": a.cpu_embed, "kv_int8": a.kv_int8, "graph": a.graph, "dense_layers": dense_layers, "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "max_new": a.max_new, "cache": True, "warm": a.warm, "runs": [], "progress": []}
    for nbytes in map(int, a.bytes.split(",")):
        for depth in map(float, a.depths.split(",")):
            rows, skipped = prepare_task(tokenizer, a.task, nbytes, a.n, depth=depth, start=a.start,
                                         question_first=a.question_first, question_offset=a.question_offset)
            warm = rows if a.warm == "all" else rows[:1] if a.warm == "first" else []
            manifest = {"bytes": nbytes, "depth": depth, "model": model_metadata(),
                        "rows": [{k: v for k, v in row.items() if k != "text"} for row in rows]}
            out_path.with_name(f"{a.tag}_bytes{nbytes}_depth{depth}_prompts.json").write_text(json.dumps(manifest, indent=2))
            warmed = []
            for row in warm:
                w = run_one(model, tokenizer, row, a.arm, a.max_new, a.kv_int8, a.graph)
                warmed.append(w)
                report["progress"].append({"phase": "warm", "bytes": nbytes, "depth": depth, **w})
                out_path.write_text(json.dumps(report, indent=2))
                print(f"WARM bytes={nbytes} depth={depth} idx={row['idx']} exact={w['exact']} gen={w['generated']!r}", flush=True)
            res = []
            for row in rows:
                r = run_one(model, tokenizer, row, a.arm, a.max_new, a.kv_int8, a.graph)
                res.append(r)
                report["progress"].append({"phase": "timed", "bytes": nbytes, "depth": depth, **r})
                out_path.write_text(json.dumps(report, indent=2))
                print(f"TIMED bytes={nbytes} depth={depth} idx={row['idx']} tok={r['tokens']} exact={r['exact']} "
                      f"prefill={r['prefill_s']:.2f}s wall={r['wall_s']:.2f}s gen={r['generated']!r} routes={r['routes']}", flush=True)
            summary = {"task": a.task, "bytes": nbytes, "depth": depth, "n": len(res), "skipped_ambiguous": skipped,
                       "exact": sum(r["exact"] for r in res), "partial": sum(r["partial"] for r in res) / len(res),
                       "last_digit": sum(r["last_digit"] for r in res),
                       "mean_prefill_s": sum(r["prefill_s"] for r in res) / len(res),
                       "mean_tokens": sum(r["tokens"] for r in res) / len(res),
                       "peak_gb": max(r["peak_gb"] for r in res), "examples": res, "warmed_examples": warmed,
                       "warm_timed_equal": all(w["generated_ids"] == next(r["generated_ids"] for r in res if r["idx"] == w["idx"]) for w in warmed)}
            report["runs"].append(summary)
            out_path.write_text(json.dumps(report, indent=2))
            print(f"SUMMARY arm={a.arm} task={a.task} bytes={nbytes} depth={depth} exact={summary['exact']}/{len(res)} "
                  f"partial={summary['partial']:.3f} tokens={summary['mean_tokens']:.0f} "
                  f"last_digit={summary['last_digit']}/{len(res)} prefill={summary['mean_prefill_s']:.2f}s", flush=True)
    print("DONE", out_path, flush=True)


if __name__ == "__main__":
    main()
