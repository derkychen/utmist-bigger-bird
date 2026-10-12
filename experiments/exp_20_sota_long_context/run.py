"""Exp 20: Bigger Bird v3 vs dense on long-context benchmarks (LongBench v2, InfiniteBench Code.Debug,
LongCodeBench LongCodeQA). Tracks accuracy, memory and speed.

Each example is tokenised once and run through every arm in the same process (paired comparison):
  1. prefill the prompt with a KV cache (CUDA-event timed)            -> prefill_s, peak_gb
  2. read the next-token distribution over the letters A-D            -> pred, letter probs (no-CoT answer)
  3. greedy-decode `--decode-tokens` more tokens from the cache       -> decode_ms_per_token
The prompt ends with R1's chat template, an empty think block and the cue "The correct answer is (", so the
answer is the next token (LongBench v2's no-CoT setting, read from logits instead of a regex).

Arms: `dense` is the unmodified HF SDPA model (exp 0 equivalent). Any other arm installs
modal_rd.attention.RDAttention with that arm's JSON config (sparse prefill, exact cached decode).
Default sparse arm `bb_v3` is the best all-sparse config from modal_rd (sink64 + window16K + vertical512 +
exact tail 256). Prompts no longer than sink + window run dense in that arm by construction.

  python -m experiments.exp_20_sota_long_context.run --tasks lbv2,codedebug,lcqa32k,lcqa64k \
      --max-tokens 65536 --limit 50 --tag a100_64k

Results append to <results>/<tag>/<task>.jsonl (one line per example x arm) and runs resume from them.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from experiments.exp_20_sota_long_context.tasks import DATA_DIR, LETTERS, load_task
from modal_rd import attention as rd
from modal_rd.mlp_chunk import install_chunked_mlp
from modal_rd.niah import MODEL_PATH, load_tokenizer

BB_V3 = {"sink": 64, "window": 16384, "vertical": 512, "exact_tail": 256}
# Same prefill as bb_v3 (so the same answer letter); the KV cache is then cut to sink + window + routed keys
# (PR #19 recommended T1 settings), which is what lowers memory and decode cost.
BB_V3_EVICT = {**BB_V3, "kv_evict": True, "evict_pool": 8, "evict_keep": 8192, "evict_window": 16384}
DEFAULT_ARMS = {"dense": None, "bb_v3": BB_V3, "bb_v3_evict": BB_V3_EVICT}
RESULTS = Path(os.environ.get("BB_RESULTS", Path(os.environ.get("SCRATCH", "/vol")) / "results")) / "exp20"
CUE = "The correct answer is ("


class Prompter:
    """R1 chat template around a middle-truncated context, built in token space."""

    def __init__(self, tokenizer, max_tokens):
        self.tok, self.max_tokens = tokenizer, max_tokens
        enc = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]
        self.enc = enc
        self.head = [tokenizer.bos_token_id] + enc("<｜User｜>")
        self.tail = enc("<｜Assistant｜><think>\n\n</think>\n\n" + CUE)
        cue = enc(CUE)
        self.letter_ids = []
        for L in LETTERS:
            ids = enc(CUE + L)
            if ids[:len(cue)] != cue or len(ids) != len(cue) + 1:
                raise RuntimeError(f"letter {L} is not a single token after the cue")
            self.letter_ids.append(ids[-1])

    def build(self, row):
        pre, ctx, suf = self.enc(row["prefix"]), self.enc(row["context"]), self.enc(row["suffix"])
        fixed = len(self.head) + len(pre) + len(suf) + len(self.tail)
        room = self.max_tokens - fixed
        if room <= 0:
            raise ValueError(f"{row['id']}: question alone exceeds max_tokens")
        full = len(ctx)
        if full > room:  # LongBench-style middle truncation: keep the start and end of the context
            ctx = ctx[:room // 2] + ctx[full - (room - room // 2):]
        ids = self.head + pre + ctx + suf + self.tail
        return ids, {"context_tokens": full, "kept_context_tokens": len(ctx), "truncated": full > room}


def build_model(mlp_chunk):
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 low_cpu_mem_usage=True).cuda().eval()
    if mlp_chunk:
        install_chunked_mlp(model, mlp_chunk)
    return model


def set_arm(model, cfg):
    """cfg None -> original HF attention; dict -> RDAttention with that config (projections are shared)."""
    if cfg is None:
        for layer in model.model.layers:
            layer.self_attn = getattr(layer.self_attn, "_rd_base", layer.self_attn)
    else:
        rd.install(model, "vs", **cfg)


@torch.inference_mode()
def run_arm(model, ids, letter_ids, decode_tokens):
    x = torch.tensor([ids], device="cuda")
    cache = DynamicCache(config=model.config)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    out = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
    t1.record()
    torch.cuda.synchronize()
    prefill_s = t0.elapsed_time(t1) / 1000
    prefill_peak = torch.cuda.max_memory_allocated() / 1e9
    logits = out.logits[0, -1].float()
    letter_logp = logits.log_softmax(-1)[letter_ids]
    probs = letter_logp.softmax(-1)  # renormalised over A-D
    pred = LETTERS[int(probs.argmax())]
    nxt = logits.argmax().view(1, 1)
    gen = [int(nxt)]
    decode_ms = None
    if decode_tokens:
        torch.cuda.synchronize()
        s = time.perf_counter()
        for _ in range(decode_tokens):
            o = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            nxt = o.logits[0, -1].argmax().view(1, 1)
            gen.append(int(nxt))
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - s) * 1000 / decode_tokens
    peak = torch.cuda.max_memory_allocated() / 1e9
    del cache, out
    return {"pred": pred, "probs": [round(float(p), 5) for p in probs],
            "letter_mass": round(float(letter_logp.exp().sum()), 5), "top_token": gen[0],
            "gen_ids": gen, "prefill_s": prefill_s, "decode_ms_per_token": decode_ms,
            "prefill_peak_gb": prefill_peak, "peak_gb": peak}


def warmup(model, arms, n):
    """Compile/specialise Triton kernels before timing (two lengths: n % 16 == 0 and != 0)."""
    for name, cfg in arms.items():
        set_arm(model, cfg)
        for m in (n, n + 1):
            run_arm(model, torch.randint(1000, 30000, (m,)).tolist(), [0, 1, 2, 3], 2)
        print(f"WARM arm={name} n={n}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="lbv2,codedebug,lcqa32k,lcqa64k")
    p.add_argument("--arms", default=json.dumps(DEFAULT_ARMS), help='JSON {"name": null | RDAttention config}')
    p.add_argument("--max-tokens", type=int, default=65536, help="prompt budget; longer contexts are middle-truncated")
    p.add_argument("--limit", type=int, default=0, help="examples per task (seeded shuffle; 0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-tokens", type=int, default=0, help="skip prompts shorter than this after truncation")
    p.add_argument("--decode-tokens", type=int, default=16)
    p.add_argument("--mlp-chunk", type=int, default=8192, help="exact sequence-chunked MLP, applied to every arm")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--tag", required=True)
    a = p.parse_args()

    arms = json.loads(a.arms)
    tok = load_tokenizer()
    prompter = Prompter(tok, a.max_tokens)
    model = build_model(a.mlp_chunk)
    out_dir = RESULTS / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"arms": arms, "max_tokens": a.max_tokens, "limit": a.limit, "seed": a.seed, "min_tokens": a.min_tokens,
            "decode_tokens": a.decode_tokens, "mlp_chunk": a.mlp_chunk, "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "model": MODEL_PATH, "cue": CUE}
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        changed = {k for k in ("max_tokens", "mlp_chunk", "seed") if old.get(k) != meta[k]}
        if changed:
            raise SystemExit(f"tag {a.tag} was run with different {changed}; use a new --tag")
        meta["arms"] = {**old.get("arms", {}), **arms}
    meta_path.write_text(json.dumps(meta, indent=2))
    warmup(model, arms, min(a.max_tokens - 64, 20000))

    for task in a.tasks.split(","):
        rows = load_task(task, a.data_dir)
        random.Random(a.seed).shuffle(rows)
        path = out_dir / f"{task}.jsonl"
        done = set()
        if path.exists():
            done = {(r["id"], r["arm"]) for r in map(json.loads, path.read_text().splitlines())}
        n_run = 0
        with open(path, "a") as f:
            for row in rows:
                if a.limit and n_run >= a.limit:
                    break
                ids, info = prompter.build(row)
                if len(ids) < a.min_tokens:
                    continue
                n_run += 1
                for name, cfg in arms.items():
                    if (row["id"], name) in done:
                        continue
                    set_arm(model, cfg)
                    try:
                        res = run_arm(model, ids, prompter.letter_ids, a.decode_tokens)
                    except torch.OutOfMemoryError:
                        res = {"oom": True}
                    if res.get("oom"):  # outside the except block so the traceback's tensors are freed
                        torch.cuda.empty_cache()
                    rec = {"task": task, "id": row["id"], "group": row["group"], "arm": name, "gold": row["gold"],
                           "tokens": len(ids), **info, **{k: row[k] for k in ("difficulty", "length_bucket") if k in row},
                           **res}
                    rec["correct"] = rec.get("pred") == row["gold"]
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"{task} {row['id']} arm={name} tok={len(ids)} gold={row['gold']} pred={rec.get('pred')} "
                          f"p={rec.get('probs')} prefill={rec.get('prefill_s', 0):.2f}s "
                          f"peak={rec.get('peak_gb', 0):.1f}GB oom={rec.get('oom', False)}", flush=True)
    print("DONE", out_dir, flush=True)


if __name__ == "__main__":
    main()
