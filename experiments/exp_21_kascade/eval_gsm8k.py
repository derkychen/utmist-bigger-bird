"""Phase 2b: GSM8K accuracy of sparse decode with anchor reuse (R1 chain-of-thought).

R1 writes long reasoning traces, so decode dominates. Prompts are short, so the
window is small (default 512) for decode sparsity to apply. Arms:
  dense        exact attention (prefill and decode)
  all          sparse decode, every layer routes
  0,11 / ...   sparse decode, only these anchor layers route; others reuse
Flags after the anchors, e.g. "0,2,8,13,14:remap:pool": "remap" = Kascade head remapping
from the Phase 0 run --map-tag; "pool" = one pick set per KV head (Kascade GQA pooling).
Reports accuracy, output length and KV tokens read per generated token vs dense.

  python -m experiments.exp_21_kascade.eval_gsm8k --sets "dense;all;0,11;0,7,13,23" --n 20 --tag p2b_gsm8k
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset

from experiments.exp_21_kascade import sparse_decode as sd
from experiments.exp_21_kascade.model import load_head_map
from experiments.exp_21_kascade.common import load_model_4bit
from modal_rd import attention as rd
from modal_rd.niah import load_tokenizer  # AutoTokenizer drops spaces for this checkpoint

OUT = Path(__file__).parents[2] / "benchmarks" / "exp_21_kascade" / "phase2b"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def boxed(text):
    """Content of the last \\boxed{...} (balanced braces), or None."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 0, i + len("\\boxed{") - 1
    for j in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i + len("\\boxed{"):j]
    return None


def same_number(pred, gold):
    if pred is None:
        return False
    pred = re.sub(r"\\text\{[^}]*\}|\\\$|\\%|\\!|[,$%\s]", "", pred)
    try:
        return abs(float(pred) - float(gold.replace(",", ""))) < 1e-6
    except ValueError:
        return False


@torch.inference_mode()
def generate(model, ids, max_new, eos):
    """Greedy decode; returns new token ids and the cache length at each decode step."""
    o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    new, lengths = [o.logits[0, -1].argmax().item()], []
    while len(new) < max_new and new[-1] not in eos:
        lengths.append(ids.shape[1] + len(new))
        o = model(input_ids=torch.tensor([[new[-1]]], device="cuda"), past_key_values=o.past_key_values,
                  use_cache=True, logits_to_keep=1)
        new.append(o.logits[0, -1].argmax().item())
    return new, lengths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-new", type=int, default=2048)
    p.add_argument("--attn", default='{"sink":64,"window":512,"vertical":256,"exact_tail":0}')
    p.add_argument("--map-tag", default="p0_16k_tail0_pool")
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    attn = json.loads(a.attn)
    OUT.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer()
    eos = {tok.eos_token_id}
    data = load_dataset("openai/gsm8k", "main", split="test").select(range(a.start, a.start + a.n))
    model = load_model_4bit()
    L = len(model.model.layers)
    report = {"attn": attn, "max_new": a.max_new, "gpu": torch.cuda.get_device_name(0), "quant": "nf4", "arms": []}
    for spec in a.sets.split(";"):
        if spec == "dense":
            anchors = []
            rd.install(model, "dense_rd", **attn)
        else:
            layers, *flags = spec.split(":")
            anchors = list(range(L)) if layers == "all" else [int(x) for x in layers.split(",")]
            hmap = load_head_map(a.map_tag, anchors) if "remap" in flags else None
            sd.install(model, anchors, head_map=hmap, pool="pool" in flags, **attn)
        spread = 1 if spec == "dense" or "pool" in spec else model.config.num_attention_heads // model.config.num_key_value_heads
        runs = []
        for i, ex in enumerate(data):
            gold = ex["answer"].split("####")[-1].strip()
            text = tok.apply_chat_template([{"role": "user", "content": ex["question"] + SUFFIX}],
                                           tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            new, lengths = generate(model, ids, a.max_new, eos)
            torch.cuda.synchronize(); wall = time.perf_counter() - t0
            out = tok.decode(new, skip_special_tokens=True)
            pred = boxed(out)
            reads = [sd.kv_reads(n, L, anchors, attn["sink"], attn["window"], attn["vertical"], spread) for n in lengths]
            sparse_reads = sum(r[0] for r in reads) if spec != "dense" else sum(r[1] for r in reads)
            run = {"i": a.start + i, "gold": gold, "pred": pred, "correct": same_number(pred, gold),
                   "prompt_tokens": ids.shape[1], "new_tokens": len(new), "finished": new[-1] in eos,
                   "kv_read_ratio": sparse_reads / max(1, sum(r[1] for r in reads)), "wall_s": wall}
            runs.append(run)
            print(f"{spec} i={run['i']} correct={run['correct']} pred={pred!r} gold={gold} new={len(new)} "
                  f"reads={run['kv_read_ratio']:.2f} t={wall:.0f}s", flush=True)
        arm = {"spec": spec, "anchors": anchors, "correct": sum(r["correct"] for r in runs), "n": len(runs),
               "mean_new_tokens": sum(r["new_tokens"] for r in runs) / len(runs),
               "unfinished": sum(not r["finished"] for r in runs),
               "kv_read_ratio": sum(r["kv_read_ratio"] for r in runs) / len(runs), "runs": runs}
        report["arms"].append(arm)
        print(f"== gsm8k {spec}: {arm['correct']}/{arm['n']} new_tokens={arm['mean_new_tokens']:.0f} "
              f"unfinished={arm['unfinished']} kv_read_ratio={arm['kv_read_ratio']:.2f}", flush=True)
        (OUT / f"{a.tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
