#!/usr/bin/env python3
"""Does exp 19 route the needle's *number* tokens, or only its key?

Runs one prefill per prompt (no generation) through R1 with exp 19 attention,
then reads each layer's selected route indices (``last_selected_indices``,
one row per head) and reports, per needle, the fraction of heads that keep:

  key     every token of the needle's key name (e.g. "jade-comet")
  number  every token of its 7-digit value
  digit   at least one token of its value

The 128-token exact recent window is always attended in addition to these
routes; needles at depth 0.5 are far outside it.

  python scripts/probe_exp19_needle_routing.py --load-in-8bit
  python scripts/probe_exp19_needle_routing.py --load-in-8bit --cases "niah_multikey_1:5,6" --per-layer
"""
import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.lra_llama.lra_llama_dataset import _ids_to_text
from eval.ruler.ruler_dataset import build_ruler_dataset
from eval.ruler_llama.run_generative import MODEL_PATH, build_generative_model
from experiments.exp_19_bigger_bird_flash.tokenizer import load_checkpoint_tokenizer

NEEDLE_RE = re.compile(r"The special magic ([\w-]+) is: (\d+)\.")
QUESTION_RE = re.compile(r"What is the special magic ([\w-]+)\?")


def token_span(offsets, start, end):
    """Token positions whose character range overlaps [start, end)."""
    return [i for i, (a, b) in enumerate(offsets) if b > a and b > start and a < end]


def coverage(selected, positions):
    """Per-head (all positions kept, any position kept) as float fractions over heads."""
    pos = torch.tensor(positions, device=selected.device)
    hit = (selected.unsqueeze(-1) == pos).any(1)  # [heads, len(pos)]
    return hit.all(-1).float().mean().item(), hit.any(-1).float().mean().item()


def probe(model, tokenizer, text):
    prompt = text.rstrip() + " The answer is:"
    enc = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    question = QUESTION_RE.search(prompt)[1]
    needles = []
    for m in NEEDLE_RE.finditer(prompt):
        needles.append({
            "key": m[1],
            "queried": m[1] == question,
            "key_tokens": token_span(offsets, m.start(1), m.end(1)),
            "number_tokens": token_span(offsets, m.start(2), m.end(2)),
        })
    with torch.no_grad():
        model(**{k: v.to("cuda") for k, v in enc.items()}, use_cache=False)

    layers = model.model.layers
    rows = []
    for needle in needles:
        per_layer = []
        for layer in layers:
            selected = layer.self_attn.last_selected_indices  # [heads, slots], -1 = unused
            key_all, _ = coverage(selected, needle["key_tokens"])
            num_all, num_any = coverage(selected, needle["number_tokens"])
            per_layer.append((key_all, num_all, num_any))
        rows.append((needle, per_layer))
    return enc["input_ids"].shape[1], rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default="niah_multikey_1:0,1,5,6,8 niah_single_2:0,1",
                        help="space-separated task:idx,idx groups")
    parser.add_argument("--seq", type=int, default=32768)
    parser.add_argument("--depth", type=float, default=0.5)
    parser.add_argument("--attn-kwargs", default=None, help="JSON overrides for exp 19 settings")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--per-layer", action="store_true", help="also print every layer for the queried needle")
    args = parser.parse_args()

    if args.attn_kwargs:
        import json
        from eval.ruler_llama import run_generative
        module, cls, defaults = run_generative.EXP_REGISTRY[19]
        run_generative.EXP_REGISTRY[19] = (module, cls, {**defaults, **json.loads(args.attn_kwargs)})

    tokenizer = load_checkpoint_tokenizer(MODEL_PATH)
    model = build_generative_model(19, load_in_8bit=args.load_in_8bit)
    if not args.load_in_8bit:
        model = model.to("cuda")

    for group in args.cases.split():
        task, idxs = group.split(":")
        idxs = [int(i) for i in idxs.split(",")]
        data = build_ruler_dataset(task=task, seq_len=args.seq, needle_depth=args.depth,
                                   train_samples=10, eval_samples=max(128, max(idxs) + 1), seed=42)["validation"]
        print(f"\n=== {task} seq={args.seq} ===")
        print("fraction of heads (mean over 32 layers) keeping:  key / full number / any digit")
        for idx in idxs:
            n_tokens, rows = probe(model, tokenizer, _ids_to_text(data[idx]["input_ids"]))
            print(f"\nidx {idx}  ({n_tokens} tokens)")
            for needle, per_layer in rows:
                mean = [sum(v[j] for v in per_layer) / len(per_layer) for j in range(3)]
                tag = "QUERIED   " if needle["queried"] else "distractor"
                print(f"  {tag} {needle['key']:20s} key {mean[0]:.2f}  number {mean[1]:.2f}  "
                      f"digit {mean[2]:.2f}   (key {len(needle['key_tokens'])} tok, "
                      f"number {len(needle['number_tokens'])} tok)")
                if args.per_layer and needle["queried"]:
                    print("    layer: " + " ".join(f"{i:>4d}" for i in range(len(per_layer))))
                    print("    key  : " + " ".join(f"{v[0]:4.2f}" for v in per_layer))
                    print("    num  : " + " ".join(f"{v[1]:4.2f}" for v in per_layer))


if __name__ == "__main__":
    main()
