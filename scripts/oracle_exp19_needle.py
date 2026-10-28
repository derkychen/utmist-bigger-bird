#!/usr/bin/env python3
"""Oracle diagnostic: does exp 19 answer correctly if the queried needle is forced into every head?

Answer-aware and never a benchmark result. After exp 19's normal selection,
the 128 tokens centred on the queried needle statement are forced into every
head's route set of the chosen layers (``force_diagnostic_span``; the route
budget stays the same, so some ordinary routes are dropped). If the model
still fails, the failure is not caused by routing missing the needle.

  python scripts/oracle_exp19_needle.py --load-in-8bit --oracle all
  python scripts/oracle_exp19_needle.py --load-in-8bit --oracle none   # same prompts, no forcing
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.lra_llama.lra_llama_dataset import _ids_to_text
from eval.ruler.ruler_dataset import build_ruler_dataset
from eval.ruler_llama.run_generative import (MODEL_PATH, build_generative_model, exact_match,
                                             generate_answer, gold_answers)
from experiments.exp_19_bigger_bird_flash import model_llama as exp19
from experiments.exp_19_bigger_bird_flash.tokenizer import load_checkpoint_tokenizer
from kernels.bigger_bird_query_routing import force_diagnostic_span

QUESTION_RE = re.compile(r"What is the special magic ([\w-]+)\?")

# Set per prompt; read by the patched selector for layers in ORACLE["layers"].
ORACLE = {"span": None, "layers": frozenset(), "active": False}
_select = exp19.select_bigger_bird


def _select_with_oracle(q, k, **kwargs):
    indices, chunk = _select(q, k, **kwargs)
    if ORACLE["active"] and ORACLE["span"] is not None:
        indices = force_diagnostic_span(indices, ORACLE["span"], k.shape[1])
    return indices, chunk


exp19.select_bigger_bird = _select_with_oracle


class OracleFlashAttention(exp19.BiggerBirdFlashAttention):
    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        ORACLE["active"] = self.layer_idx in ORACLE["layers"]
        try:
            return super().sparse_attention(Q, K, V, token_mask, bsz, num_heads, is_causal)
        finally:
            ORACLE["active"] = False


def needle_span(tokenizer, prompt, width):
    """`width` tokens centred on the queried needle statement."""
    key = QUESTION_RE.search(prompt)[1]
    m = re.search(r"The special magic " + re.escape(key) + r" is: [\w-]+\.", prompt)
    offsets = tokenizer(prompt, return_offsets_mapping=True)["offset_mapping"]
    toks = [i for i, (a, b) in enumerate(offsets) if b > a and b > m.start() and a < m.end()]
    centre = (toks[0] + toks[-1]) // 2
    begin = max(0, centre - width // 2)
    return begin, min(len(offsets), begin + width)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="niah_multikey_1")
    parser.add_argument("--seq", type=int, default=32768)
    parser.add_argument("--depth", type=float, default=0.5)
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--oracle", choices=["none", "all", "late8"], default="all",
                        help="layers that get the forced needle span (late8 = layers 24-31)")
    parser.add_argument("--span", type=int, default=128, help="forced tokens around the needle")
    parser.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()

    ORACLE["layers"] = {"none": frozenset(), "all": frozenset(range(32)),
                        "late8": frozenset(range(24, 32))}[args.oracle]

    tokenizer = load_checkpoint_tokenizer(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_generative_model(19, load_in_8bit=args.load_in_8bit)
    if not args.load_in_8bit:
        model = model.to("cuda")
    for layer in model.model.layers:
        layer.self_attn.__class__ = OracleFlashAttention

    data = build_ruler_dataset(task=args.task, seq_len=args.seq, needle_depth=args.depth,
                               train_samples=10, eval_samples=128, seed=42)["validation"]
    examples, exact = [], 0
    t0 = time.time()
    for i in range(args.max_examples):
        text = _ids_to_text(data[i]["input_ids"])
        prompt = text.rstrip() + " The answer is:"  # what generate_answer feeds the model
        answers = gold_answers(text)
        ORACLE["span"] = needle_span(tokenizer, prompt, args.span)
        generated = generate_answer(model, tokenizer, text)
        ok = exact_match(generated, answers)
        exact += ok
        examples.append({"idx": i, "answers": answers, "exact": ok, "span": ORACLE["span"],
                         "generated": generated[:200]})
        print(f"  [{i+1}/{args.max_examples}] gold={answers} exact={ok} generated={generated[:40]!r}", flush=True)

    acc = exact / len(examples)
    print(f"\nOracle={args.oracle} task={args.task} seq={args.seq}: exact {acc:.4f} ({exact}/{len(examples)}) "
          f"in {time.time() - t0:.0f}s")

    out_dir = f"benchmarks_local/oracle_exp19_{args.task}_seq{args.seq}"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"oracle_{args.oracle}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump({"task": args.task, "seq_len": args.seq, "depth": args.depth, "oracle": args.oracle,
                   "oracle_layers": sorted(ORACLE["layers"]), "span_tokens": args.span,
                   "exact_accuracy": acc, "n_examples": len(examples),
                   "quantization": "8bit" if args.load_in_8bit else None,
                   "attention_config": dict(model.model.layers[0].self_attn.schedule_kwargs,
                                            recent_window=model.model.layers[0].self_attn.recent_window),
                   "note": "answer-aware diagnostic, not a benchmark result",
                   "examples": examples}, f, indent=2)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
