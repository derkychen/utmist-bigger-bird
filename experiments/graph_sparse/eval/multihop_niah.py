"""Multi-hop needle-in-a-haystack benchmark for evaluating traveling paths.

Creates synthetic prompts where answering the question requires following a
chain of references across the context:

  Clue 1 (at depth d1): "The code for item A is 482913."
  Clue 2 (at depth d2): "To find item A, look for its code."
  Question (at end):     "What is the code? The answer is:"

The model must first locate the question, then the reference clue, then the
target clue — a multi-hop chain. Standard single-needle NIAH does not test
this because the needle is self-contained.

Chain lengths: 2 (single reference) and 3 (double reference).

Usage:
    python -m experiments.graph_sparse.eval.multihop_niah --hops 2 --bytes 128000 --n 10

Or via Modal:
    modal run modal_rd/app.py --module experiments.graph_sparse.eval.multihop_niah \
        --args "--hops 2 --bytes 128000 --n 10 --tag multihop_2hop_128k"
"""
import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path


FILLER_SEED = "The grass is green and the sky is blue and the sun shines bright. "
ITEM_NAMES = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
]


def _make_filler(length):
    """Generate deterministic filler text of approximately `length` characters."""
    repeats = max(1, length // len(FILLER_SEED))
    return (FILLER_SEED * repeats)[:length]


def build_multihop_prompt(rng, total_chars, hops=2, depth_spread=0.3):
    """Build one multi-hop NIAH prompt.

    Args:
        rng: random.Random instance.
        total_chars: approximate total prompt length in characters.
        hops: 2 (one reference) or 3 (two references).
        depth_spread: how far apart clues are placed (fraction of total).

    Returns:
        dict with keys: text, answer, hops, clue_depths, item_name.
    """
    item = rng.choice(ITEM_NAMES)
    answer = str(rng.randint(100000, 999999))

    if hops == 2:
        clue_target = f"One of the special magic numbers for {item} is {answer}."
        clue_ref = f"To find the magic number, look for the entry about {item}."
        clues = [clue_target, clue_ref]
    elif hops == 3:
        codename = rng.choice([n for n in ITEM_NAMES if n != item])
        clue_target = f"One of the special magic numbers for {item} is {answer}."
        clue_bridge = f"The entry for {codename} contains the number for {item}. Look for {codename}."
        clue_ref = f"To find the magic number, look for the entry about {codename}."
        clues = [clue_target, clue_bridge, clue_ref]
    else:
        raise ValueError(f"hops must be 2 or 3, got {hops}")

    question = f" What is the magic number? The answer is:"
    clue_total_len = sum(len(c) for c in clues) + len(question)
    filler_len = max(100, total_chars - clue_total_len)

    depths = sorted(rng.sample(
        [i / 10.0 for i in range(1, 10) if abs(i / 10.0 - 0.5) < depth_spread + 0.2],
        k=len(clues),
    ))

    parts = []
    last_pos = 0
    for i, (clue, depth) in enumerate(zip(clues, depths)):
        insert_pos = int(depth * filler_len)
        parts.append(_make_filler(insert_pos - last_pos))
        parts.append(" " + clue + " ")
        last_pos = insert_pos

    parts.append(_make_filler(filler_len - last_pos))
    parts.append(question)
    text = "".join(parts)

    return {
        "text": text,
        "answer": answer,
        "hops": hops,
        "clue_depths": depths,
        "item_name": item,
    }


def prepare_multihop(tokenizer, n_bytes, n_examples, hops=2, seed=42):
    """Build a set of multi-hop NIAH prompts.

    Args:
        tokenizer: HF tokenizer.
        n_bytes: approximate prompt length in characters.
        n_examples: number of examples to generate.
        hops: 2 or 3.
        seed: random seed.

    Returns:
        List of dicts with: text, answer, hops, tokens, token_sha256, idx.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(n_examples):
        row = build_multihop_prompt(rng, n_bytes, hops=hops)
        ids = tokenizer(row["text"], add_special_tokens=True, truncation=False)["input_ids"]
        row["tokens"] = len(ids)
        row["token_sha256"] = hashlib.sha256(
            json.dumps(ids).encode()).hexdigest()[:16]
        row["idx"] = i
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hops", type=int, default=2, choices=[2, 3])
    p.add_argument("--bytes", type=int, default=128000)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="multihop_test")
    p.add_argument("--dry-run", action="store_true",
                    help="Print prompts without running model.")
    a = p.parse_args()

    if a.dry_run:
        from transformers import AutoTokenizer
        model_path = os.path.join(
            os.environ.get("SCRATCH", "/vol"), "models",
            "DeepSeek-R1-Distill-Llama-8B")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        rows = prepare_multihop(tokenizer, a.bytes, a.n, hops=a.hops, seed=a.seed)
        for r in rows:
            print(f"idx={r['idx']} tokens={r['tokens']} answer={r['answer']} "
                  f"hops={r['hops']} depths={r['clue_depths']}")
            print(f"  text[:200] = {r['text'][:200]!r}")
            print(f"  text[-100:] = {r['text'][-100:]!r}")
        print(f"\n{len(rows)} prompts, mean tokens: "
              f"{sum(r['tokens'] for r in rows) / len(rows):.0f}")
        return

    from modal_rd.niah import load_tokenizer
    from modal_rd.eval_niah import build, run_one
    import torch

    tokenizer = load_tokenizer()
    rows = prepare_multihop(tokenizer, a.bytes, a.n, hops=a.hops, seed=a.seed)

    results_dir = Path(os.environ.get("BB_RESULTS",
                       Path(os.environ.get("SCRATCH", "/vol")) / "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{a.tag}.json"

    report = {
        "task": f"multihop_{a.hops}hop",
        "hops": a.hops,
        "bytes": a.bytes,
        "n": a.n,
        "seed": a.seed,
        "runs": [],
    }

    print(f"Multi-hop NIAH: {a.hops} hops, {a.bytes} bytes, {a.n} examples")
    for r in rows:
        print(f"  idx={r['idx']} tokens={r['tokens']} answer={r['answer']}")

    out_path.write_text(json.dumps(report, indent=2))
    print(f"Prompt manifest written to {out_path}")
    print("Run eval with run_eval.py passing this task.")


if __name__ == "__main__":
    main()
