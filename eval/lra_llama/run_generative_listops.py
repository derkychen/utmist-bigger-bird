#!/usr/bin/env python3
"""Generative evaluation for LRA Listops on R1-Distill-Llama-8B.

Listops is a computation task: given an expression like
  [MAX [MIN 3 5] [SM 2 4 6]]
the model must compute the answer (a single digit 0-9).

Llama was not pretrained on this syntax, so zero-shot won't work.
We use few-shot prompting: provide K example expressions with their
answers, then ask the model to compute a new expression.

Usage:
  python -m eval.lra_llama.run_generative_listops --exp 0 --seq 2048 --shots 5
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.lra.lra_dataset import build_lra_dataset, _listops_tree, _listops_value
from eval.lra_llama.lra_llama_dataset import _ids_to_listops_text
from patches.llama.llama_patched_model import patch_llama

MODEL_PATH = os.path.join(
    os.environ.get("SCRATCH", "/scratch/$USER"),
    "models", "DeepSeek-R1-Distill-Llama-8B"
)

EXP_REGISTRY = {
    0: ("exp_0_baseline.model_llama", "DenseAttention", {}),
    1: ("exp_1_deepseek_topk.model_llama", "DeepSeekTopKAttention",
        {"top_k": 128, "low_rank_dim": 64, "use_triton": False}),
    8: ("exp_8_token_drop.model_llama", "TokenDropAttention",
        {"drop_after_layer": 3, "drop_ratio": 0.3, "use_triton": False}),
    13: ("exp_13_dynamic_context.model_llama", "DynamicContextAttention",
         {"drop_after_layer": 3, "target_budget": 4096, "chunk_size": 8192, "use_triton": False}),
    14: ("exp_14_token_drop_deepseek.model_llama", "TokenDropDeepSeekAttention",
         {"drop_after_layer": 3, "drop_ratio": 0.3, "top_k": 128, "low_rank_dim": 64, "use_triton": False}),
    15: ("exp_15_bigger_bird.model_llama", "BiggerBirdAttention",
         {"fragment_size": 128, "max_k": 64, "min_k": 56, "globals_per_head": 6,
          "teleports_per_head": 4, "use_teleports": False, "use_triton": False}),
}


def build_generative_model(exp_num, model_path=MODEL_PATH):
    """Load Llama for generative evaluation with patched attention."""
    if exp_num not in EXP_REGISTRY:
        raise ValueError(f"exp {exp_num} not in registry: {list(EXP_REGISTRY.keys())}")

    module_name, cls_name, attn_kwargs = EXP_REGISTRY[exp_num]
    mod = importlib.import_module(module_name)
    attn_cls = getattr(mod, cls_name)

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    model.eval()

    print(f"Patching attention with {cls_name} (exp {exp_num})...")
    patch_llama(model, attn_cls, **attn_kwargs)

    # Set is_causal=True for generative evaluation
    inner = getattr(model, "model", model)
    for layer in inner.layers:
        layer.self_attn.is_causal = True

    return model


def make_few_shot_examples(n_shots, seed=12345):
    """Generate n_shots listops examples for few-shot prompting.

    Uses small expressions (depth 2-3) so they fit in context and
    are easy for the model to learn from.
    """
    import random
    rng = random.Random(seed)
    examples = []
    for _ in range(n_shots):
        # Generate a small, clear example
        depth = rng.randint(2, 3)
        toks, val = _listops_tree(rng, max_depth=depth, max_args=4, prob_op=0.7)
        expr = " ".join(toks)
        examples.append((expr, val))
    return examples


def build_prompt(test_expr, few_shot_examples):
    """Build a few-shot prompt for listops evaluation.

    Format:
      Compute the result of each listops expression.
      [MAX 3 5] = 5
      [MIN 2 7 4] = 2
      ...
      [MAX [MIN 3 5] [SM 2 4 6]] =
    """
    lines = ["Compute the result of each listops expression. The operations are:",
             "  [MAX ...] = maximum of the arguments",
             "  [MIN ...] = minimum of the arguments",
             "  [MED ...] = median (floor) of the arguments",
             "  [SM ...] = sum of arguments modulo 10",
             ""]
    for expr, ans in few_shot_examples:
        lines.append(f"{expr} = {ans}")
    lines.append(f"{test_expr} =")
    return "\n".join(lines)


def generate_answer(model, tokenizer, prompt, max_new_tokens=5, device="cuda"):
    """Generate answer from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][input_len:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated


def parse_prediction(generated_text):
    """Extract the first digit from generated text."""
    # Look for a digit
    match = re.search(r'\d', generated_text)
    if match:
        return int(match.group())
    return -1


def evaluate(model, tokenizer, dataset, few_shot_examples, device="cuda", max_examples=None):
    """Run generative evaluation on listops dataset."""
    correct = 0
    total = 0
    examples = []

    n = len(dataset) if max_examples is None else min(max_examples, len(dataset))

    for i in range(n):
        row = dataset[i]
        ids = row["input_ids"]
        label = int(row["labels"])
        test_expr = _ids_to_listops_text(ids)

        prompt = build_prompt(test_expr, few_shot_examples)
        generated = generate_answer(model, tokenizer, prompt, device=device)
        pred = parse_prediction(generated)

        is_correct = (pred == label)
        if is_correct:
            correct += 1
        total += 1

        if i < 5 or (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n}] label={label} pred={pred} correct={is_correct}")
            if i < 5:
                print(f"    expr: {test_expr[:80]}...")
                print(f"    generated: {generated[:50]!r}")

        examples.append({
            "idx": i,
            "label": label,
            "pred": pred,
            "correct": bool(is_correct),
            "generated": generated[:100],
            "expr": test_expr[:200],
        })

    accuracy = correct / total if total > 0 else 0
    return accuracy, examples


def main():
    parser = argparse.ArgumentParser(
        description="Generative LRA Listops evaluation on R1-Llama-8B (few-shot)",
    )
    parser.add_argument("--exp", type=int, default=0, help="Experiment number")
    parser.add_argument("--seq", type=int, default=2048, help="Sequence length")
    parser.add_argument("--shots", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--eval-samples", type=int, default=128, help="Number of eval samples")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit examples")
    args = parser.parse_args()

    print(f"LRA Listops Generative: exp {args.exp} | seq {args.seq} | shots {args.shots}")
    print(f"{'='*70}\n")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- Build dataset ---
    print("Building dataset...")
    data = build_lra_dataset(
        task="listops",
        seq_len=args.seq,
        train_samples=10,
        eval_samples=args.eval_samples,
        seed=42,
    )
    eval_ds = data["validation"]
    print(f"Eval: {len(eval_ds)} samples")

    # --- Few-shot examples ---
    few_shot = make_few_shot_examples(args.shots)
    print(f"Few-shot examples ({args.shots}):")
    for expr, ans in few_shot:
        print(f"  {expr} = {ans}")
    print()

    # --- Build model ---
    model = build_generative_model(args.exp)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # --- Evaluate ---
    print(f"Starting generative evaluation...")
    t0 = time.time()
    accuracy, examples = evaluate(
        model, tokenizer, eval_ds, few_shot, device=device,
        max_examples=args.max_examples,
    )
    elapsed = time.time() - t0

    num_labels = 10
    random_baseline = 1.0 / num_labels

    print(f"\n{'='*70}")
    print(f"Results: listops | exp {args.exp} | seq {args.seq} | shots {args.shots}")
    print(f"  Accuracy: {accuracy:.4f} ({int(accuracy * len(examples))}/{len(examples)})")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(examples):.2f}s/example)")
    print(f"  Random baseline: {random_baseline:.4f}")

    # --- Save results ---
    results = {
        "task": "listops",
        "exp": args.exp,
        "seq_len": args.seq,
        "shots": args.shots,
        "accuracy": accuracy,
        "n_examples": len(examples),
        "time_seconds": elapsed,
        "random_baseline": random_baseline,
        "examples": examples[:20],
    }

    exp_name = EXP_REGISTRY[args.exp][1].replace("Attention", "")
    output_dir = f"benchmarks/exp_{args.exp}_{exp_name}_generative_listops_seq{args.seq}"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
