#!/usr/bin/env python3
"""Generative evaluation for RULER tasks on R1-Distill-Llama-8B.

Instead of training a classification head, this uses the model's native
generative capability: the model reads the context + query and generates
the answer. This is how RULER was designed to be evaluated.

For niah tasks, the model generates the full number (e.g., "1234567")
and we extract the last digit as the prediction (matching the label).

Zero-shot: no training needed. The model uses its pretrained ability
to retrieve information from context.

Usage:
  python -m eval.ruler_llama.run_generative --task niah --exp 0 --seq 4096
  python -m eval.ruler_llama.run_generative --task niah --exp 0 --seq 4096 --max-examples 20
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.ruler.ruler_dataset import build_ruler_dataset
from eval.lra_llama.lra_llama_dataset import _ids_to_text
from patches.llama.llama_patched_model import patch_llama

CONFIG_DIR = Path(__file__).parents[2] / "configs"
RULER_CFG = OmegaConf.load(CONFIG_DIR / "benchmarks" / "ruler.yaml")
TASK_INFO = RULER_CFG.task_info

MODEL_PATH = os.path.join(
    os.environ.get("SCRATCH", "/scratch/$USER"),
    "models", "DeepSeek-R1-Distill-Llama-8B"
)

# Map exp_num to (module_name, attention_class_name, attn_kwargs)
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
         {"fragment_size": 64, "max_k": 512, "min_k": 64, "globals_per_head": 8,
          "teleports_per_head": 4, "use_teleports": False, "use_triton": False}),
}


def build_generative_model(exp_num, model_path=MODEL_PATH):
    """Load Llama for generative evaluation with patched attention.

    Uses AutoModelForCausalLM (with LM head) instead of AutoModel.
    No LoRA, no classification head — just the pretrained model with
    patched attention for the experiment's sparse variant.
    """
    if exp_num not in EXP_REGISTRY:
        raise ValueError(f"exp {exp_num} not in registry: {list(EXP_REGISTRY.keys())}")

    module_name, cls_name, attn_kwargs = EXP_REGISTRY[exp_num]
    mod = importlib.import_module(module_name)
    attn_cls = getattr(mod, cls_name)

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    # Patch attention with the experiment's sparse variant
    print(f"Patching attention with {cls_name} (exp {exp_num})...")
    patch_llama(model, attn_cls, **attn_kwargs)

    # Set is_causal=True on all attention layers for generative evaluation
    # (Llama was pretrained with causal attention; generative eval needs it)
    inner = getattr(model, "model", model)
    for layer in inner.layers:
        layer.self_attn.is_causal = True

    return model


def generate_answer(model, tokenizer, text, max_new_tokens=10, device="cuda"):
    """Generate answer from text prompt.

    Appends ' The answer is:' to the text and generates tokens.
    Returns the generated text (after the prompt).
    """
    prompt = text.rstrip() + " The answer is:"
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
            use_cache=False,  # our custom attention doesn't support KV cache
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][input_len:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated


def parse_prediction(generated_text, task="niah"):
    """Extract prediction from generated text.

    For niah/mq_niah/qa: extract the first number and return its last digit.
    For vt: extract the first number (single digit).
    For cwe/fwe: extract the first WORD token (WORD0-WORD9).
    """
    # Find all digit sequences
    numbers = re.findall(r'\d+', generated_text)
    if numbers:
        # Take the first number's last digit
        return int(numbers[0][-1])
    # Try WORD tokens for cwe/fwe
    word_match = re.search(r'WORD(\d+)', generated_text)
    if word_match:
        return int(word_match.group(1))
    return -1  # No digit found


def evaluate(model, tokenizer, dataset, task, device="cuda", max_examples=None):
    """Run generative evaluation on a dataset."""
    correct = 0
    total = 0
    examples = []

    n = len(dataset) if max_examples is None else min(max_examples, len(dataset))

    for i in range(n):
        row = dataset[i]
        ids = row["input_ids"]
        label = row["labels"]
        text = _ids_to_text(ids)

        generated = generate_answer(model, tokenizer, text, device=device)
        pred = parse_prediction(generated, task=task)

        is_correct = (pred == label)
        if is_correct:
            correct += 1
        total += 1

        if i < 5 or (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n}] label={label} pred={pred} correct={is_correct}")
            if i < 5:
                print(f"    generated: {generated[:100]!r}")

        examples.append({
            "idx": i,
            "label": int(label),
            "pred": int(pred),
            "correct": bool(is_correct),
            "generated": generated[:200],
        })

    accuracy = correct / total if total > 0 else 0
    return accuracy, examples


def main():
    parser = argparse.ArgumentParser(
        description="Generative RULER evaluation on R1-Llama-8B (zero-shot)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", default="niah", help="RULER task name")
    parser.add_argument("--exp", type=int, default=0, help="Experiment number")
    parser.add_argument("--seq", type=int, default=4096, help="Context window")
    parser.add_argument("--depth", type=float, default=0.5, help="Needle depth")
    parser.add_argument("--eval-samples", type=int, default=128, help="Number of eval samples")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit examples (for testing)")
    args = parser.parse_args()

    print(f"RULER Generative: {args.task} | exp {args.exp} | seq {args.seq} | depth {args.depth}")
    print(f"{'='*70}\n")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- Build dataset ---
    print("Building dataset...")
    data = build_ruler_dataset(
        task=args.task,
        seq_len=args.seq,
        needle_depth=args.depth,
        train_samples=10,  # not used for generative eval
        eval_samples=args.eval_samples,
        seed=42,
    )
    eval_ds = data["validation"]
    print(f"Eval: {len(eval_ds)} samples")

    # --- Build model ---
    model = build_generative_model(args.exp)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # --- Evaluate ---
    print(f"\nStarting generative evaluation...")
    t0 = time.time()
    accuracy, examples = evaluate(
        model, tokenizer, eval_ds, args.task, device=device,
        max_examples=args.max_examples,
    )
    elapsed = time.time() - t0

    num_labels = TASK_INFO.get(args.task, {}).get("num_labels", 10)
    random_baseline = 1.0 / num_labels

    print(f"\n{'='*70}")
    print(f"Results: {args.task} | exp {args.exp} | seq {args.seq} | depth {args.depth}")
    print(f"  Accuracy: {accuracy:.4f} ({int(accuracy * len(examples))}/{len(examples)})")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(examples):.2f}s/example)")
    print(f"  Random baseline: {random_baseline:.4f}")

    # --- Save results ---
    results = {
        "task": args.task,
        "exp": args.exp,
        "seq_len": args.seq,
        "depth": args.depth,
        "accuracy": accuracy,
        "n_examples": len(examples),
        "time_seconds": elapsed,
        "random_baseline": random_baseline,
        "examples": examples[:20],
    }

    exp_name = EXP_REGISTRY[args.exp][1].replace("Attention", "")
    output_dir = f"benchmarks/exp_{args.exp}_{exp_name}_generative_ruler_{args.task}_seq{args.seq}"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
