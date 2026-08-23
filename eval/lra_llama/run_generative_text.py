#!/usr/bin/env python3
"""Generative evaluation for LRA Text (IMDb sentiment) on R1-Distill-Llama-8B.

Instead of training a classification head, this uses the model's native
generative capability: the model reads the movie review and generates
"positive" or "negative".

Zero-shot: no training needed. The model uses its pretrained ability
to classify sentiment from text.

The LRA text task uses byte-level IMDb reviews. We convert the byte ids
back to UTF-8 text, format a prompt, and parse the generated answer.

Usage:
  python -m eval.lra_llama.run_generative_text --exp 0 --seq 4096
  python -m eval.lra_llama.run_generative_text --exp 0 --seq 4096 --max-examples 30
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.lra.lra_dataset import build_lra_dataset
from eval.lra_llama.lra_llama_dataset import _ids_to_text
from patches.llama.llama_patched_model import patch_llama

CONFIG_DIR = Path(__file__).parents[2] / "configs"
LRA_CFG = OmegaConf.load(CONFIG_DIR / "benchmarks" / "lra.yaml")
TASK_INFO = LRA_CFG.task_info

MODEL_PATH = os.path.join(
    os.environ.get("SCRATCH", "/scratch/$USER"),
    "models", "DeepSeek-R1-Distill-Llama-8B"
)

# Same EXP_REGISTRY as RULER generative — shared across all eval tracks
EXP_REGISTRY = {
    0: ("experiments.exp_0_baseline.model_llama", "DenseAttention", {}),
    1: ("experiments.exp_1_deepseek_topk.model_llama", "DeepSeekTopKAttention",
        {"top_k": 128, "low_rank_dim": 64, "use_triton": False}),
    2: ("experiments.exp_2_lightning_hybrid.model_llama", "LightningHybridAttention",
        {"block_size": 128, "use_triton": False}),
    3: ("experiments.exp_3_dynamic_globals.model_llama", "DynamicGlobalAttention",
        {"window_size": 64, "num_globals": 16, "use_triton": False}),
    4: ("experiments.exp_4_pbs_attn.model_llama", "PBSAttention",
        {"block_size": 64, "num_blocks": 2, "use_triton": False}),
    5: ("experiments.exp_5_bigger_bird.model_llama", "BiggerBirdAttention",
        {"window_size": 64, "local_k": 32, "num_globals": 16, "use_triton": False}),
    6: ("experiments.exp_6_deepseek_pbs.model_llama", "DeepSeekPBSAttention",
        {"top_k": 64, "low_rank_dim": 16, "block_size": 32, "use_triton": False}),
    7: ("experiments.exp_7_layer_adaptive.model_llama", "LayerAdaptiveAttention",
        {"k_early": 192, "k_mid": 64, "k_late": 32, "use_triton": False}),
    8: ("experiments.exp_8_token_drop.model_llama", "TokenDropAttention",
        {"drop_after_layer": 3, "drop_ratio": 0.3, "use_triton": False}),
    9: ("experiments.exp_9_attn_specul.model_llama", "AttnSpeculAttention",
        {"window_size": 64, "num_anchors": 4, "verify_every": 4, "use_triton": False}),
    10: ("experiments.exp_10_gqa_sparse.model_llama", "GQASparseLlamaAttention",
         {"top_k": 64, "low_rank_dim": 16, "use_triton": False}),
    11: ("experiments.exp_11_nsa.model_llama", "NSAAttention",
         {"block_size": 32, "stride": 32, "topk_blocks": 4, "use_triton": False}),
    12: ("experiments.exp_12_s2_hhst.model_llama", "S2HHSTAttention",
         {"shard_size": 32, "local_blocks": 2, "use_triton": False}),
    13: ("experiments.exp_13_dynamic_context.model_llama", "DynamicContextAttention",
         {"drop_after_layer": 3, "target_budget": 4096, "chunk_size": 8192, "use_triton": False}),
    14: ("experiments.exp_14_token_drop_deepseek.model_llama", "TokenDropDeepSeekAttention",
         {"drop_after_layer": 3, "drop_ratio": 0.3, "top_k": 128, "low_rank_dim": 64, "use_triton": False}),
    15: ("experiments.exp_15_bigger_bird.model_llama", "BiggerBirdAttention",
         {"fragment_size": 64, "max_k": 512, "min_k": 64, "globals_per_head": 8,
          "teleports_per_head": 4, "use_teleports": False, "use_triton": False}),
    18: ("experiments.exp_18_confidence_gated.model_llama", "ConfidenceGatedAttention",
         {"top_k": 512, "low_rank_dim": 128, "window_size": 256,
          "gate_threshold": 0.5, "peak_threshold": -1.0,
          "linear_weight": 0.5, "use_triton": False, "always_global": True,
          "num_route_queries": 4}),
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
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    print(f"Patching attention with {cls_name} (exp {exp_num})...")
    patch_llama(model, attn_cls, **attn_kwargs)
    model.eval()

    inner = getattr(model, "model", model)
    for layer in inner.layers:
        layer.self_attn.is_causal = True

    return model


def generate_answer(model, tokenizer, text, max_new_tokens=5, device="cuda"):
    """Generate sentiment answer from a movie review prompt.

    Returns the generated text (after the prompt).
    """
    prompt = (
        "Read the following movie review and classify its sentiment "
        "as either positive or negative. Reply with a single word.\n\n"
        f"Review: {text.rstrip()}\n\n"
        "Sentiment:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=131072)
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
    """Parse sentiment from generated text.

    Returns 1 (positive) or 0 (negative). Returns -1 if unclear.
    """
    text = generated_text.lower().strip()
    # Check which word appears first
    pos_idx = text.find("positive")
    neg_idx = text.find("negative")
    if pos_idx == -1 and neg_idx == -1:
        # Try partial matches
        if "pos" in text:
            return 1
        if "neg" in text:
            return 0
        return -1
    if pos_idx == -1:
        return 0  # only "negative" found
    if neg_idx == -1:
        return 1  # only "positive" found
    # Both found — return whichever comes first
    return 1 if pos_idx < neg_idx else 0


def evaluate(model, tokenizer, dataset, device="cuda", max_examples=None):
    """Run generative sentiment evaluation on the dataset."""
    correct = 0
    total = 0
    examples = []

    n = len(dataset) if max_examples is None else min(max_examples, len(dataset))

    for i in range(n):
        row = dataset[i]
        ids = row["input_ids"]
        label = int(row["labels"])
        text = _ids_to_text(ids)

        generated = generate_answer(model, tokenizer, text, device=device)
        pred = parse_prediction(generated)

        is_correct = (pred == label)
        if is_correct:
            correct += 1
        total += 1

        if i < 5 or (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] label={label} pred={pred} correct={is_correct}")
            if i < 5:
                print(f"    generated: {generated[:100]!r}")

        examples.append({
            "idx": i,
            "label": label,
            "pred": int(pred) if pred >= 0 else -1,
            "correct": bool(is_correct),
            "generated": generated[:200],
        })

    accuracy = correct / total if total > 0 else 0
    return accuracy, examples


def main():
    parser = argparse.ArgumentParser(
        description="Generative LRA Text (IMDb sentiment) evaluation on R1-Llama-8B (zero-shot)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", default="text", help="LRA task (default: text)")
    parser.add_argument("--exp", type=int, default=0, help="Experiment number")
    parser.add_argument("--seq", type=int, default=4096, help="Context window (byte-level)")
    parser.add_argument("--eval-samples", type=int, default=128, help="Number of eval samples")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit examples")
    args = parser.parse_args()

    print(f"LRA Text Generative: {args.task} | exp {args.exp} | seq {args.seq}")
    print(f"{'='*70}\n")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- Build dataset ---
    print("Building dataset...")
    data = build_lra_dataset(
        task=args.task,
        seq_len=args.seq,
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
        model, tokenizer, eval_ds, device=device,
        max_examples=args.max_examples,
    )
    elapsed = time.time() - t0

    num_labels = TASK_INFO.get(args.task, {}).get("num_labels", 2)
    random_baseline = 1.0 / num_labels

    print(f"\n{'='*70}")
    print(f"Accuracy: {accuracy:.4f} ({int(accuracy * len(examples))}/{len(examples)})")
    print(f"Random baseline: {random_baseline:.4f}")
    print(f"Time: {elapsed:.1f}s")

    # --- Save results ---
    results = {
        "task": args.task,
        "exp": args.exp,
        "seq_len": args.seq,
        "depth": None,
        "accuracy": accuracy,
        "n_examples": len(examples),
        "time_seconds": elapsed,
        "random_baseline": random_baseline,
        "examples": examples[:20],
    }

    exp_name = EXP_REGISTRY[args.exp][1].replace("Attention", "")
    output_dir = f"benchmarks/exp_{args.exp}_{exp_name}_generative_lra_{args.task}_seq{args.seq}"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
