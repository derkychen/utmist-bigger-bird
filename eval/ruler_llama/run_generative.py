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
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

from eval.ruler.ruler_dataset import build_ruler_dataset
from eval.lra_llama.lra_llama_dataset import _ids_to_text
from patches.llama.llama_patched_model import patch_llama
from patches.llama.tokenizer import load_llama_tokenizer

CONFIG_DIR = Path(__file__).parents[2] / "configs"
RULER_CFG = OmegaConf.load(CONFIG_DIR / "benchmarks" / "ruler.yaml")
TASK_INFO = RULER_CFG.task_info

MODEL_PATH = os.path.join(
    os.environ.get("SCRATCH", "/scratch/$USER"),
    "models", "DeepSeek-R1-Distill-Llama-8B"
)

# Map exp_num to (module_name, attention_class_name, attn_kwargs)
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
         {"budget_mode": "length", "min_budget": 512, "max_budget": 4096, "tokens_per_key": 64,
          "num_route_queries": 1, "num_sinks": 4, "local_window": 256, "use_triton": False}),
    14: ("experiments.exp_14_token_drop_deepseek.model_llama", "TokenDropDeepSeekAttention",
         {"drop_after_layer": 3, "drop_ratio": 0.3, "top_k": 128, "low_rank_dim": 64, "use_triton": False}),
    15: ("experiments.exp_15_bigger_bird.model_llama", "BiggerBirdAttention",
         {"fragment_size": 64, "max_k": 512, "min_k": 64, "globals_per_head": 8,
          "teleports_per_head": 4, "use_teleports": False, "use_triton": False}),
    16: ("experiments.exp_16_nsa_freerider.model_llama", "FreeNSAAttention",
         {"block_size": 64, "topk_blocks": 16, "window_size": 256, "use_triton": False}),
    17: ("experiments.exp_17_coarse_to_fine.model_llama", "CoarseToFineAttention",
         {"block_size": 128, "topk_blocks": 32, "fine_k": 512, "window_size": 256, "use_triton": False}),
    18: ("experiments.exp_18_confidence_gated.model_llama", "ConfidenceGatedAttention",
         {"top_k": 512, "low_rank_dim": 128, "window_size": 256,
          "gate_threshold": 0.5, "peak_threshold": -1.0,
          "linear_weight": 0.5, "use_triton": False, "always_global": True,
          "num_route_queries": 4}),
    19: ("experiments.exp_19_bigger_bird_flash.model_llama", "BiggerBirdFlashAttention",
         {"middle_min": 512, "middle_max": 2048, "middle_ratio": 64,
          "window_min": 1024, "window_max": 8192,
          "local_min": 128, "local_max": 512,
          "globals_min": 32, "globals_max": 64,
          "recent_window": 128, "diversity": 0.05,
          "low_rank_dim": 128, "routing_mode": "last_query",
          "route_chunk": 1024, "use_triton": True}),
}


def build_generative_model(exp_num, model_path=MODEL_PATH, load_in_8bit=False):
    """Load Llama for generative evaluation with patched attention.

    Uses AutoModelForCausalLM (with LM head) instead of AutoModel.
    No LoRA, no classification head — just the pretrained model with
    patched attention for the experiment's sparse variant.

    ``load_in_8bit`` quantizes the linear layers with bitsandbytes so the 8B
    model fits on ~12 GB GPUs. Attention math is unchanged, but results are
    not directly comparable with the bf16 runs.
    """
    if exp_num not in EXP_REGISTRY:
        raise ValueError(f"exp {exp_num} not in registry: {list(EXP_REGISTRY.keys())}")

    module_name, cls_name, attn_kwargs = EXP_REGISTRY[exp_num]
    mod = importlib.import_module(module_name)
    attn_cls = getattr(mod, cls_name)

    print(f"Loading model from {model_path}...")
    quant_kwargs = {}
    if load_in_8bit:
        import logging
        from transformers import BitsAndBytesConfig
        # bf16 activations are cast to fp16 for every int8 matmul; the per-layer notice is noise
        logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)
        quant_kwargs = {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            "device_map": {"": 0},
        }
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **quant_kwargs
    )

    # Patch attention with the experiment's sparse variant before switching to
    # eval mode so newly-created attention modules also receive eval=True.
    print(f"Patching attention with {cls_name} (exp {exp_num})...")
    patch_llama(model, attn_cls, **attn_kwargs)
    model.eval()

    # Set is_causal=True on all attention layers for generative evaluation
    # (Llama was pretrained with causal attention; generative eval needs it)
    inner = getattr(model, "model", model)
    for layer in inner.layers:
        layer.self_attn.is_causal = True

    return model


NUMBER_RE = re.compile(r"\d+(?:,\d+)*")


class FirstNumberComplete(StoppingCriteria):
    """Stop once the first number in the generated text is complete.

    A number may contain digit-group commas ("2,940,341"), so a comma only ends
    it when the next character is not a digit. ``parse_prediction`` only reads
    the first number, so later tokens cannot change the prediction. Without a
    KV cache every extra token re-runs the full prompt, so this saves most of
    the generation time.
    """

    def __init__(self, tokenizer, input_len):
        self.tokenizer = tokenizer
        self.input_len = input_len

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0, self.input_len:], skip_special_tokens=True)
        done = re.search(r"\d(?:[^\d,]|,[^\d])", text) is not None
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


def generate_answer(model, tokenizer, text, max_new_tokens=10, device="cuda", early_stop=True):
    """Generate answer from text prompt.

    Appends ' The answer is:' to the text and generates tokens.
    Returns the generated text (after the prompt).
    """
    prompt = text.rstrip() + " The answer is:"
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
            use_cache=False,  # our custom attention doesn't support KV cache
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=(
                StoppingCriteriaList([FirstNumberComplete(tokenizer, input_len)])
                if early_stop else None
            ),
        )

    new_tokens = outputs[0][input_len:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated


def collect_attention_diagnostics(model):
    """Aggregate optional per-layer sparse-attention diagnostics."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", [])
    bigger_bird = [getattr(layer.self_attn, "last_diagnostics", {}) for layer in layers]
    bigger_bird = [item for item in bigger_bird if item]
    stats = [
        getattr(layer.self_attn, "last_stats", None)
        for layer in layers
        if hasattr(layer, "self_attn")
    ]
    stats = [item for item in stats if item is not None]
    if not stats:
        return {"bigger_bird": {"layers": len(bigger_bird), **bigger_bird[-1]}} if bigger_bird else {}
    total_heads = sum(int(item.total_heads) for item in stats)
    active_heads = sum(int(item.active_heads) for item in stats)
    return {
        "layers": len(stats),
        "active_heads": active_heads,
        "total_heads": total_heads,
        "active_fraction": active_heads / max(1, total_heads),
        "route_k": max(int(item.route_k) for item in stats),
        "margin_mean": sum(float(item.margin_mean) for item in stats) / len(stats),
        "margin_max": max(float(item.margin_max) for item in stats),
        "peakiness_mean": sum(float(item.peakiness_mean) for item in stats) / len(stats),
        "peakiness_max": max(float(item.peakiness_max) for item in stats),
        "used_triton_local": sum(bool(item.used_triton_local) for item in stats),
        "used_triton_linear": sum(bool(item.used_triton_linear) for item in stats),
        "used_triton_global": sum(bool(item.used_triton_global) for item in stats),
        "kernel_failures": sum(int(item.kernel_failures) for item in stats),
    }


def parse_prediction(generated_text, task="niah"):
    """Extract prediction from generated text.

    For niah/mq_niah/qa: extract the first number and return its last digit.
    For vt: extract the first number (single digit).
    For cwe/fwe: extract the first WORD token (WORD0-WORD9).
    """
    # Find all numbers (digit-group commas allowed, e.g. "2,940,341")
    numbers = NUMBER_RE.findall(generated_text)
    if numbers:
        # Take the first number's last digit
        return int(numbers[0].replace(",", "")[-1])
    # Try WORD tokens for cwe/fwe
    word_match = re.search(r'WORD(\d+)', generated_text)
    if word_match:
        return int(word_match.group(1))
    return -1  # No digit found


NIAH_CUE_RE = re.compile(
    r"What (?:is the special magic (?P<one>[\w-]+)|are all values for the special magic "
    r"(?P<multi>[\w-]+)|are the special magics for: (?P<keys>[^?]+))\?"
)


def gold_answers(text):
    """Full gold values for a NIAH prompt, read from its needle sentences.

    The dataset labels keep only a 10-way digit, so the full answer is
    recovered from the prompt text (same rule as modal_rd/niah.py). Returns
    None for non-NIAH tasks, or for prompts whose queried key is stated with
    conflicting values (ambiguous).
    """
    cue = NIAH_CUE_RE.search(text)
    if cue is None:
        return None
    keys = [cue["one"] or cue["multi"]] if not cue["keys"] else [k.strip() for k in cue["keys"].split(",")]
    answers = []
    for key in keys:
        vals = re.findall(r"The special magic " + re.escape(key) + r" is: ([\w-]+)\.", text)
        if not vals or (cue["multi"] is None and len(set(vals)) > 1):
            return None
        answers += vals
    return answers


def exact_match(generated_text, answers):
    """One numeric answer: the first generated number must equal it.
    Otherwise (several values, or uuids): every value must appear in the output."""
    if len(answers) == 1 and answers[0].isdigit():
        numbers = NUMBER_RE.findall(generated_text)
        return bool(numbers) and numbers[0].replace(",", "") == answers[0]
    flat = re.sub(r"\s", "", generated_text)
    return all(a in flat for a in answers)


def evaluate(model, tokenizer, dataset, task, device="cuda", max_examples=None, early_stop=True):
    """Run generative evaluation on a dataset.

    Returns (last-digit accuracy, exact-match accuracy or None, examples).
    """
    correct = 0
    total = 0
    exact_correct = 0
    exact_total = 0
    examples = []

    n = len(dataset) if max_examples is None else min(max_examples, len(dataset))

    for i in range(n):
        row = dataset[i]
        ids = row["input_ids"]
        label = row["labels"]
        text = _ids_to_text(ids)

        generated = generate_answer(model, tokenizer, text, device=device, early_stop=early_stop)
        pred = parse_prediction(generated, task=task)

        is_correct = (pred == label)
        if is_correct:
            correct += 1
        total += 1

        answers = gold_answers(text)
        is_exact = exact_match(generated, answers) if answers else None
        if is_exact is not None:
            exact_correct += is_exact
            exact_total += 1

        if i < 5 or (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n}] label={label} pred={pred} correct={is_correct} "
                  f"answer={answers} exact={is_exact}")
            if i < 5:
                print(f"    generated: {generated[:100]!r}")

        examples.append({
            "idx": i,
            "label": int(label),
            "pred": int(pred),
            "correct": bool(is_correct),
            "answers": answers,
            "exact": is_exact,
            "generated": generated[:200],
        })

    accuracy = correct / total if total > 0 else 0
    exact_accuracy = exact_correct / exact_total if exact_total > 0 else None
    return accuracy, exact_accuracy, examples


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
    parser.add_argument("--attn-kwargs", type=str, default=None,
                        help="JSON string to override attention kwargs in EXP_REGISTRY")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Always generate max_new_tokens instead of stopping after the first number")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Quantize weights to 8-bit (bitsandbytes) for small GPUs; "
                             "results go to benchmarks_local/ instead of benchmarks/")
    args = parser.parse_args()

    # Override attention kwargs if provided
    if args.attn_kwargs:
        import json as _json
        override = _json.loads(args.attn_kwargs)
        module_name, cls_name, default_kwargs = EXP_REGISTRY[args.exp]
        merged = {**default_kwargs, **override}
        EXP_REGISTRY[args.exp] = (module_name, cls_name, merged)
        print(f"Overridden attn_kwargs: {merged}")

    print(f"RULER Generative: {args.task} | exp {args.exp} | seq {args.seq} | depth {args.depth}")
    print(f"{'='*70}\n")

    # --- Tokenizer ---
    if args.exp == 19:
        from experiments.exp_19_bigger_bird_flash.tokenizer import load_checkpoint_tokenizer
        tokenizer = load_checkpoint_tokenizer(MODEL_PATH)
    else:
        tokenizer = load_llama_tokenizer(MODEL_PATH)
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
    model = build_generative_model(args.exp, load_in_8bit=args.load_in_8bit)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.load_in_8bit:  # quantized models are already placed and reject .to()
        model = model.to(device)

    # --- Evaluate ---
    print(f"\nStarting generative evaluation...")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    if args.task in ("niah_multivalue", "niah_multiquery") and not args.no_early_stop:
        print("WARNING: this task needs several numbers, but early stop ends generation after "
              "the first one, so exact accuracy will undercount. Pass --no-early-stop.")
    accuracy, exact_accuracy, examples = evaluate(
        model, tokenizer, eval_ds, args.task, device=device,
        max_examples=args.max_examples, early_stop=not args.no_early_stop,
    )
    n_exact = sum(e["exact"] is not None for e in examples)
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0

    num_labels = TASK_INFO.get(args.task, {}).get("num_labels", 10)
    random_baseline = 1.0 / num_labels

    print(f"\n{'='*70}")
    print(f"Results: {args.task} | exp {args.exp} | seq {args.seq} | depth {args.depth}")
    print(f"  Accuracy: {accuracy:.4f} ({int(accuracy * len(examples))}/{len(examples)})  [last digit]")
    if exact_accuracy is not None:
        print(f"  Exact accuracy: {exact_accuracy:.4f} ({round(exact_accuracy * n_exact)}/{n_exact})  [full answer]")
    else:
        print("  Exact accuracy: n/a (task has no recoverable full answer)")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(examples):.2f}s/example)")
    print(f"  Peak GPU memory: {peak_mem:.2f} GB")
    print(f"  Random baseline: {random_baseline:.4f}")

    # --- Save results ---
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    results = {
        "task": args.task,
        "exp": args.exp,
        "seq_len": args.seq,
        "depth": args.depth,
        "accuracy": accuracy,
        "accuracy_rule": "last digit of first generated number",
        "exact_accuracy": exact_accuracy,
        "n_exact_scored": n_exact,
        "n_examples": len(examples),
        "time_seconds": elapsed,
        "peak_memory_gb": peak_mem,
        "random_baseline": random_baseline,
        "use_cache": False,
        "early_stop": not args.no_early_stop,
        "is_causal": True,
        "model": "DeepSeek-R1-Distill-Llama-8B",
        "quantization": "8bit" if args.load_in_8bit else None,
        "gpu": gpu_name,
        "attention_config": EXP_REGISTRY[args.exp][2],
        "attention_diagnostics": collect_attention_diagnostics(model),
        "examples": examples[:20],
    }

    exp_name = EXP_REGISTRY[args.exp][1].replace("Attention", "")
    # Quantized runs stay out of benchmarks/ so the dashboard only sees bf16 results
    results_root = "benchmarks_local" if args.load_in_8bit else "benchmarks"
    output_dir = f"{results_root}/exp_{args.exp}_{exp_name}_generative_ruler_{args.task}_seq{args.seq}"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
