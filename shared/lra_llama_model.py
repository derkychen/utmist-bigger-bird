"""Llama-based (R1-Distill-Llama-8B) model builder for LRA/RULER eval tracks.

Replaces the from-scratch BART encoder with R1-Distill-Llama-8B + LoRA, so the
long-context eval tracks benefit from the same pretrained backbone as the IMDb
experiments. The LRA/RULER datasets are converted from byte-level ids to text
strings and tokenized with the Llama tokenizer.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.llama_patched_model import LlamaPatchedModel, apply_lora

# Experiment registry for Llama — maps exp_num to (module_name, exp_name)
LLAMA_EXPERIMENTS = {
    0:  ("exp_0_baseline.model_llama",            "exp_0_baseline"),
    1:  ("exp_1_deepseek_topk.model_llama",       "exp_1_deepseek_topk"),
    8:  ("exp_8_token_drop.model_llama",          "exp_8_token_drop"),
    13: ("exp_13_dynamic_context.model_llama",    "exp_13_dynamic_context"),
    14: ("exp_14_token_drop_deepseek.model_llama", "exp_14_token_drop_deepseek"),
    15: ("exp_15_bigger_bird.model_llama",        "exp_15_bigger_bird"),
}

MODEL_PATH = os.path.join(
    os.environ.get("SCRATCH", "/scratch/$USER"),
    "models", "DeepSeek-R1-Distill-Llama-8B"
)


def build_lra_llama_model(exp_num, num_labels=2, lora_r=16, lora_alpha=32):
    """Build a R1-Llama-8B model with the given experiment's sparse attention + LoRA."""
    import importlib
    if exp_num not in LLAMA_EXPERIMENTS:
        raise ValueError(f"exp {exp_num} not in Llama registry: {list(LLAMA_EXPERIMENTS.keys())}")
    module_name, exp_name = LLAMA_EXPERIMENTS[exp_num]
    mod = importlib.import_module(module_name)
    model = mod.build_model(
        model_path=MODEL_PATH,
        num_labels=num_labels,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )
    meta = {"base_model": "r1-distill-llama-8b", "exp_num": exp_num, "exp_name": exp_name}
    return model, exp_name, meta
