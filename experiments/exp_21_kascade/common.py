"""Model loading, output paths and greedy decoding for exp 21 (local RTX 3080 or Modal H100).

Locally (RTX 3080, 10 GB) the model is loaded in 4-bit NF4 so it fits. Three things keep
~16K-token prefill under the limit:
  * the MLP runs in 4096-token chunks (identical output, smaller activation peak);
  * logits are computed for the last position only;
  * the KV cache is offloaded to CPU between layers.
Under WSL an over-full GPU spills into system RAM (very slow) instead of raising
OOM, so watch the peak memory reported by `greedy`.

On Modal (`modal_rd/app.py` sets SCRATCH=/vol) the model is loaded in BF16, the KV cache stays on
the GPU and results go to the volume (/vol/results/exp_21_kascade). BB_QUANT=nf4|bf16 overrides.
The checkpoint and prompt format follow `modal_rd.model_config` (BB_MODEL=r1|nemotron), unless
BB_MODEL_PATH points at another checkpoint.
"""
import importlib
import json
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modal_rd.model_config import PROMPT_FORMAT  # noqa: E402
from modal_rd.niah import MODEL_PATH  # noqa: E402  ($SCRATCH/models/<BB_MODEL checkpoint>)

# BB_MODEL_PATH switches every exp 21 script to another checkpoint (Llama-3.1 family, Qwen2, GLM-4;
# e.g. $SCRATCH/models/Llama-3.1-8B-UltraLong-1M-Instruct); default R1-Distill-Llama-8B.
MODEL = os.environ.get("BB_MODEL_PATH", MODEL_PATH)
ON_MODAL = os.environ.get("SCRATCH") == "/vol"
QUANT = os.environ.get("BB_QUANT", "bf16" if ON_MODAL else "nf4")
OUT = (Path("/vol/results") if ON_MODAL else Path(__file__).parents[2] / "benchmarks") / "exp_21_kascade"


def model_type(model_path=MODEL):
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f).get("model_type", "llama")


def load_tok(model_path=MODEL):
    """Prompt tokenizer. Llama-3 checkpoints use the project loader (AutoTokenizer drops spaces on
    transformers 5.x and BOS must be explicit); other families use AutoTokenizer after a round-trip check."""
    if model_type(model_path) == "llama":
        from modal_rd.niah import load_tokenizer
        return load_tokenizer(model_path)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    probe = "One of the special magic numbers is 2940341.\nWhat is it?"
    if tok.decode(tok(probe, add_special_tokens=False)["input_ids"]) != probe:
        raise RuntimeError(f"Tokenizer round-trip failed for {model_path}")
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    tok.model_max_length = 1_100_000
    return tok


def use_model_rope(model):
    """Point v3 (modal_rd.attention) and sparse decode at the model's own RoPE.

    Both call Llama's apply_rotary_pos_emb; GLM-4 rotates only half of each head, interleaved,
    so Llama's version would silently scramble positions. Qwen2's is identical to Llama's.
    """
    from experiments.exp_21_kascade import sparse_decode
    from modal_rd import attention as rd
    attn = model.model.layers[0].self_attn
    attn = getattr(attn, "_rd_base", attn)
    fn = importlib.import_module(type(attn).__module__).apply_rotary_pos_emb
    rd.apply_rotary_pos_emb = fn
    rd.rope_inplace = lambda x, cos, sin: fn(x, x, cos, sin)[0]
    sparse_decode.apply_rotary_pos_emb = fn


def chunk_mlp(mlp, size=4096):
    f = mlp.forward
    mlp.forward = lambda x: torch.cat([f(x[:, i:i + size]) for i in range(0, x.shape[1], size)], 1)


def fixed_config(model_path=MODEL):
    """Config with known checkpoint fixes applied.

    THUDM/glm-4-9b-chat-1m-hf ships rope_theta=10000, but the original GLM-4-1M code uses
    base = 10000 * rope_ratio with rope_ratio=10000 (modeling_chatglm.py), i.e. theta = 1e8.
    With 10000 the model answers at ~1.5K tokens and degenerates by ~16K even with dense attention.
    """
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path)
    rp = getattr(cfg, "rope_parameters", None)
    if cfg.model_type == "glm" and cfg.max_position_embeddings >= 1_000_000 and rp and rp.get("rope_theta") == 10000.0:
        rp["rope_theta"] = 1e8
        print("fixed_config: GLM-4-1M rope_theta 10000 -> 1e8 (10000 * rope_ratio 10000)", flush=True)
    return cfg


def load_model(model_path=MODEL, mlp_chunk=None):
    """NF4 (local) or BF16 (Modal) model with chunked MLP; see QUANT."""
    q = None
    if QUANT == "nf4":
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=fixed_config(model_path), quantization_config=q,
                                                 dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map="cuda").eval()
    for layer in model.model.layers:
        chunk_mlp(layer.mlp, mlp_chunk or (4096 if QUANT == "nf4" else 8192))
    use_model_rope(model)
    return model


@torch.inference_mode()
def greedy(model, tokenizer, text, max_new=10, add_special_tokens=None, offload=None, stop_eos=False):
    """Prefill + greedy decode; returns (generated text, peak GB).

    add_special_tokens defaults to True for raw prompts and False for chat-templated ones
    (modal_rd.niah renders those with BOS already); pass it explicitly for other text.
    offload keeps the KV cache on the CPU between layers; default on for NF4 (local), off for BF16.
    stop_eos=True stops at the end-of-sequence token (long reasoning outputs).
    Positions follow the cache length, so do not use with KV eviction (see eval_decode.py).
    """
    if add_special_tokens is None:
        add_special_tokens = PROMPT_FORMAT == "raw"
    if offload is None:
        offload = QUANT == "nf4"
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens).to("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cache = DynamicCache(config=model.config, offloading=offload)
    o = model(**ids, use_cache=True, logits_to_keep=1, past_key_values=cache)
    new = [o.logits[0, -1].argmax().item()]
    for _ in range(max_new - 1):
        if stop_eos and new[-1] == tokenizer.eos_token_id:
            break
        o = model(input_ids=torch.tensor([[new[-1]]], device="cuda"), past_key_values=o.past_key_values,
                  use_cache=True, logits_to_keep=1)
        new.append(o.logits[0, -1].argmax().item())
    return tokenizer.decode(new, skip_special_tokens=True), torch.cuda.max_memory_allocated() / 1e9


def exact_number(gen, answer):
    m = re.search(r"\d+", gen)
    return bool(m and m[0] == answer)
