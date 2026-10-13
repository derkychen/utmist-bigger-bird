"""Local (RTX 3080, 10 GB) model loading and greedy decoding for exp 21.

R1-Distill-Llama-8B is loaded in 4-bit NF4 so it fits in 10 GB. Three things keep
~16K-token prefill under the limit:
  * the MLP runs in 4096-token chunks (identical output, smaller activation peak);
  * logits are computed for the last position only;
  * the KV cache is offloaded to CPU between layers.
Under WSL an over-full GPU spills into system RAM (very slow) instead of raising
OOM, so watch the peak memory reported by `greedy`.
"""
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modal_rd.niah import MODEL_PATH  # noqa: E402  ($SCRATCH/models/DeepSeek-R1-Distill-Llama-8B)


def chunk_mlp(mlp, size=4096):
    f = mlp.forward
    mlp.forward = lambda x: torch.cat([f(x[:, i:i + size]) for i in range(0, x.shape[1], size)], 1)


def load_model_4bit(model_path=MODEL_PATH, mlp_chunk=4096):
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=q, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map="cuda").eval()
    for layer in model.model.layers:
        chunk_mlp(layer.mlp, mlp_chunk)
    return model


@torch.inference_mode()
def greedy(model, tokenizer, text, max_new=10, add_special_tokens=True, offload=True, stop_eos=False):
    """Prefill + greedy decode; returns (generated text, peak GB).

    Pass add_special_tokens=False for chat-templated text that already starts with BOS.
    offload=False keeps the KV cache on the GPU (faster decode; fine up to ~8-9K tokens).
    stop_eos=True stops at the end-of-sequence token (long reasoning outputs).
    """
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
