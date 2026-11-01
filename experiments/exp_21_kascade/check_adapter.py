"""Check that v3 / exp 21 attention reproduces a model's own attention (adapter sanity check).

On a ~1.5K-token NIAH prompt, compares last-token logits and a short greedy continuation of the
model's native HF attention against:
  dense_rd     v3 wrapper, exact path            (RoPE + projection wiring)
  full_sparse  v3 sparse kernel, routes cover every key (kernel + GQA grouping)
  full_decode  exp 21 sparse decode covering every key (decode path)
All three should match native up to bf16/4-bit noise. A RoPE mistake shows up as a large logit gap.

  BB_MODEL_PATH=$SCRATCH/models/glm-4-9b-chat-1m-hf python -m experiments.exp_21_kascade.check_adapter
"""
import json
import sys

import torch

from experiments.exp_21_kascade import sparse_decode as sd
from experiments.exp_21_kascade.common import MODEL, load_model, load_tok, model_type
from modal_rd import attention as rd
from modal_rd.niah import prepare


class FullDecode(sd.SparseDecodeAttention):
    """Sparse decode forced on for every generated token, with routes covering every cached key."""

    def forward(self, hidden_states=None, **kw):
        # prefill: budget covers everything; decode: tiny budget so the sparse-decode branch is taken
        if hidden_states.shape[1] == 1:
            self.decode_vertical = 1
        else:
            self.vertical = 10 ** 9
        return super().forward(hidden_states, **kw)

    def sparse_decode(self, q, k, v):
        self.decode_vertical = k.shape[-2] - self.sink - self.decode_window
        return super().sparse_decode(q, k, v)


def install_full_decode(model, **attn):
    orig, sd.SparseDecodeAttention = sd.SparseDecodeAttention, FullDecode
    try:
        sd.install(model, list(range(len(model.model.layers))), **attn)
    finally:
        sd.SparseDecodeAttention = orig


@torch.inference_mode()
def run(model, ids, steps=8):
    o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    first = o.logits[0, -1].float()
    new = [first.argmax().item()]
    for _ in range(steps - 1):
        o = model(input_ids=torch.tensor([[new[-1]]], device="cuda"), past_key_values=o.past_key_values,
                  use_cache=True, logits_to_keep=1)
        new.append(o.logits[0, -1].argmax().item())
    return first, new


def main():
    tok = load_tok(MODEL)
    row = prepare(tok, 6000, 1, verify=False)[0]
    ids = tok(row["text"], return_tensors="pt").input_ids.cuda()
    n = ids.shape[1]
    model = load_model()
    print(f"model={MODEL} type={model_type()} layers={len(model.model.layers)} "
          f"heads={model.config.num_attention_heads}/{model.config.num_key_value_heads} tokens={n}", flush=True)

    ref_logits, ref_tokens = run(model, ids)
    full = dict(sink=16, window=64, vertical=n, exact_tail=0)            # routes cover every key
    arms = {
        "dense_rd": lambda: rd.install(model, "dense_rd", **full),
        "full_sparse": lambda: rd.install(model, "vs", **full),
        "full_decode": lambda: install_full_decode(model, **full),
    }
    report = {"model": MODEL, "tokens": n, "native_answer": tok.decode(ref_tokens), "answer": row["answer"], "arms": {}}
    print(f"native: {tok.decode(ref_tokens)!r} (gold {row['answer']})", flush=True)
    ok = True
    for name, install in arms.items():
        install()
        logits, tokens = run(model, ids)
        gap = (logits - ref_logits).abs().max().item()
        same = tokens == ref_tokens
        report["arms"][name] = {"max_logit_gap": gap, "same_tokens": same, "text": tok.decode(tokens)}
        ok &= same and gap < 1.0
        print(f"{name:12s} max|dlogit|={gap:.4f} same_tokens={same} text={tok.decode(tokens)!r}", flush=True)
    print("ADAPTER " + ("PASS" if ok else "CHECK"), flush=True)
    print(json.dumps(report))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
