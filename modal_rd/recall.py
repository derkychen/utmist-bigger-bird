"""Per-layer attention-mass recall of candidate sparse patterns vs exact dense.

One dense prefill per prompt. In every layer, for sampled query rows (`mid`
rows spread over the prompt, `tail` = final prompt tokens that produce the
answer) compute exact softmax rows and report, for each candidate key set:
  recall = dense probability mass on the selected keys
  cos    = cosine(sparse output renormalised over the selected keys, dense output)
The renormalised output is exactly what a sparse kernel computes on that set.
Patterns use only Q/K (no gold positions). Output: /vol/results/<tag>.json
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from modal_rd.niah import MODEL_PATH, load_tokenizer, prepare
from modal_rd.model_config import model_metadata, tokenize_prompt

PATTERNS = {
    # name: dict(sink, local, vertical (last-64-query mean prob cols), lastq (last-query score cols), oracle)
    "exp19like_l128_q4096": dict(local=128, lastq=4096),
    "exp19like_l128_q4096_sink64": dict(sink=64, local=128, lastq=4096),
    "stream_sink64_l4096": dict(sink=64, local=4096),
    "vs_s64_l1024_v2048": dict(sink=64, local=1024, vertical=2048),
    "vs_s64_l4096_v2048": dict(sink=64, local=4096, vertical=2048),
    "vs_s64_l4096_v4096": dict(sink=64, local=4096, vertical=4096),
    "vs_s64_l8192_v4096": dict(sink=64, local=8192, vertical=4096),
    "oracle_top2048": dict(oracle=2048),
    "oracle_top8192": dict(oracle=8192),
}


class DiagAttention(nn.Module):
    def __init__(self, base, store, mid, tail, n_route):
        super().__init__()
        cfg = base.config
        self.layer_idx, self.head_dim, self.scaling = base.layer_idx, base.head_dim, base.scaling
        self.h, self.hk = cfg.num_attention_heads, cfg.num_key_value_heads
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = base.q_proj, base.k_proj, base.v_proj, base.o_proj
        self.store, self.mid, self.tail, self.n_route = store, mid, tail, n_route

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kw):
        b, t, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, t, self.h, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, t, self.hk, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, t, self.hk, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scaling, enable_gqa=True)
        self.store.append(self.diagnose(q[0], k[0], v[0]))
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1)), None

    @torch.no_grad()
    def diagnose(self, q, k, v):
        n, g = k.shape[1], self.h // self.hk
        rows = torch.cat([self.mid, self.tail]).to(q.device)
        nm = len(self.mid)
        keypos = torch.arange(n, device=q.device)
        future = keypos[None, None, :] > rows[None, :, None]
        rpos = torch.arange(n - self.n_route, n, device=q.device)
        acc = {p: {"recall": [], "cos": []} for p in PATTERNS}
        for h0 in range(0, self.h, 8):
            hs = slice(h0, h0 + 8)
            kh = k.repeat_interleave(g, 0)[hs].float()
            vh = v.repeat_interleave(g, 0)[hs].float()
            s = torch.matmul(q[hs][:, rows].float(), kh.transpose(-1, -2)) * self.scaling
            probs = s.masked_fill(future, -float("inf")).softmax(-1)             # [8, R, n]
            dense = torch.matmul(probs, vh)
            sr = torch.matmul(q[hs][:, -self.n_route:].float(), kh.transpose(-1, -2)) * self.scaling
            col = sr.masked_fill(keypos[None, None, :] > rpos[None, :, None], -float("inf")).softmax(-1).mean(1)
            lastq = sr[:, -1]
            for name, cfg in PATTERNS.items():
                if "oracle" in cfg:
                    kk = min(cfg["oracle"], n)
                    mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, probs.topk(kk, -1).indices, True)
                else:
                    sink, local = cfg.get("sink", 0), cfg.get("local", 0)
                    mask = (keypos[None, None, :] < sink) | (keypos[None, None, :] > rows[None, :, None] - local)
                    for key, score in (("vertical", col), ("lastq", lastq)):
                        if cfg.get(key):
                            sc = score.clone()
                            sc[:, :sink] = -float("inf")
                            sc[:, max(0, n - (local if key == "vertical" else 128)):] = -float("inf")
                            sel = torch.zeros(sc.shape, dtype=torch.bool, device=q.device)
                            sel.scatter_(-1, sc.topk(min(cfg[key], n), -1).indices, True)
                            mask = mask | sel[:, None, :]
                    mask = mask & ~future
                pm = probs * mask
                rec = pm.sum(-1)
                sparse = torch.matmul(pm, vh) / rec.clamp_min(1e-20)[..., None]
                acc[name]["recall"].append(rec)
                acc[name]["cos"].append(F.cosine_similarity(sparse, dense, dim=-1))
        res = {"layer": self.layer_idx}
        for name, d in acc.items():
            rec, cos = torch.cat(d["recall"]), torch.cat(d["cos"])                # [H, R]
            res[name] = {"recall_mid": rec[:, :nm].mean().item(), "recall_tail": rec[:, nm:].mean().item(),
                         "cos_mid": cos[:, :nm].mean().item(), "cos_tail": cos[:, nm:].mean().item(),
                         "recall_tail_worst_head": rec[:, nm:].mean(1).min().item(),
                         "cos_mid_worst_head": cos[:, :nm].mean(1).min().item()}
        return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bytes", default="512000")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--mid", type=int, default=48)
    p.add_argument("--tail", type=int, default=16)
    p.add_argument("--n-route", type=int, default=64)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    tok = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    bases = [layer.self_attn for layer in model.model.layers]
    report = {**model_metadata(), "patterns": PATTERNS, "prompts": []}
    for nbytes in map(int, a.bytes.split(",")):
        for row in prepare(tok, nbytes, a.n):
            n = row["tokens"]
            mid = torch.linspace(4096, n - a.tail - 1, a.mid).long()
            tail = torch.arange(n - a.tail, n)
            store = []
            for layer, base in zip(model.model.layers, bases):
                layer.self_attn = DiagAttention(base, store, mid, tail, a.n_route)
            ids = tokenize_prompt(tok, row["text"], return_tensors="pt").input_ids.cuda()
            with torch.inference_mode():
                h = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
                top = model.lm_head(h).argmax(-1).item()
            report["prompts"].append({"bytes": nbytes, "idx": row["idx"], "tokens": n, "token_sha256": row["token_sha256"],
                                      "dense_next_token": tok.decode([top]), "layers": store})
            print(f"PROMPT bytes={nbytes} idx={row['idx']} tokens={n} next={tok.decode([top])!r}", flush=True)
            for name in PATTERNS:
                r = [l[name] for l in store]
                grp = lambda key, lo, hi: sum(x[key] for x in r[lo:hi]) / (hi - lo)
                print(f"  {name:32s} recall_tail L0-7 {grp('recall_tail',0,8):.3f} L8-15 {grp('recall_tail',8,16):.3f} "
                      f"L16-31 {grp('recall_tail',16,32):.3f} | cos_mid L0-7 {grp('cos_mid',0,8):.3f} "
                      f"L8-15 {grp('cos_mid',8,16):.3f} L16-31 {grp('cos_mid',16,32):.3f} | cos_tail all {grp('cos_tail',0,32):.3f}", flush=True)
            out = Path("/vol/results") / f"{a.tag}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2))
    for layer, base in zip(model.model.layers, bases):
        layer.self_attn = base
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
