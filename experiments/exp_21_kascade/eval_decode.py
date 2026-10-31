"""Exp 21 phase 4: decode tests, i.e. what happens after the prompt is read.

  --mode long      The question asks for a ~300-word summary of the text before the answer, so the
                   answer comes after a long generation. Measures decode ms/token (whole model and
                   attention modules alone), answer accuracy and how long each arm's tokens agree
                   with the first arm's (first divergence).
  --mode followup  Question 1 (the prompt's own), then question 2 appended to the same KV cache, with
                   no second prefill. KV eviction keeps only the keys question 1 routed to, so v3+evict
                   is expected to miss question 2; exp 21 keeps the full cache and re-routes every
                   decode step. --q2 needle asks for another multikey needle (too easy: the needles share
                   a template, so question 1's routes keep them all); --q2 fact (default) asks for a
                   differently worded fact inserted at --fact-depth of the haystack.

Prompts are RULER niah_multikey_1 (4 needles with different keys, the question asks for one), in the
checkpoint's chat format (BB_MODEL=nemotron). Arms: "dense", "v3", "v3+evict" (MEMORY_DEFAULTS
eviction) and `model.install_spec` reuse specs, e.g. "dp5:remap:pool:decodeonly:dw4096". v3 decodes
exactly, so it is the reference for decode-only changes.

Positions are passed explicitly (eviction shortens the cache), and appended turns are fed one token
at a time (multi-token appends to a cache are unmasked in v3 / sparse decode).

--graph decodes with CUDA graphs (`modal_rd/graph_decode.py`) after the same eager prefill: no Python or
launch overhead per token, so ms/token reflects GPU work. Generation stops within 8 steps of an EOS (cache
rewound to match), and attention-only time is not measured.

  BB_MODEL=nemotron python -m experiments.exp_21_kascade.eval_decode --mode followup \\
      --sets "v3;v3+evict;dp5:remap:pool:decodeonly" --map-tag x --bytes 512000 --n 6 --tag y
"""
import argparse
import json
import random
import re
import time

import torch
from transformers import DynamicCache

from experiments.exp_21_kascade import model as kascade
from experiments.exp_21_kascade.common import MODEL, OUT as EXP_OUT, QUANT, load_model, load_tok
from experiments.exp_21_kascade.model import EVICT
from experiments.exp_21_kascade.sparse_decode import kv_reads
from modal_rd.graph_decode import graph_prefill
from modal_rd.model_config import PROMPT_FORMAT, model_metadata
from modal_rd.niah import prepare_task

OUT = EXP_OUT / "phase4"
LONG_ASK = (" First summarize the text above in about 300 words. Then give the answer on its own final line"
            " as 'The answer is: <number>'.")


def needles(text):
    return re.findall(r"The special magic ([\w-]+) is: (\d+)\.", text)


COLORS = ["blue", "green", "red", "yellow", "orange", "purple", "silver", "black", "white", "brown"]


def insert_fact(text, idx, depth):
    """Insert 'The access code for the <color> locker is <6 digits>.' at a sentence boundary near `depth`
    of the haystack (the user turn). Returns (text, question, answer)."""
    rng = random.Random(1000 + idx)
    color, code = rng.choice(COLORS), str(rng.randrange(100000, 1000000))
    lo = text.index("<|start_header_id|>user<|end_header_id|>")
    hi = text.rindex("<|eot_id|><|start_header_id|>assistant")
    at = text.index(". ", lo + int((hi - lo) * depth)) + 2
    fact = f"The access code for the {color} locker is {code}. "
    return text[:at] + fact + text[at:], f"What is the access code for the {color} locker? The answer is:", code


def turn_suffix(tok, question):
    """Chat-template text that ends the assistant turn and adds a new user turn plus the assistant header."""
    m = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    a = tok.apply_chat_template(m, tokenize=False)
    b = tok.apply_chat_template(m + [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    assert b.startswith(a)
    return a[a.rindex("y") + 1:], b[len(a):]


class Session:
    """One prompt's KV cache with explicit absolute positions and CUDA-synchronised timing.

    During `generate`, CUDA events around every attention module also time attention alone
    (`attn_s`), separating it from the MLP / projections / Python overhead in `decode_s`."""

    offload = False

    def __init__(self, model, eos):
        self.model, self.eos, self.pos, self.forwards, self.decode_s, self.attn_s = model, eos, 0, 0, 0.0, 0.0
        self.cache = DynamicCache(config=model.config, offloading=self.offload)
        self.events = None

    def _hooks(self):
        def pre(m, a, kw):
            if self.events is not None:
                self.events.append([torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)])
                self.events[-1][0].record()

        def post(m, a, kw, o):
            if self.events is not None:
                self.events[-1][1].record()
        return [h for layer in self.model.model.layers for h in (
            layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True),
            layer.self_attn.register_forward_hook(post, with_kwargs=True))]

    @torch.inference_mode()
    def forward(self, ids):
        t = len(ids)
        pos = torch.arange(self.pos, self.pos + t, device="cuda")[None]
        o = self.model(input_ids=torch.tensor([ids], device="cuda"), position_ids=pos,
                       past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        self.pos += t
        self.forwards += 1
        return o.logits[0, -1].argmax().item()

    def prefill(self, ids):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        nxt = self.forward(ids)
        torch.cuda.synchronize()
        return nxt, time.perf_counter() - t0

    def append(self, ids):
        nxt = None
        for i in ids:
            nxt = self.forward([i])
        return nxt

    def generate(self, nxt, max_new):
        """Greedy from the already-predicted token `nxt`; stops after an EOS (kept in the output)."""
        out, steps, hooks = [], 0, self._hooks()
        self.events = []
        try:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            while True:
                out.append(nxt)
                if nxt in self.eos or len(out) >= max_new:
                    break
                nxt = self.forward([nxt])
                steps += 1
            torch.cuda.synchronize()
            self.decode_s += time.perf_counter() - t0
            self.attn_s += sum(a.elapsed_time(b) for a, b in self.events) / 1000
        finally:
            self.events = None
            for h in hooks:
                h.remove()
        return out, steps


class GraphSession(Session):
    """Session whose decode steps replay a CUDA graph (modal_rd.graph_decode); prefill stays eager."""

    slack = 0                       # cache slots needed after prefill (generated + appended tokens)
    turbo_bits = 0                  # > 0: compressed full cache (--kv-bits)

    def prefill(self, ids):
        self.dec, out, t = graph_prefill(self.model, torch.tensor([ids], device="cuda"), self.slack,
                                         turbo_bits=self.turbo_bits)
        self.cache = None
        self.pos += len(ids)
        self.forwards += 1
        return out.logits[0, -1].argmax().item(), t

    def append(self, ids):
        for i in ids:
            self.dec.feed(i)
            self.dec.run(1)
        self.pos += len(ids)
        return int(self.dec.token)

    def generate(self, nxt, max_new):
        self.dec.feed(nxt)
        if self.dec.graph is None:
            self.dec.capture()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        toks, steps = self.dec.generate(max_new - 1, self.eos)
        torch.cuda.synchronize()
        self.decode_s += time.perf_counter() - t0
        out = [nxt] + toks
        self.pos += len(out) - 1
        return out, steps


class OffloadSession(Session):
    """Session with the CPU-offloaded KV cache (offload_cache.py): exp 21 decodeonly arms only."""

    slack = 0

    def prefill(self, ids):
        from experiments.exp_21_kascade.offload_cache import OffloadKVCache
        self.cache = OffloadKVCache(self.model, len(ids) + self.slack)
        out = super().prefill(ids)
        self.cache.after_prefill()
        return out


class TurboSession(Session):
    """Session with the TurboQuant-style compressed full KV cache (modal_rd/turbo_kv.py); `bits` per coordinate."""

    slack, bits = 0, 4

    def prefill(self, ids):
        from modal_rd.turbo_kv import TurboKVCache
        # codes staged on the CPU during prefill so they do not add to its activation peak (~1M tokens)
        self.cache = TurboKVCache(self.model, len(ids) + self.slack, self.bits, stage_on_cpu=len(ids) > 262144)
        out = super().prefill(ids)
        self.cache.after_prefill()
        return out


def final_number(text):
    m = re.findall(r"answer is:?\**\s*\**(\d+)", text)
    return m[-1] if m else None


def run_long(sess, tok, row, max_new):
    text = row["text"].replace("? The answer is:", "?" + LONG_ASK, 1)
    assert text != row["text"]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    nxt, prefill_s = sess.prefill(ids)
    gen, steps = sess.generate(nxt, max_new)
    out = tok.decode(gen, skip_special_tokens=True)
    return {"tokens": len(ids), "prefill_s": prefill_s, "generated_ids": gen, "generated": out,
            "gen_tokens": len(gen), "decode_ms_per_token": 1000 * sess.decode_s / max(1, steps),
            "attn_ms_per_token": 1000 * sess.attn_s / max(1, steps),
            "exact": final_number(out) == row["answer"], "finished": gen[-1] in sess.eos}


def run_followup(sess, tok, row, max_new, q2="fact", fact_depth=0.25, fact_first=False):
    answer1 = row["answer"]
    if q2 == "fact":
        text, question2, answer2 = insert_fact(row["text"], row["idx"], fact_depth)
        key2 = question2
        if fact_first:      # control: ask the fact first (eviction keeps what question 1 routes to), needle second
            needle_q = re.search(r"What is the special magic [\w-]+\? The answer is:", text)[0]
            text = text.replace(needle_q, question2, 1)
            question2, key2, answer1, answer2 = needle_q, needle_q, answer2, row["answer"]
    else:
        text = row["text"]
        found = dict(needles(text))
        target = next(k for k, v in found.items() if v == row["answer"])
        key2 = next(k for k in found if k != target)
        question2, answer2 = f"What is the special magic {key2}? The answer is:", found[key2]
    end_turn, suffix = turn_suffix(tok, question2)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    nxt, prefill_s = sess.prefill(ids)
    gen1, steps1 = sess.generate(nxt, max_new)
    tail = [] if gen1[-1] in sess.eos else tok(end_turn, add_special_tokens=False)["input_ids"]
    nxt = sess.append(tail + tok(suffix, add_special_tokens=False)["input_ids"])
    gen2, steps2 = sess.generate(nxt, max_new)
    out1, out2 = tok.decode(gen1, skip_special_tokens=True), tok.decode(gen2, skip_special_tokens=True)
    hit = lambda out, ans: ans in re.sub(r"[\s,]", "", out)
    return {"tokens": len(ids), "prefill_s": prefill_s, "q2": q2, "key2": key2, "answer2": answer2,
            "generated": out1, "generated2": out2, "generated_ids": gen1 + gen2,
            "exact": hit(out1, answer1), "exact2": hit(out2, answer2), "fact_first": fact_first,
            "decode_ms_per_token": 1000 * sess.decode_s / max(1, steps1 + steps2),
            "attn_ms_per_token": 1000 * sess.attn_s / max(1, steps1 + steps2)}


def first_divergence(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main(argv=None, loaded=None):
    """`loaded` = (tokenizer, model) to reuse an already loaded model (h100_plan.py)."""
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["long", "followup"])
    p.add_argument("--sets", required=True, help='";"-separated arms: dense, v3, v3+evict or install_spec specs')
    p.add_argument("--map-tag", default=None, help="Phase 0 run for remap head maps")
    p.add_argument("--attn", default='{"sink":64,"window":16384,"vertical":512,"exact_tail":256}')
    p.add_argument("--task", default="niah_multikey_1")
    p.add_argument("--bytes", type=int, default=512000)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-new", type=int, default=0, help="default 384 (long) / 32 (followup)")
    p.add_argument("--q2", default="fact", choices=["fact", "needle"], help="followup: what question 2 asks for")
    p.add_argument("--fact-depth", type=float, default=0.25, help="followup --q2 fact: where the fact goes")
    p.add_argument("--fact-first", action="store_true", help="followup --q2 fact: ask the fact first, the needle second")
    p.add_argument("--warm", type=int, default=1, help="untimed warm-up prompts per arm (Triton compiles)")
    p.add_argument("--graph", action="store_true", help="decode with CUDA graphs (graph_decode.py)")
    p.add_argument("--kv-bits", type=int, default=0, help="TurboQuant-style compressed full KV cache, bits per value")
    p.add_argument("--cpu-cache", action="store_true", help="CPU-offloaded KV cache (offload_cache.py; exp 21 decodeonly)")
    p.add_argument("--offload", action="store_true", help="CPU-offload the KV cache (local runs past ~9K tokens)")
    p.add_argument("--tag", required=True)
    a = p.parse_args(argv)
    Session.offload = a.offload
    if PROMPT_FORMAT != "chat":
        raise ValueError("eval_decode needs a chat-format checkpoint (BB_MODEL=nemotron)")
    max_new = a.max_new or (384 if a.mode == "long" else 32)
    attn = json.loads(a.attn)
    OUT.mkdir(parents=True, exist_ok=True)

    tok, model = loaded or (load_tok(MODEL), load_model())
    L = len(model.model.layers)
    eos = {tok.eos_token_id, *([model.generation_config.eos_token_id] if isinstance(model.generation_config.eos_token_id, int)
                                else model.generation_config.eos_token_id or [])}
    rows, skipped = prepare_task(tok, a.task, a.bytes, a.n, start=a.start)
    S = GraphSession if a.graph else OffloadSession if a.cpu_cache else TurboSession if a.kv_bits else Session
    GraphSession.turbo_bits = a.kv_bits
    OffloadSession.slack = TurboSession.slack = 2 * max_new + 256
    TurboSession.bits = a.kv_bits
    GraphSession.slack = 2 * max_new + 256
    run = run_long if a.mode == "long" else \
        lambda sess, tok, row, max_new: run_followup(sess, tok, row, max_new, a.q2, a.fact_depth, a.fact_first)
    report = {"mode": a.mode, "graph": a.graph, "cpu_cache": a.cpu_cache, "kv_bits": a.kv_bits, "q2": a.q2, "fact_depth": a.fact_depth, "fact_first": a.fact_first, "task": a.task, "bytes": a.bytes, "max_new": max_new, "attn": attn, "quant": QUANT,
              "gpu": torch.cuda.get_device_name(0), "model": model_metadata(), "skipped_ambiguous": skipped,
              "manifest": [{k: v for k, v in r.items() if k != "text"} for r in rows], "arms": []}
    for spec in a.sets.split(";"):
        if spec == "v3+evict":
            anchors = kascade.install_spec(model, "v3", {**EVICT, **attn})
        else:
            anchors = kascade.install_spec(model, spec, attn, a.map_tag)
        for row in rows[:a.warm]:
            run(S(model, eos), tok, row, 8)
        res = []
        for row in rows:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            r = run(S(model, eos), tok, row, max_new)
            r.update(idx=row["idx"], answer=row["answer"], peak_gb=torch.cuda.max_memory_allocated() / 1e9)
            res.append(r)
            print(f"{spec} idx={row['idx']} tok={r['tokens']} exact={r['exact']}"
                  + (f" exact2={r['exact2']} q2={r['generated2']!r}" if a.mode == "followup" else f" gen={r['gen_tokens']}")
                  + f" prefill={r['prefill_s']:.1f}s decode={r['decode_ms_per_token']:.1f}ms/tok"
                  + f" (attn {r['attn_ms_per_token']:.1f}) peak={r['peak_gb']:.1f}GB"
                  + f" out={r['generated'][-80:]!r}", flush=True)
        arm = {"spec": spec, "anchors": anchors, "n": len(res), "exact": sum(r["exact"] for r in res),
               "decode_ms_per_token": sum(r["decode_ms_per_token"] for r in res) / len(res),
               "attn_ms_per_token": sum(r["attn_ms_per_token"] for r in res) / len(res),
               "prefill_s": sum(r["prefill_s"] for r in res) / len(res),
               "peak_gb": max(r["peak_gb"] for r in res), "runs": res}
        if a.mode == "followup":
            arm["exact2"] = sum(r["exact2"] for r in res)
        if "decode" in spec:
            n = sum(r["tokens"] for r in res) // len(res)
            num = lambda prefix, default: next((int(f[2:]) for f in spec.split(":") if f.startswith(prefix)), default)
            sparse, dense = kv_reads(n, L, anchors, attn["sink"], num("dw", attn["window"]), num("dv", attn["vertical"]))
            arm["kv_read_fraction"] = sparse / dense
        if report["arms"]:
            ref = {r["idx"]: r["generated_ids"] for r in report["arms"][0]["runs"]}
            arm["first_divergence_vs_" + report["arms"][0]["spec"]] = [first_divergence(ref[r["idx"]], r["generated_ids"]) for r in res]
        report["arms"].append(arm)
        print(f"== {a.mode} {spec}: q1 {arm['exact']}/{arm['n']}" + (f" q2 {arm['exact2']}/{arm['n']}" if a.mode == "followup" else "")
              + f" decode {arm['decode_ms_per_token']:.1f} ms/tok (attn {arm['attn_ms_per_token']:.1f}) prefill {arm['prefill_s']:.1f}s peak {arm['peak_gb']:.1f}GB", flush=True)
        (OUT / f"{a.tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
