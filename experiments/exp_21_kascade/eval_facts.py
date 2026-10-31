"""Follow-up fact retention at long context: KV eviction budgets vs a compressed full cache.

Test A (facts): each ~1M-token multikey_1 prompt gets 4 planted facts with different templates (no look-alikes) at
15 / 35 / 55 / 80% of the haystack. One prefill, then a chat: the prompt's own needle question, then each fact in
turn (order shuffled per prompt). Every answer is scored by exact 6-digit / 7-digit match.
Test B (long): the question asks for a ~300-word summary before the answer; scored on the final number.

Arms
  evict   v3 + KV eviction keeping --keep routed keys per KV head (MEMORY_DEFAULTS pooling); one budget per run.
  turbo   v3 with no eviction over the TurboQuant-style compressed full cache (modal_rd/turbo_kv.py, --kv-bits).
Decode is the CUDA-graph decoder (modal_rd/graph_decode.py) in both arms.

  BB_MODEL=nemotron python -m experiments.exp_21_kascade.eval_facts --arm evict --keep 65536 \\
      --bytes 4848732 --start 5 --n 25 --long-n 10 --fp8 --tag facts_1m_evict
"""
import argparse
import gc
import hashlib
import json
import random
import re
import time

import torch

from experiments.exp_21_kascade.common import MODEL, OUT as EXP_OUT, QUANT, load_model, load_tok
from experiments.exp_21_kascade.eval_decode import LONG_ASK, final_number, turn_suffix
from modal_rd import attention as rd
from modal_rd import graph_decode
from modal_rd.model_config import model_metadata
from modal_rd.niah import prepare_task

OUT = EXP_OUT / "phase5"
ATTN = {"sink": 64, "window": 16384, "vertical": 512, "exact_tail": 256}
EVICT = {"kv_evict": True, "evict_pool": 8}
COLORS = ["blue", "green", "red", "yellow", "orange", "purple", "silver", "black", "white", "brown"]
ANIMALS = ["otter", "falcon", "badger", "heron", "lynx", "walrus", "gecko", "bison", "marmot", "ibis"]
RIVERS = ["Danube", "Mekong", "Yukon", "Volga", "Orinoco", "Zambezi", "Rhine", "Murray", "Indus", "Loire"]
PLANETS = ["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Ceres", "Eris"]
TEMPLATES = [
    (COLORS, "The access code for the {} locker is {}.", "What is the access code for the {} locker?"),
    (ANIMALS, "The serial number of the {} printer is {}.", "What is the serial number of the {} printer?"),
    (RIVERS, "The ticket number for the {} ferry is {}.", "What is the ticket number for the {} ferry?"),
    (PLANETS, "The PIN for the {} account is {}.", "What is the PIN for the {} account?"),
]
DEPTHS = (0.15, 0.35, 0.55, 0.80)


def plant_facts(text, idx):
    """Insert one fact per template at DEPTHS of the user turn. Returns (text, [(depth, question, answer)])."""
    rng = random.Random(7000 + idx)
    lo = text.index("<|start_header_id|>user<|end_header_id|>")
    hi = text.rindex("<|eot_id|><|start_header_id|>assistant")
    facts, inserts = [], []
    for depth, (names, fact, question) in zip(DEPTHS, rng.sample(TEMPLATES, len(TEMPLATES))):
        name, code = rng.choice(names), str(rng.randrange(100000, 1000000))
        at = text.index(". ", lo + int((hi - lo) * depth)) + 2
        inserts.append((at, fact.format(name, code) + " "))
        facts.append((depth, question.format(name) + " The answer is:", code))
    for at, s in sorted(inserts, reverse=True):            # back to front so earlier offsets stay valid
        text = text[:at] + s + text[at:]
    rng.shuffle(facts)
    return text, facts


def hit(out, answer):
    return answer in re.sub(r"[\s,]", "", out)


class Chat:
    """A greedy conversation on a StaticDecoder loaded with a prefilled cache."""

    def __init__(self, dec, tok, eos):
        self.dec, self.tok, self.eos = dec, tok, eos

    def answer(self, first, max_new):
        self.dec.feed(first)
        toks, _ = self.dec.generate(max_new - 1, self.eos)
        return [first] + toks

    def ask(self, last, question, max_new):
        end_turn, suffix = turn_suffix(self.tok, question)
        ids = ([] if last in self.eos else self.tok(end_turn, add_special_tokens=False)["input_ids"])
        ids += self.tok(suffix, add_special_tokens=False)["input_ids"]
        for i in ids:
            self.dec.feed(i)
            self.dec.run(1)
        return self.answer(int(self.dec.token), max_new)


def conversation(chat, row, facts, first):
    out = {"q1": chat.answer(first, 32)}
    res = {"needle": hit(chat.tok.decode(out["q1"], skip_special_tokens=True), row["answer"]), "facts": []}
    last = out["q1"][-1]
    for depth, question, answer in facts:
        ids = chat.ask(last, question, 32)
        text = chat.tok.decode(ids, skip_special_tokens=True)
        res["facts"].append({"depth": depth, "question": question, "answer": answer, "output": text,
                             "correct": hit(text, answer)})
        last = ids[-1]
    res["needle_output"] = chat.tok.decode(out["q1"], skip_special_tokens=True)
    return res


def load_prompts(tok, a):
    """Planted-fact and long-answer prompts as token ids, built once and cached on the volume (building ~1M-token
    prompts takes ~10-60 s each on CPU; run with --build-only in a CPU-only job first)."""
    path = OUT / f"prompts_mk1_{a.bytes}_{a.start}_{a.n}_{a.long_n}.pt"
    if path.exists():
        return torch.load(path), path
    import modal_rd.hard_1m_pilot  # noqa: F401  (fast byte decoder: ~5x faster prompt building)
    rows = prepare_task(tok, "niah_multikey_1", a.bytes, a.n, start=a.start)[0]
    prompts = []
    for i, row in enumerate(rows):
        text, facts = plant_facts(row["text"], row["idx"])
        item = {"idx": row["idx"], "answer": row["answer"], "facts": facts,
                "facts_ids": torch.tensor(tok(text, add_special_tokens=False)["input_ids"], dtype=torch.int32)}
        if i < a.long_n:
            long_text = row["text"].replace("? The answer is:", "?" + LONG_ASK, 1)
            item["long_ids"] = torch.tensor(tok(long_text, add_special_tokens=False)["input_ids"], dtype=torch.int32)
        item["sha256"] = hashlib.sha256(item["facts_ids"].numpy().tobytes()).hexdigest()
        prompts.append(item)
        print(f"built prompt idx={row['idx']} tokens={len(item['facts_ids'])}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(prompts, path)
    return prompts, path


def main(argv=None, loaded=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["evict", "turbo"])
    p.add_argument("--keep", type=int, default=16384)
    p.add_argument("--kv-bits", type=int, default=4)
    p.add_argument("--uniform", action="store_true", help="turbo: evenly spaced codebook (faster decode)")
    p.add_argument("--bytes", type=int, default=4848732)
    p.add_argument("--start", type=int, default=5)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--long-n", type=int, default=10)
    p.add_argument("--long-max", type=int, default=384)
    p.add_argument("--attn", default=json.dumps(ATTN))
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--build-only", action="store_true", help="build and cache the prompts (CPU only), then exit")
    p.add_argument("--tag", default="")
    a = p.parse_args(argv)
    attn = json.loads(a.attn)
    tok = loaded[0] if loaded else load_tok(MODEL)
    prompts, prompt_path = load_prompts(tok, a)
    if a.build_only:
        print(f"PROMPTS {prompt_path} ({len(prompts)} prompts)", flush=True)
        return
    model = loaded[1] if loaded else load_model()
    if a.fp8:
        from modal_rd.fp8 import install_fp8
        install_fp8(model)
    eos = {tok.eos_token_id, *([model.generation_config.eos_token_id] if isinstance(model.generation_config.eos_token_id, int)
                                else model.generation_config.eos_token_id or [])}
    if a.arm == "evict":
        rd.install(model, "vs", **{**attn, **EVICT, "evict_keep": a.keep})
        arms = [f"evict{a.keep}"]
    else:
        rd.install(model, "vs", **attn)
        arms = [f"turbo{a.kv_bits}" + ("u" if a.uniform else "")]
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"prompts": str(prompt_path), "arm": a.arm, "arms": arms, "keep": a.keep, "kv_bits": a.kv_bits, "attn": attn, "fp8": a.fp8, "quant": QUANT,
              "gpu": torch.cuda.get_device_name(0), "model": model_metadata(), "depths": DEPTHS,
              "facts": {x: [] for x in arms}, "long": {x: [] for x in arms}}
    path = OUT / f"{a.tag}.json"
    jobs = [("facts", r) for r in prompts] + [("long", r) for r in prompts if "long_ids" in r]
    dec = out = chat = None
    for test, row in jobs:
        # free the previous prompt's decoder (also held by `chat`) and logits before reading the next ~1M prompt
        dec = out = chat = None
        graph_decode._DECODERS.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        facts = row["facts"] if test == "facts" else None
        ids = row["facts_ids" if test == "facts" else "long_ids"].long()[None].cuda()
        n = ids.shape[1]
        max_new = 4 * 32 + 32 + 512 if test == "facts" else a.long_max + 16
        t0 = time.perf_counter()
        with torch.inference_mode():
            dec, out, prefill_s = graph_decode.graph_prefill(model, ids, max_new,
                                                             turbo_bits=a.kv_bits if a.arm == "turbo" else 0,
                                                             turbo_uniform=a.uniform)
            first = out.logits[0, -1].argmax().item()
            for name in arms:
                if dec.graph is None:
                    dec.capture()
                chat = Chat(dec, tok, eos)
                if test == "facts":
                    res = conversation(chat, row, facts, first)
                else:
                    ids_out = chat.answer(first, a.long_max)
                    textout = tok.decode(ids_out, skip_special_tokens=True)
                    res = {"output": textout[-200:], "gen_tokens": len(ids_out),
                           "correct": final_number(textout) == row["answer"]}
                res.update(idx=row["idx"], tokens=n, prefill_s=prefill_s, peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                           kept=dec.length.item() if a.arm == "evict" else None)
                report[test][name].append(res)
                if test == "facts":
                    print(f"{test} {name} idx={row['idx']} needle={res['needle']} facts="
                          f"{sum(f['correct'] for f in res['facts'])}/4 prefill={prefill_s:.0f}s "
                          f"peak={res['peak_gb']:.1f}GB", flush=True)
                else:
                    print(f"{test} {name} idx={row['idx']} correct={res['correct']} gen={res['gen_tokens']}", flush=True)
        report["elapsed_s"] = report.get("elapsed_s", 0) + time.perf_counter() - t0
        path.write_text(json.dumps(report, indent=1))
    for name in arms:
        f = report["facts"][name]
        print(f"== {name}: needle {sum(r['needle'] for r in f)}/{len(f)}  facts "
              f"{sum(x['correct'] for r in f for x in r['facts'])}/{4 * len(f)}  long "
              f"{sum(r['correct'] for r in report['long'][name])}/{len(report['long'][name])}", flush=True)


if __name__ == "__main__":
    main()
