"""Generative NoLiMa eval: needle retrieval without literal overlap (Modarressi et al., ICML 2025).

The needle and the question share no keywords ("{CHAR} lives next to the Kiasma museum"
vs "Which character has been to Helsinki?"), so content routing cannot match on surface
words. Prompts reuse the repo's NoLiMa loader (`eval.nolima.nolima_dataset`)
with the official task template, sent as an R1 chat turn. By default the thinking block is
already closed ("<think>\n\n</think>\n\n") so the model answers directly; with --think it
reasons first and only the text after "</think>" is scored (no "</think>" = unfinished = wrong).
Correct = the gold character name appears in the answer. Without thinking, R1-Distill-8B
fails the latent inference even with dense attention at 2K (3/10; literal control 10/10). Data: `bash scripts/get_nolima_data.sh <dir>` and NOLIMA_DATA_DIR=<dir>.

Arms (";"-separated):
  dense                      exact attention
  all / all:tail256          v3 with every layer routing, exact tail 0 / 256
  0,2,8,13,14:remap:pool     exp 21 combined (anchors, Kascade head map, KV pooling)
  add ":decode" for sparse decode with the same reuse (otherwise decode is dense)

  python -m experiments.exp_21_kascade.eval_nolima --sets "dense;all:tail256;all;0,2,8,13,14:remap:pool" \
      --chars 62000 --n 40 --tag p3_nolima_16k
"""
import argparse
import json
import random
from pathlib import Path

import torch

from eval.nolima.nolima_dataset import (_TASK_TEMPLATE, _expand_pairs, _load_haystacks, _load_needle_groups,
                                        resolve_data_dir)
from experiments.exp_21_kascade import model as kascade
from experiments.exp_21_kascade.common import MODEL, load_tok, greedy, load_model, QUANT, OUT as EXP_OUT
from modal_rd import attention as rd

OUT = EXP_OUT / "phase3"


def place(book, needle, chars, depth, rng):
    """A random ~`chars`-character window of whole lines with the needle inserted at `depth`.

    (The loader's `_place_in_haystack` can start near the end of a book and return a much
    shorter window; here the window always has the full budget.)
    """
    start = rng.randrange(0, len(book) - chars - 1)
    start = book.index("\n", start) + 1
    lines = book[start:start + chars].split("\n")[:-1]           # drop the cut-off last line
    lines.insert(int(depth * len(lines)), needle)
    return "\n".join(lines)


def build_rows(tok, needle_file, chars, n, depths, seed=7, think=False):
    root = resolve_data_dir()
    pairs = _expand_pairs(_load_needle_groups(str(root / "needlesets" / needle_file)), None)
    hays = _load_haystacks(str(root / "haystack" / "rand_shuffle"))
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        pair = pairs[(i * 7) % len(pairs)]          # stride through groups, hops and tests
        name = rng.choice(pair["character_set"])
        needle = pair["needle_tmpl"].replace("{CHAR}", name)
        question = pair["question_tmpl"].replace("{CHAR}", name)
        depth = depths[i % len(depths)]
        placed = place(hays[i % len(hays)], needle, chars, depth, rng)
        prompt = (pair["task_template"] or _TASK_TEMPLATE).format(haystack=placed, question=question)
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True) + ("" if think else "\n</think>\n\n")
        rows.append({"i": i, "group": pair["group_id"], "test": pair["test_id"], "hop": pair["hop"],
                     "depth": depth, "answer": name, "others": [c for c in pair["character_set"] if c != name],
                     "needle": needle, "question": question, "text": text,
                     "tokens": len(tok(text, add_special_tokens=False)["input_ids"])})
    return rows


def install(model, spec, attn, map_tag):
    layers, *flags = spec.split(":")
    cfg = dict(attn, exact_tail=256 if "tail256" in flags else attn["exact_tail"])
    if layers == "dense":
        rd.install(model, "dense_rd", **cfg)
        return
    L = len(model.model.layers)
    anchors = list(range(L)) if layers == "all" else [int(x) for x in layers.split(",")]
    hmap = kascade.load_head_map(map_tag, anchors) if "remap" in flags else None
    kascade.install(model, anchors, head_map=hmap, pool="pool" in flags, decode="decode" in flags, **cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", required=True)
    p.add_argument("--needles", default="needle_set.json")
    p.add_argument("--chars", type=int, default=62000, help="haystack characters (~4 chars/token)")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--depths", default="0.25,0.5,0.75")
    p.add_argument("--max-new", type=int, default=16)
    p.add_argument("--attn", default='{"sink":64,"window":4096,"vertical":512,"exact_tail":0}')
    p.add_argument("--map-tag", default="p0_16k_tail0_pool")
    p.add_argument("--think", action="store_true", help="let R1 reason before answering")
    p.add_argument("--offload", action="store_true", help="CPU-offload the KV cache (needed near 16K)")
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    attn = json.loads(a.attn)
    OUT.mkdir(parents=True, exist_ok=True)

    tok = load_tok(MODEL)
    rows = build_rows(tok, a.needles, a.chars, a.n, [float(x) for x in a.depths.split(",")], think=a.think)
    print(f"prompts: {len(rows)}, tokens {min(r['tokens'] for r in rows)}-{max(r['tokens'] for r in rows)}", flush=True)
    model = load_model()
    report = {"needles": a.needles, "think": a.think, "chars": a.chars, "attn": attn, "max_new": a.max_new,
              "gpu": torch.cuda.get_device_name(0), "quant": QUANT, "arms": []}
    for spec in a.sets.split(";"):
        install(model, spec, attn, a.map_tag)
        runs = []
        for r in rows:
            gen, peak = greedy(model, tok, r["text"], a.max_new, add_special_tokens=False,
                               offload=a.offload, stop_eos=a.think)
            finished = not a.think or "</think>" in gen
            final = gen.split("</think>")[-1] if a.think else gen
            correct = finished and r["answer"] in final
            runs.append({k: r[k] for k in ("i", "group", "test", "hop", "depth", "answer", "tokens")} |
                        {"generated": gen, "final": final, "finished": finished, "correct": correct,
                         "other_name": any(o in final for o in r["others"]),
                         "peak_gb": peak})
            print(f"{spec} i={r['i']} {r['hop']} d={r['depth']} correct={correct} finished={finished} "
                  f"gold={r['answer']} final={final.strip()[:80]!r}", flush=True)
        by_hop = {}
        for x in runs:
            by_hop.setdefault(x["hop"], []).append(x["correct"])
        arm = {"spec": spec, "correct": sum(x["correct"] for x in runs), "n": len(runs),
               "unfinished": sum(not x["finished"] for x in runs),
               "by_hop": {h: f"{sum(v)}/{len(v)}" for h, v in sorted(by_hop.items())}, "runs": runs}
        report["arms"].append(arm)
        print(f"== nolima {spec}: {arm['correct']}/{arm['n']} unfinished={arm['unfinished']} by_hop={arm['by_hop']}", flush=True)
        (OUT / f"{a.tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
