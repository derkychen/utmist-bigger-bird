"""Matched synthetic RULER-style NIAH prompts (mirror of the Round 4 builder).

Prompt text, answers and BOS-prefixed token IDs reproduce the CCDB Round 4
screens exactly; `prepare(..., verify=True)` checks token hashes against the
saved manifests when one exists for the requested source-byte length.
"""
import hashlib
import json
import os
import re
from pathlib import Path

from tokenizers import processors
from transformers import PreTrainedTokenizerFast

from eval.lra_llama.lra_llama_dataset import _ids_to_text
from eval.ruler.ruler_dataset import build_ruler_dataset

from modal_rd.model_config import (MODEL_DIR as MODEL_PATH, MAX_CONTEXT, MODEL_ID, R1_ID,
                                   PROMPT_FORMAT, render_prompt, tokenize_prompt)
MANIFESTS = Path(__file__).parent / "manifests"
MANIFEST_KEYS = ("idx", "label", "answer", "tokens", "token_sha256", "source_sha256", "spans", "question_tokens")


def load_tokenizer(model_path=MODEL_PATH):
    """tokenizer.json backend + explicit BOS (fdf13e8); verifies exact round trip."""
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    bos = tokenizer.bos_token
    tokenizer.backend_tokenizer.post_processor = processors.Sequence([
        processors.ByteLevel(trim_offsets=False),
        processors.TemplateProcessing(single=f"{bos}:0 $A:0", pair=f"{bos}:0 $A:0 {bos}:1 $B:1",
                                      special_tokens=[(bos, tokenizer.bos_token_id)]),
    ])
    probe = "One of the special magic numbers is 2940341.\nWhat is it?"
    if tokenizer.decode(tokenizer(probe, add_special_tokens=False)["input_ids"]) != probe:
        raise RuntimeError("Tokenizer round-trip failed")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = MAX_CONTEXT
    return tokenizer


def move_question_first(text, offset=0):
    """Move the trailing question near the start; the prompt still ends with ' The answer is:'.

    offset > 0 inserts it at the first sentence boundary after `offset` characters, so it
    falls outside the always-visible sink tokens.
    """
    cue = re.search(r" ?What (?:is|are)[^?]*\? The answer is:$", text)
    assert cue, "question not at the end"
    question = cue[0].strip()[:-len(" The answer is:")].strip()
    body = text[:cue.start()].rstrip()
    cut = body.index(". ", offset) + 2 if offset else 0
    return body[:cut] + question + ("\n" if not offset else " ") + body[cut:] + " The answer is:"


def prepare(tokenizer, seq_bytes, count, depth=0.5, seed=42, start=0, verify=True, question_first=False,
            question_offset=0, minimal=False):
    # Train/validation RNG streams are independent. A shorter validation split has
    # the identical prefix; minimal mode avoids materializing 128 huge probe rows.
    data = build_ruler_dataset(task="niah", seq_len=seq_bytes, needle_depth=depth,
                               train_samples=1 if minimal else 10,
                               eval_samples=start + count if minimal else max(128, start + count), seed=seed)["validation"]
    rows = []
    for idx in range(start, start + count):
        text = _ids_to_text(data[idx]["input_ids"]).rstrip() + " The answer is:"
        if question_first:
            text = move_question_first(text, question_offset)
        text = render_prompt(tokenizer, text)
        question = re.search(r"What is the special magic ([\w-]+)\?", text)
        needle = re.search(r"The special magic " + re.escape(question[1]) + r" is: (\d+)\.", text)
        assert needle and int(needle[1][-1]) == int(data[idx]["labels"]), (seq_bytes, idx)
        enc = tokenize_prompt(tokenizer, text, return_offsets_mapping=True, truncation=False)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        assert len(ids) + 10 <= MAX_CONTEXT, (seq_bytes, idx, len(ids))
        hits = lambda a0, b0: [i for i, (a, b) in enumerate(offsets) if b > a0 and a < b0 and b > a]
        spans = {k: [min(h), max(h) + 1] for k, h in (("needle", hits(needle.start(), needle.end())),
                                                       ("number", hits(needle.start(1), needle.end(1))))}
        q = hits(question.start(), question.end())
        rows.append({"idx": idx, "text": text, "label": int(data[idx]["labels"]), "answer": needle[1],
                     "tokens": len(ids), "token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                     "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "spans": spans, "question_tokens": [min(q), max(q) + 1], "depth": depth, "seed": seed})
    _verify(rows, seq_bytes, depth, seed, verify and not question_first)
    return rows


def prepare_task(tokenizer, task, seq_bytes, count, depth=0.5, seed=42, start=0, question_first=False,
                 question_offset=0):
    """Any RULER NIAH variant. Gold answers = every value stated for the queried key(s).

    Prompts whose queried key also appears with a different value elsewhere (possible
    with the distractor-needle haystack) are skipped as ambiguous; the skip count is
    returned alongside the rows.
    """
    if task in ("niah", "niah_single_1"):
        return prepare(tokenizer, seq_bytes, count, depth=depth, seed=seed, start=start,
                       question_first=question_first, question_offset=question_offset), 0
    if question_first:
        raise ValueError("question_first is only implemented for the single-needle niah task")
    data = build_ruler_dataset(task=task, seq_len=seq_bytes, needle_depth=depth,
                               train_samples=1, eval_samples=2 * (start + count) + 4, seed=seed)["validation"]
    rows, skipped, idx = [], 0, start
    while len(rows) < count and idx < len(data):
        text = _ids_to_text(data[idx]["input_ids"]).rstrip() + " The answer is:"
        text = render_prompt(tokenizer, text)
        cue = re.search(r"What (?:is the special magic (?P<one>[\w-]+)|are all values for the special magic "
                        r"(?P<multi>[\w-]+)|are the special magics for: (?P<keys>[^?]+))\?", text)
        keys = [cue["one"] or cue["multi"]] if not cue["keys"] else [k.strip() for k in cue["keys"].split(",")]
        answers, ambiguous = [], False
        for key in keys:
            vals = re.findall(r"The special magic " + re.escape(key) + r" is: ([\w-]+)\.", text)
            ambiguous |= not vals or (cue["multi"] is None and len(set(vals)) > 1)
            answers += vals
        idx += 1
        if ambiguous:
            skipped += 1
            continue
        ids = tokenize_prompt(tokenizer, text, truncation=False)["input_ids"]
        if len(ids) + 64 > MAX_CONTEXT:
            raise ValueError(f"{task} at {seq_bytes} bytes is {len(ids)} tokens; use fewer bytes")
        rows.append({"idx": idx - 1, "task": task, "text": text, "answers": answers, "answer": answers[0],
                     "label": int(data[idx - 1]["labels"]), "tokens": len(ids),
                     "token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                     "depth": depth, "seed": seed, "spans": None})
    return rows, skipped


def _verify(rows, seq_bytes, depth, seed, verify):
    manifest = MANIFESTS / f"bytes{seq_bytes}.json"
    if verify and MODEL_ID == R1_ID and PROMPT_FORMAT == "raw" and depth == 0.5 and seed == 42 and manifest.exists():
        saved = {r["idx"]: r for r in json.loads(manifest.read_text())}
        for r in rows:
            if r["idx"] in saved:
                assert {k: r[k] for k in MANIFEST_KEYS} == saved[r["idx"]], f"prompt {r['idx']} differs from CCDB manifest"
    return rows
