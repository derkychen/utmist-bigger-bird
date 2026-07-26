"""Convert LRA/RULER byte-level datasets to Llama-tokenized text datasets.

The original LRA/RULER datasets use a tiny byte-level vocabulary (260 tokens)
designed for from-scratch BART. To use R1-Distill-Llama-8B, we convert the
byte ids back to text strings and tokenize with the Llama tokenizer.

For listops: the integer ids map back to symbolic tokens like "[MAX", "3", "]".
For text/retrieval: byte ids map back to UTF-8 text.
For RULER (niah/mq_niah): byte ids map back to the haystack+needle text.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets import Dataset

from shared.lra_dataset import (
    NUM_SPECIAL, CLS_ID, EOS_ID, PAD_ID,
    _LISTOPS_TOKENS, _LISTOPS_VOCAB,
)
from shared.ruler_dataset import TASK_INFO as RULER_TASK_INFO
from shared.lra_dataset import TASK_INFO as LRA_TASK_INFO


def _ids_to_listops_text(ids):
    """Convert listops integer ids back to text."""
    tokens = []
    for i in ids:
        if i == CLS_ID or i == EOS_ID or i == PAD_ID:
            continue
        if i < NUM_SPECIAL:
            continue
        idx = i - NUM_SPECIAL
        if idx < len(_LISTOPS_TOKENS):
            tokens.append(_LISTOPS_TOKENS[idx])
    return " ".join(tokens)


def _ids_to_text(ids):
    """Convert byte-level ids back to UTF-8 text."""
    raw_bytes = []
    for i in ids:
        if i == CLS_ID or i == EOS_ID or i == PAD_ID:
            continue
        if i < NUM_SPECIAL:
            continue
        raw_bytes.append(i - NUM_SPECIAL)
    return bytes(raw_bytes).decode("utf-8", errors="ignore")


def _convert_row(row, task, tokenizer, max_len):
    """Convert a single LRA/RULER row to Llama-tokenized format."""
    ids = row["input_ids"]
    label = row["labels"]

    if task == "listops":
        text = _ids_to_listops_text(ids)
    else:
        # text, retrieval, niah, mq_niah — all byte-level
        text = _ids_to_text(ids)

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"][0],
        "attention_mask": enc["attention_mask"][0],
        "labels": label,
    }


def build_llama_dataset(task, tokenizer, original_ds, max_len, pair=False):
    """Convert an existing LRA/RULER dataset to Llama-tokenized format.

    Args:
        task: "listops", "text", "niah", "mq_niah"
        tokenizer: Llama tokenizer
        original_ds: dict with "train" and "validation" Datasets
        max_len: max sequence length for tokenization
        pair: True for retrieval (dual-tower)

    Returns:
        dict with "train" and "validation" Datasets, tokenized for Llama
    """
    if pair:
        # Retrieval: convert both documents
        def convert_pair(row):
            a = _convert_row({"input_ids": row.get("input_ids_a", row.get("input_ids", [])), "labels": 0}, task, tokenizer, max_len)
            b = _convert_row({"input_ids": row.get("input_ids_b", row.get("input_ids", [])), "labels": 0}, task, tokenizer, max_len)
            return {
                "input_ids_a": a["input_ids"],
                "attention_mask_a": a["attention_mask"],
                "input_ids_b": b["input_ids"],
                "attention_mask_b": b["attention_mask"],
                "labels": row["labels"],
            }
        return {
            split: ds.map(convert_pair, remove_columns=ds.column_names)
            for split, ds in original_ds.items()
        }

    def convert_single(row):
        return _convert_row(row, task, tokenizer, max_len)

    return {
        split: ds.map(convert_single, remove_columns=ds.column_names)
        for split, ds in original_ds.items()
    }


def get_num_labels(task):
    """Return number of labels for a task."""
    if task in LRA_TASK_INFO:
        return LRA_TASK_INFO[task]["num_labels"]
    if task in RULER_TASK_INFO:
        return RULER_TASK_INFO[task]["num_labels"]
    raise ValueError(f"Unknown task: {task}")
