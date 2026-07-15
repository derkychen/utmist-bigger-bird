"""RULER-style synthetic long-context datasets for the modern eval track.

Adapted from the NVIDIA RULER / needle-in-a-haystack family for encoder-only
classification (no generative decoding). Each example buries one or more passkeys
in a long filler haystack; the model must classify the retrieved value from the
[CLS] pooled representation.

Tasks
-----
- ``niah``    : single passkey buried at a configurable depth fraction (10-way digit).
- ``mq_niah`` : two passkeys (KEY_ALPHA / KEY_BETA); classify KEY_ALPHA's digit.

Byte-level tokenization matches the LRA Text track (vocab size 260). Filler text
is drawn from a fixed pool of noise sentences repeated to fill the context window.
"""

import random

from datasets import Dataset

from shared.lra_dataset import BYTE_VOCAB_SIZE, CLS_ID, NUM_SPECIAL, PAD_ID, _pad_ids

TASK_INFO = {
    "niah": {"num_labels": 10, "pair": False},
    "mq_niah": {"num_labels": 10, "pair": False},
}

# Noise sentences for the haystack (RULER-style distractor text).
_FILLER = [
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ",
    "A quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump. Bright vixens jump; dozy fowl quack. ",
    "The five boxing wizards jump quickly. Sphinx of black quartz, judge my vow. ",
    "Waltz, bad nymph, for quick jigs vex. Glib jocks quiz nymph to vex dwarf. ",
    "Jackdaws love my big sphinx of quartz. The job requires extra pluck and zeal. ",
    "All questions asked by five watched experts amaze the judge. ",
    "We promptly judged antique ivory buckles for the next prize. ",
]


def _bytes_to_ids(text: str) -> list:
    raw = text.encode("utf-8", errors="ignore")
    return [NUM_SPECIAL + b for b in raw]


def _build_haystack(rng: random.Random, content_budget: int) -> list:
    """Return a list of byte token ids filling ``content_budget`` slots (excludes [CLS])."""
    ids = []
    while len(ids) < content_budget:
        ids.extend(_bytes_to_ids(rng.choice(_FILLER)))
    return ids[:content_budget]


def _insert_needle(content_ids: list, needle_ids: list, depth_frac: float) -> list:
    """Insert ``needle_ids`` into ``content_ids`` at ``depth_frac`` (0=start, 1=end)."""
    if not needle_ids:
        return content_ids
    max_start = max(0, len(content_ids) - len(needle_ids))
    start = int(depth_frac * max_start)
    out = content_ids[:start] + needle_ids + content_ids[start + len(needle_ids) :]
    return out[: len(content_ids)]


def _niah_example(rng: random.Random, seq_len: int, depth_frac: float):
    digit = rng.randint(0, 9)
    needle = f" The secret passkey is: {digit}. "
    needle_ids = _bytes_to_ids(needle)
    budget = seq_len - 1  # leave room for [CLS]
    content = _build_haystack(rng, budget)
    content = _insert_needle(content, needle_ids, depth_frac)
    input_ids, attn = _pad_ids(content, seq_len)
    return input_ids, attn, digit


def _mq_niah_example(rng: random.Random, seq_len: int, depth_frac: float):
    """Two needles; label is KEY_ALPHA's digit (tests selective retrieval)."""
    alpha = rng.randint(0, 9)
    beta = rng.randint(0, 9)
    while beta == alpha:
        beta = rng.randint(0, 9)
    needle_a = f" KEY_ALPHA holds: {alpha}. "
    needle_b = f" KEY_BETA holds: {beta}. "
    needle_a_ids = _bytes_to_ids(needle_a)
    needle_b_ids = _bytes_to_ids(needle_b)
    budget = seq_len - 1
    content = _build_haystack(rng, budget)
    # Place ALPHA at depth_frac; BETA at a different depth (offset by ~30% of context).
    content = _insert_needle(content, needle_a_ids, depth_frac)
    beta_depth = min(1.0, max(0.0, depth_frac + 0.3))
    content = _insert_needle(content, needle_b_ids, beta_depth)
    input_ids, attn = _pad_ids(content, seq_len)
    return input_ids, attn, alpha


def _build_split(task, seq_len, depth_frac, n_samples, seed):
    rng = random.Random(seed)
    builder = _niah_example if task == "niah" else _mq_niah_example
    rows = []
    for _ in range(n_samples):
        ids, attn, label = builder(rng, seq_len, depth_frac)
        rows.append({"input_ids": ids, "attention_mask": attn, "labels": label})
    return Dataset.from_list(rows).with_format("torch")


def build_ruler_dataset(
    task: str,
    seq_len: int,
    needle_depth: float = 0.5,
    train_samples: int = 1000,
    eval_samples: int = 200,
    seed: int = 42,
):
    """Build train/validation splits for a RULER-style task.

    Args:
        task: ``niah`` or ``mq_niah``.
        seq_len: Fixed context window (includes [CLS]).
        needle_depth: Relative insertion point in 0..1 (0=near start, 1=near end).
        train_samples: Training examples (unique random seeds per row).
        eval_samples: Validation examples.
        seed: Base RNG seed.

    Returns:
        dict with ``train``, ``validation``, ``vocab_size``, ``num_labels``, ``pair``.
    """
    if task not in TASK_INFO:
        raise ValueError(f"Unknown RULER task {task!r}; choose from {list(TASK_INFO)}")

    info = TASK_INFO[task]
    depth = float(max(0.0, min(1.0, needle_depth)))
    train = _build_split(task, seq_len, depth, train_samples, seed)
    val = _build_split(task, seq_len, depth, eval_samples, seed + 1_000_003)

    return {
        "train": train,
        "validation": val,
        "vocab_size": BYTE_VOCAB_SIZE,
        "num_labels": info["num_labels"],
        "pair": info["pair"],
        "needle_depth": depth,
    }
