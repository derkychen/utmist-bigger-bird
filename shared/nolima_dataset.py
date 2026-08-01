"""NoLiMa long-context dataset adapted for encoder-only classification.

Official benchmark (Modarressi et al., ICML 2025):
  https://github.com/adobe-research/NoLiMa
  https://huggingface.co/datasets/amodaresi/NoLiMa

The original eval is generative LLM QA (return a character name). This project
hosts sparse-attention experiments as from-scratch sequence classifiers (same
as LRA / RULER), so we reduce each NoLiMa example to:

  input  = book haystack with a latent-association needle + retrieval question
           + an ``Options: 0=<name> 1=<name> …`` legend for the character pool
  label  = index of the answer character in the fixed 10-name character pool

Lexical overlap between question and needle stays minimal (the point of NoLiMa);
the model must attend to the needle without surface cues. Associations are
learnable from the train split (same needle-set distribution as eval), matching
how RULER trains from-scratch encoders on synthetic recall.

The options legend is what makes the label readable at all: the needle names the
character but the target is its *index*, so without the legend a from-scratch
byte-level encoder has to memorise ten spelling-to-index mappings and instead
collapses to the uniform prior. See ``_format_options``.

Download data first::

    bash scripts/get_nolima_data.sh
"""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from pathlib import Path

from datasets import Dataset

from shared.lra_dataset import BYTE_VOCAB_SIZE, NUM_SPECIAL, _pad_ids

# --------------------------------------------------------------------------------------
# Task registry
# --------------------------------------------------------------------------------------

_NUM_CHARS = 10  # official character_set size

TASK_INFO = {
    # Official needle_set.json, filtered by hop type
    "onehop": {"num_labels": _NUM_CHARS, "pair": False, "uses_depth": True, "needle_file": "needle_set.json", "hops": ("onehop",)},
    "twohop": {"num_labels": _NUM_CHARS, "pair": False, "uses_depth": True, "needle_file": "needle_set.json", "hops": ("twohop", "twohop2")},
    "all": {"num_labels": _NUM_CHARS, "pair": False, "uses_depth": True, "needle_file": "needle_set.json", "hops": None},
    # Hard subset (10 most challenging pairs)
    "hard": {"num_labels": _NUM_CHARS, "pair": False, "uses_depth": True, "needle_file": "needle_set_hard.json", "hops": None},
    # Ablation needle sets from the paper / HF release
    "direct": {"num_labels": _NUM_CHARS, "pair": False, "uses_depth": True, "needle_file": "needle_set_ONLYDirect.json", "hops": None},
    "mc": {"num_labels": _NUM_CHARS, "pair": False, "uses_depth": True, "needle_file": "needle_set_MC.json", "hops": None},
    "distractor": {"num_labels": _NUM_CHARS, "pair": False, "uses_depth": True, "needle_file": "needle_set_w_Distractor.json", "hops": None},
}

OFFICIAL_TASKS = ["onehop", "twohop", "all", "hard"]
ABLATION_TASKS = ["direct", "mc", "distractor"]

_TASK_TEMPLATE = (
    "You will answer a question based on the following book snippet:\n\n"
    "{haystack}\n\n"
    "Use the information provided in the book snippet to answer the question. "
    "Your answer should be short and based on either explicitly stated facts or "
    "strong, logical inferences.\n\n"
    "Question: {question}\n\n"
    "Return only the final answer with no additional explanation or reasoning."
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _REPO_ROOT / "nolima_data"


def _bytes_to_ids(text: str) -> list[int]:
    return [NUM_SPECIAL + b for b in text.encode("utf-8", errors="ignore")]


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    env = os.environ.get("NOLIMA_DATA_DIR")
    if env:
        return Path(env)
    return _DEFAULT_DATA_DIR


def _require_data(data_dir: Path) -> None:
    needles = data_dir / "needlesets"
    hay = data_dir / "haystack" / "rand_shuffle"
    if not needles.is_dir() or not hay.is_dir():
        raise FileNotFoundError(
            f"NoLiMa data not found under {data_dir}. Run:\n"
            f"  bash scripts/get_nolima_data.sh {data_dir}"
        )


@lru_cache(maxsize=8)
def _load_needle_groups(needle_path: str) -> tuple[dict, ...]:
    with open(needle_path, "r", encoding="utf-8") as f:
        groups = json.load(f)
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"Empty or invalid needle set at {needle_path}")
    return tuple(groups)


@lru_cache(maxsize=1)
def _load_haystacks(haystack_dir: str) -> tuple[str, ...]:
    paths = sorted(Path(haystack_dir).glob("rand_book_*.txt"))
    if not paths:
        raise FileNotFoundError(f"No rand_book_*.txt under {haystack_dir}")
    texts = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return tuple(texts)


def _expand_pairs(groups: tuple[dict, ...], hops: tuple[str, ...] | None) -> list[dict]:
    """Flatten needle groups into concrete (needle, question, character_set, …) pairs."""
    pairs = []
    for g in groups:
        char_set = list(g["character_set"])
        if len(char_set) != _NUM_CHARS:
            raise ValueError(
                f"Expected {_NUM_CHARS} characters in group {g.get('id')}, got {len(char_set)}"
            )
        for qtype, qtmpl in g["questions"].items():
            if hops is not None and qtype not in hops:
                continue
            for test_id, test in g["tests"].items():
                args = test["input_args"]
                needle = g["needle"]
                question = qtmpl
                distractor = None
                if "distractors" in g and qtype in g["distractors"]:
                    distractor = g["distractors"][qtype]
                for i, arg in enumerate(args):
                    ph = "{" + str(i + 1) + "}"
                    needle = needle.replace(ph, arg)
                    question = question.replace(ph, arg)
                    if distractor is not None:
                        distractor = distractor.replace(ph, arg)
                pairs.append({
                    "group_id": g["id"],
                    "test_id": test_id,
                    "hop": qtype,
                    "needle_tmpl": needle,  # still may contain {CHAR}
                    "question_tmpl": question,
                    "distractor_tmpl": distractor,
                    "character_set": char_set,
                    "task_template": g.get("task_template") or _TASK_TEMPLATE,
                })
    if not pairs:
        raise ValueError(f"No needle/question pairs after hop filter {hops!r}")
    return pairs


def _place_in_haystack(
    haystack: str,
    needle: str,
    *,
    char_budget: int,
    depth_frac: float,
    distractor: str | None = None,
    rng: random.Random,
) -> str:
    """Insert ``needle`` (and optional distractor) into a haystack window.

    Placement is newline-aware (official BookHaystack style) but sized by
    UTF-8 byte budget so it aligns with our byte-level encoder vocab.
    """
    # Over-sample a window large enough, then trim by bytes after insertion.
    lines = haystack.split("\n")
    if len(lines) < 4:
        raise ValueError("Haystack too short / malformed")

    # Pick a random starting line so different samples see different regions.
    max_start = max(0, len(lines) - 32)
    start_line = rng.randint(0, max_start) if max_start else 0
    # Accumulate lines until we have ~char_budget characters (bytes ≈ chars for English).
    chunk_lines: list[str] = []
    total = 0
    for line in lines[start_line:]:
        chunk_lines.append(line)
        total += len(line) + 1
        if total >= char_budget:
            break
    if not chunk_lines:
        chunk_lines = lines[:32]

    n_lines = len(chunk_lines)
    insert_at = int(max(0.0, min(1.0, depth_frac)) * max(n_lines - 1, 0))
    chunk_lines.insert(insert_at, needle)

    if distractor:
        # Keep distractor away from the needle (paper: distractor_free_zone ≈ 0.2).
        free = 0.2
        left = max(0.0, depth_frac - 2 * free)
        right = max(0.0, 1.0 - (depth_frac + 2 * free))
        span = left + right
        if span > 0:
            u = rng.random() * span
            dist_depth = u if u <= left else depth_frac + free + (u - left)
        else:
            dist_depth = 0.5 if abs(depth_frac - 0.5) > 0.2 else 0.1
        dist_at = int(max(0.0, min(1.0, dist_depth)) * max(len(chunk_lines) - 1, 0))
        # Avoid overwriting the needle line.
        if dist_at == insert_at:
            dist_at = min(len(chunk_lines), insert_at + 1)
        chunk_lines.insert(dist_at, distractor)

    text = "\n".join(chunk_lines)
    # Trim to budget while keeping the needle if possible.
    encoded = text.encode("utf-8", errors="ignore")
    if len(encoded) <= char_budget:
        return text
    needle_b = needle.encode("utf-8", errors="ignore")
    idx = encoded.find(needle_b)
    if idx < 0:
        return encoded[:char_budget].decode("utf-8", errors="ignore")
    # Keep a window centered on the needle.
    half = max(char_budget // 2, len(needle_b))
    left = max(0, idx - (char_budget - min(half, char_budget - len(needle_b))))
    right = left + char_budget
    if right > len(encoded):
        right = len(encoded)
        left = max(0, right - char_budget)
    return encoded[left:right].decode("utf-8", errors="ignore")


def _pack_ids(prompt: str, needle: str, seq_len: int) -> list[int]:
    """Encode ``prompt`` to byte ids, truncating while keeping needle + question when possible."""
    ids = _bytes_to_ids(prompt)
    budget = seq_len - 1  # CLS prepended in _pad_ids
    if len(ids) <= budget:
        return ids

    needle_ids = _bytes_to_ids(needle)
    # Default: keep the tail (question lives at the end of the template).
    tail = ids[-budget:]
    if not needle_ids:
        return tail

    # If the needle survived in the tail, we are done.
    for i in range(len(tail) - len(needle_ids) + 1):
        if tail[i : i + len(needle_ids)] == needle_ids:
            return tail

    # Otherwise slide a window that starts at the needle and still reaches the end
    # of the prompt when possible (so the question stays in-context).
    full = ids
    start = -1
    for i in range(len(full) - len(needle_ids) + 1):
        if full[i : i + len(needle_ids)] == needle_ids:
            start = i
            break
    if start < 0:
        return tail

    # Prefer a window ending at the prompt end (question). If that cannot cover the
    # needle, anchor on the needle instead.
    end = len(full)
    win_start = max(0, end - budget)
    if win_start <= start:
        return full[win_start:end]
    return full[start : start + budget]


def _format_options(char_set: list[str]) -> str:
    """Render the index-to-name legend that makes the label readable from context.

    The classification target is a position in ``character_set``, but the needle only
    ever contains the character's *name*. Without this legend a from-scratch byte-level
    encoder has to memorise ten spelling-to-index mappings from 32 needle templates,
    which it never does — it collapses to the uniform prior (loss ln 10). RULER stays
    learnable because its answer is a single digit copied straight out of the needle;
    this legend gives NoLiMa the same readability while leaving the latent-association
    needle, and therefore the retrieval difficulty, untouched.
    """
    return "Options: " + " ".join(f"{i}={n}" for i, n in enumerate(char_set))


def _build_example(
    pair: dict,
    haystack: str,
    rng: random.Random,
    seq_len: int,
    depth_frac: float,
) -> tuple[list[int], list[int], int]:
    char_set = pair["character_set"]
    char_idx = rng.randrange(len(char_set))
    char_name = char_set[char_idx]

    needle = pair["needle_tmpl"].replace("{CHAR}", char_name)
    question = pair["question_tmpl"].replace("{CHAR}", char_name)
    distractor = None
    if pair["distractor_tmpl"]:
        distractor = pair["distractor_tmpl"].replace("{CHAR}", char_name)

    # Question + legend are appended after truncation so the readout is never the part
    # that gets cut; only the haystack is squeezed to fit the context window.
    suffix = f"\n\nQuestion: {question}\n{_format_options(char_set)}\n"
    prefix = "Book snippet:\n\n"
    suffix_len = len(suffix.encode("utf-8", errors="ignore"))
    wrapper_overhead = len(prefix.encode()) + suffix_len
    hay_budget = max(64, seq_len - 1 - wrapper_overhead - len(needle.encode("utf-8", errors="ignore")))
    placed = _place_in_haystack(
        haystack,
        needle,
        char_budget=hay_budget,
        depth_frac=depth_frac,
        distractor=distractor,
        rng=rng,
    )
    # Prefer the official template when it fits; otherwise fall back to compact form.
    tmpl = pair["task_template"] or _TASK_TEMPLATE
    try:
        body = tmpl.format(haystack=placed, question=question)
    except (KeyError, ValueError):
        body = _TASK_TEMPLATE.format(haystack=placed, question=question)
    if len(_bytes_to_ids(body)) + suffix_len > seq_len - 1:
        body = f"{prefix}{placed}"

    body_ids = _pack_ids(body, needle, seq_len - suffix_len)
    ids = body_ids + _bytes_to_ids(suffix)
    input_ids, attn = _pad_ids(ids, seq_len)
    return input_ids, attn, char_idx

def _build_split(
    pairs: list[dict],
    haystacks: tuple[str, ...],
    seq_len: int,
    depth_frac: float,
    n_samples: int,
    seed: int,
) -> Dataset:
    rng = random.Random(seed)
    rows = []
    for i in range(n_samples):
        pair = pairs[i % len(pairs)]
        # Resample randomly after one full pass so train is not a strict cycle.
        if i >= len(pairs):
            pair = rng.choice(pairs)
        hay = haystacks[i % len(haystacks)]
        ids, attn, label = _build_example(pair, hay, rng, seq_len, depth_frac)
        rows.append({"input_ids": ids, "attention_mask": attn, "labels": label})
    return Dataset.from_list(rows).with_format("torch")


def build_nolima_dataset(
    task: str,
    seq_len: int,
    needle_depth: float = 0.5,
    train_samples: int = 1000,
    eval_samples: int = 200,
    seed: int = 42,
    data_dir: str | Path | None = None,
):
    """Build train/validation splits for a NoLiMa task.

    Returns:
        dict with ``train``, ``validation``, ``vocab_size``, ``num_labels``, ``pair``,
        ``needle_depth``, ``canonical_task``, ``num_pairs``.
    """
    if task not in TASK_INFO:
        raise ValueError(
            f"Unknown NoLiMa task {task!r}; choose from {list(TASK_INFO)}"
        )

    info = TASK_INFO[task]
    root = resolve_data_dir(data_dir)
    _require_data(root)

    needle_path = root / "needlesets" / info["needle_file"]
    if not needle_path.is_file():
        raise FileNotFoundError(
            f"Missing {needle_path}. Re-run: bash scripts/get_nolima_data.sh {root}"
        )

    groups = _load_needle_groups(str(needle_path))
    pairs = _expand_pairs(groups, info["hops"])
    haystacks = _load_haystacks(str(root / "haystack" / "rand_shuffle"))

    depth = float(max(0.0, min(1.0, needle_depth)))
    train = _build_split(pairs, haystacks, seq_len, depth, train_samples, seed)
    val = _build_split(pairs, haystacks, seq_len, depth, eval_samples, seed + 1_000_003)

    return {
        "train": train,
        "validation": val,
        "vocab_size": BYTE_VOCAB_SIZE,
        "num_labels": info["num_labels"],
        "pair": info["pair"],
        "needle_depth": depth,
        "canonical_task": task,
        "num_pairs": len(pairs),
        "data_dir": str(root),
    }
