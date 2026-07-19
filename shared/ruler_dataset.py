"""Full RULER-style synthetic long-context datasets (encoder-only classification).

Adapted from NVIDIA RULER (Hsieh et al., 2024) for encoder-only classification
(no generative decoding). The official 13-task suite is covered; each example is
reduced to a fixed-length integer-id sequence with a 10-way (or binary) label
read out from the [CLS] pooled representation.

Official tasks (``scripts/synthetic.yaml`` names)
-------------------------------------------------
Retrieval (NIAH family)
  - ``niah_single_1``   : noise haystack, word key, number value
  - ``niah_single_2``   : essay haystack, word key, number value
  - ``niah_single_3``   : essay haystack, word key, uuid value (label = last digit)
  - ``niah_multikey_1`` : essay, 4 keys / retrieve 1
  - ``niah_multikey_2`` : needle-distractor haystack, 1 key
  - ``niah_multikey_3`` : needle haystack, uuid key/value
  - ``niah_multivalue`` : 1 key with 4 values; label = sum of digits mod 10
  - ``niah_multiquery`` : 4 keys queried; label = sum of values mod 10

Multi-hop tracing
  - ``vt``              : variable-tracking chain; label = final bound digit

Aggregation
  - ``cwe``             : common-words extraction; label = id of a common word
  - ``fwe``             : frequent-words (Zeta); label = id of the top word

Question answering
  - ``qa_1``            : single-hop synthetic QA (SQuAD-style); digit answer
  - ``qa_2``            : multi-hop synthetic QA (Hotpot-style); digit answer

Backward-compatible aliases: ``niah`` → ``niah_single_1``, ``mq_niah`` → ``niah_multikey_1``.
"""

from __future__ import annotations

import random
import string
import uuid
from collections import Counter

from datasets import Dataset

from shared.lra_dataset import BYTE_VOCAB_SIZE, NUM_SPECIAL, _pad_ids

# --------------------------------------------------------------------------------------
# Task registry
# --------------------------------------------------------------------------------------

_NIAH_LABELS = 10
_WORD_VOCAB = 10  # synthetic word ids 0..9 used as classification targets for CWE/FWE

TASK_INFO = {
    # Official RULER names
    "niah_single_1": {"num_labels": _NIAH_LABELS, "pair": False, "uses_depth": True},
    "niah_single_2": {"num_labels": _NIAH_LABELS, "pair": False, "uses_depth": True},
    "niah_single_3": {"num_labels": _NIAH_LABELS, "pair": False, "uses_depth": True},
    "niah_multikey_1": {"num_labels": _NIAH_LABELS, "pair": False, "uses_depth": True},
    "niah_multikey_2": {"num_labels": _NIAH_LABELS, "pair": False, "uses_depth": True},
    "niah_multikey_3": {"num_labels": _NIAH_LABELS, "pair": False, "uses_depth": True},
    "niah_multivalue": {"num_labels": _NIAH_LABELS, "pair": False, "uses_depth": True},
    "niah_multiquery": {"num_labels": _NIAH_LABELS, "pair": False, "uses_depth": True},
    "vt": {"num_labels": _NIAH_LABELS, "pair": False, "uses_depth": True},
    "cwe": {"num_labels": _WORD_VOCAB, "pair": False, "uses_depth": False},
    "fwe": {"num_labels": _WORD_VOCAB, "pair": False, "uses_depth": False},
    "qa_1": {"num_labels": _NIAH_LABELS, "pair": False, "uses_depth": True},
    "qa_2": {"num_labels": _NIAH_LABELS, "pair": False, "uses_depth": True},
    # Aliases
    "niah": {"num_labels": _NIAH_LABELS, "pair": False, "uses_depth": True},
    "mq_niah": {"num_labels": _NIAH_LABELS, "pair": False, "uses_depth": True},
}

# Resolve aliases to canonical builders.
_TASK_ALIAS = {
    "niah": "niah_single_1",
    "mq_niah": "niah_multikey_1",
}

OFFICIAL_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]

# Noise / essay / needle haystack corpora (RULER-style distractors).
_NOISE_FILLER = [
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ",
    "A quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump. Bright vixens jump; dozy fowl quack. ",
    "The five boxing wizards jump quickly. Sphinx of black quartz, judge my vow. ",
    "Waltz, bad nymph, for quick jigs vex. Glib jocks quiz nymph to vex dwarf. ",
    "Jackdaws love my big sphinx of quartz. The job requires extra pluck and zeal. ",
    "All questions asked by five watched experts amaze the judge. ",
    "We promptly judged antique ivory buckles for the next prize. ",
]

# Essay-like paragraphs (stand-in for Paul Graham essays used in official RULER).
_ESSAY_FILLER = [
    "One of the most surprising things I discovered while working on startups is how much "
    "the quality of the people matters. In most organizations you can get away with mediocre "
    "colleagues for a while, but in a small team every hire compounds. ",
    "Writing is thinking. When you force yourself to explain an idea clearly, you often find "
    "that the idea was not as solid as it seemed. The blank page is an unforgiving critic. ",
    "Cities are machines for serendipity. Density creates collisions between people who would "
    "never otherwise meet, and those collisions are the raw material of new companies. ",
    "The best way to get startup ideas is not to try to think of startup ideas. It is to look "
    "for problems, preferably problems you have yourself. ",
    "Technology is a lever. A small number of people can create tools that amplify the work "
    "of millions. That asymmetry is why software companies can grow so quickly. ",
    "Curiosity compounds. The more you know, the more you notice, and the more you notice, "
    "the more hooks you have for new knowledge to attach to. ",
]

_ADJECTIVES = [
    "emerald", "scarlet", "azure", "golden", "silver", "violet", "crimson", "amber",
    "ivory", "jade", "coral", "indigo", "bronze", "pearl", "onyx", "ruby",
]
_NOUNS = [
    "falcon", "willow", "harbor", "lantern", "canyon", "meadow", "comet", "harbor",
    "nexus", "orchid", "quasar", "raven", "summit", "thistle", "umbra", "vortex",
]


def task_uses_depth(task: str) -> bool:
    info = TASK_INFO.get(task)
    return bool(info and info.get("uses_depth", False))


def _canonical(task: str) -> str:
    return _TASK_ALIAS.get(task, task)


def _bytes_to_ids(text: str) -> list:
    raw = text.encode("utf-8", errors="ignore")
    return [NUM_SPECIAL + b for b in raw]


def _word_key(rng: random.Random) -> str:
    return f"{rng.choice(_ADJECTIVES)}-{rng.choice(_NOUNS)}"


def _number_value(rng: random.Random) -> tuple[str, int]:
    """7-digit-ish number; classification label is last digit."""
    n = rng.randint(1000000, 9999999)
    return str(n), n % 10


def _uuid_value(rng: random.Random) -> tuple[str, int]:
    # Deterministic-ish uuid from rng bits.
    u = uuid.UUID(int=rng.getrandbits(128))
    s = str(u)
    digit = int(s[-1], 16) % 10
    return s, digit


def _build_haystack(rng: random.Random, content_budget: int, kind: str = "noise") -> list:
    if kind == "essay":
        pool = _ESSAY_FILLER
    elif kind == "needle":
        # Continuous stream of distractor key/value needles (multikey_2/3 style).
        ids = []
        while len(ids) < content_budget:
            distractor = (
                f" The special magic {_word_key(rng)} is: {rng.randint(1000000, 9999999)}. "
            )
            ids.extend(_bytes_to_ids(distractor))
        return ids[:content_budget]
    else:
        pool = _NOISE_FILLER
    ids = []
    while len(ids) < content_budget:
        ids.extend(_bytes_to_ids(rng.choice(pool)))
    return ids[:content_budget]


def _insert_needle(content_ids: list, needle_ids: list, depth_frac: float) -> list:
    if not needle_ids:
        return content_ids
    max_start = max(0, len(content_ids) - len(needle_ids))
    start = int(depth_frac * max_start)
    out = content_ids[:start] + needle_ids + content_ids[start + len(needle_ids) :]
    return out[: len(content_ids)]


def _spread_depths(base: float, n: int) -> list[float]:
    """Spread n insertion depths around ``base``, clamped to [0, 1]."""
    if n <= 1:
        return [base]
    depths = []
    for i in range(n):
        # Evenly cover the context, biased toward base for the queried needle (i=0).
        if i == 0:
            depths.append(base)
        else:
            depths.append(min(1.0, max(0.0, (i / n + base * 0.15) % 1.0)))
    return depths


# --------------------------------------------------------------------------------------
# NIAH family
# --------------------------------------------------------------------------------------

def _niah_core(
    rng: random.Random,
    seq_len: int,
    depth_frac: float,
    *,
    haystack: str,
    key_type: str,
    value_type: str,
    num_keys: int,
    num_values: int,
    num_queries: int,
):
    """Shared NIAH builder covering single/multi key/value/query variants."""
    budget = seq_len - 1
    content = _build_haystack(rng, budget, kind=haystack)

    def make_key():
        if key_type == "uuids":
            return str(uuid.UUID(int=rng.getrandbits(128)))
        return _word_key(rng)

    def make_val():
        if value_type == "uuids":
            return _uuid_value(rng)
        return _number_value(rng)

    # Create key -> list[values]
    keys = []
    kv = {}
    for _ in range(max(num_keys, num_queries)):
        k = make_key()
        while k in kv:
            k = make_key()
        vals = [make_val() for _ in range(num_values)]
        kv[k] = vals
        keys.append(k)

    depths = _spread_depths(depth_frac, len(keys))
    for k, d in zip(keys, depths):
        for vi, (vstr, _) in enumerate(kv[k]):
            # Slight per-value offset so multi-value needles do not fully overwrite.
            vd = min(1.0, max(0.0, d + 0.02 * vi))
            needle = f" The special magic {k} is: {vstr}. "
            content = _insert_needle(content, _bytes_to_ids(needle), vd)

    query_keys = keys[:num_queries]
    # Label: for single query/single value → that digit; for multi → sum mod 10.
    digits = []
    for k in query_keys:
        for _, dig in kv[k]:
            digits.append(dig)
    label = digits[0] if len(digits) == 1 else (sum(digits) % 10)

    # Append a query cue at the end so the model knows which key(s) to retrieve.
    if num_queries == 1 and num_values == 1:
        cue = f" What is the special magic {query_keys[0]}? "
    elif num_values > 1:
        cue = f" What are all values for the special magic {query_keys[0]}? "
    else:
        joined = ", ".join(query_keys)
        cue = f" What are the special magics for: {joined}? "
    content = _insert_needle(content, _bytes_to_ids(cue), 0.98)

    input_ids, attn = _pad_ids(content, seq_len)
    return input_ids, attn, label


_NIAH_SPECS = {
    "niah_single_1": dict(haystack="noise", key_type="words", value_type="numbers",
                          num_keys=1, num_values=1, num_queries=1),
    "niah_single_2": dict(haystack="essay", key_type="words", value_type="numbers",
                          num_keys=1, num_values=1, num_queries=1),
    "niah_single_3": dict(haystack="essay", key_type="words", value_type="uuids",
                          num_keys=1, num_values=1, num_queries=1),
    "niah_multikey_1": dict(haystack="essay", key_type="words", value_type="numbers",
                            num_keys=4, num_values=1, num_queries=1),
    "niah_multikey_2": dict(haystack="needle", key_type="words", value_type="numbers",
                            num_keys=1, num_values=1, num_queries=1),
    "niah_multikey_3": dict(haystack="needle", key_type="uuids", value_type="uuids",
                            num_keys=1, num_values=1, num_queries=1),
    "niah_multivalue": dict(haystack="essay", key_type="words", value_type="numbers",
                            num_keys=1, num_values=4, num_queries=1),
    "niah_multiquery": dict(haystack="essay", key_type="words", value_type="numbers",
                            num_keys=4, num_values=1, num_queries=4),
}


# --------------------------------------------------------------------------------------
# Variable tracking (VT)
# --------------------------------------------------------------------------------------

def _vt_example(rng: random.Random, seq_len: int, depth_frac: float, num_hops: int = 4):
    """Chain of assignments VAR_i = VAR_{i-1}; label is the root digit."""
    budget = seq_len - 1
    content = _build_haystack(rng, budget, kind="noise")
    root = rng.randint(0, 9)
    names = [f"VAR_{chr(ord('A') + i)}" for i in range(num_hops + 1)]
    stmts = [f" {names[0]} = {root}. "]
    for i in range(num_hops):
        stmts.append(f" {names[i + 1]} = {names[i]}. ")
    depths = _spread_depths(depth_frac, len(stmts))
    for stmt, d in zip(stmts, depths):
        content = _insert_needle(content, _bytes_to_ids(stmt), d)
    cue = f" What is the value of {names[-1]}? "
    content = _insert_needle(content, _bytes_to_ids(cue), 0.98)
    input_ids, attn = _pad_ids(content, seq_len)
    return input_ids, attn, root


# --------------------------------------------------------------------------------------
# Aggregation: CWE / FWE
# --------------------------------------------------------------------------------------

_SYN_WORDS = [f"WORD{i}" for i in range(_WORD_VOCAB)]


def _cwe_example(rng: random.Random, seq_len: int, freq_cw: int = 30, freq_ucw: int = 3, num_cw: int = 3):
    """Common-words extraction: label is the id of one common word present in the bag."""
    budget = seq_len - 1
    # Choose common and uncommon words from the synthetic vocab.
    common = rng.sample(range(_WORD_VOCAB), k=min(num_cw, _WORD_VOCAB))
    uncommon = [i for i in range(_WORD_VOCAB) if i not in common]
    bag = []
    for i in common:
        bag.extend([_SYN_WORDS[i]] * freq_cw)
    # Fill remaining budget-ish with uncommon words at low frequency.
    while len(" ".join(bag)) < budget * 0.8 and uncommon:
        u = rng.choice(uncommon)
        bag.extend([_SYN_WORDS[u]] * freq_ucw)
        if len(bag) > budget // 2:
            break
    rng.shuffle(bag)
    text = " " + " ".join(bag) + " "
    # Ask which of a listed candidate is common; answer is that word's id.
    query_word = rng.choice(common)
    text += f" Which word is common: {_SYN_WORDS[query_word]}? "
    ids = _bytes_to_ids(text)[:budget]
    if len(ids) < budget:
        ids = ids + _bytes_to_ids(" pad") * ((budget - len(ids)) // 4 + 1)
    ids = ids[:budget]
    input_ids, attn = _pad_ids(ids, seq_len)
    return input_ids, attn, query_word


def _zeta_sample(rng: random.Random, alpha: float, n_words: int) -> int:
    """Sample an index in 0..n_words-1 from a truncated Zeta(alpha) distribution."""
    weights = [(i + 1) ** (-alpha) for i in range(n_words)]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return n_words - 1


def _fwe_example(rng: random.Random, seq_len: int, alpha: float = 2.0):
    """Frequent-words extraction: label is the empirical top-1 word id."""
    budget = seq_len - 1
    # Draw enough tokens to fill the window.
    tokens = []
    target_chars = int(budget * 0.85)
    while sum(len(t) + 1 for t in tokens) < target_chars:
        tokens.append(_SYN_WORDS[_zeta_sample(rng, alpha, _WORD_VOCAB)])
    counts = Counter(tokens)
    top_word, _ = counts.most_common(1)[0]
    label = _SYN_WORDS.index(top_word)
    text = " " + " ".join(tokens) + f" What is the most frequent word? "
    ids = _bytes_to_ids(text)[:budget]
    if len(ids) < budget:
        ids = ids + _bytes_to_ids(" .") * (budget - len(ids))
        ids = ids[:budget]
    input_ids, attn = _pad_ids(ids, seq_len)
    return input_ids, attn, label


# --------------------------------------------------------------------------------------
# QA (synthetic single-hop / multi-hop)
# --------------------------------------------------------------------------------------

_QA_FACTS = [
    ("City of Aurora", "population", None),
    ("Lake Meridian", "depth_meters", None),
    ("Mount Cinder", "elevation_km", None),
    ("Bridge of Elders", "length_meters", None),
    ("Museum of Dawn", "founding_year_mod", None),
]


def _qa1_example(rng: random.Random, seq_len: int, depth_frac: float):
    """Single-hop QA: one supporting fact buried in filler; answer is a digit."""
    budget = seq_len - 1
    content = _build_haystack(rng, budget, kind="essay")
    entity, attr, _ = rng.choice(_QA_FACTS)
    answer = rng.randint(0, 9)
    fact = f" The {attr.replace('_', ' ')} of {entity} is {answer}. "
    content = _insert_needle(content, _bytes_to_ids(fact), depth_frac)
    # Distractor facts
    for _ in range(3):
        e2, a2, _ = rng.choice(_QA_FACTS)
        if e2 == entity and a2 == attr:
            continue
        d = rng.uniform(0.05, 0.95)
        content = _insert_needle(
            content,
            _bytes_to_ids(f" The {a2.replace('_', ' ')} of {e2} is {rng.randint(0, 9)}. "),
            d,
        )
    cue = f" What is the {attr.replace('_', ' ')} of {entity}? "
    content = _insert_needle(content, _bytes_to_ids(cue), 0.98)
    input_ids, attn = _pad_ids(content, seq_len)
    return input_ids, attn, answer


def _qa2_example(rng: random.Random, seq_len: int, depth_frac: float):
    """Multi-hop QA: bridge entity then attribute; answer is a digit."""
    budget = seq_len - 1
    content = _build_haystack(rng, budget, kind="essay")
    person = f"Person_{rng.choice(string.ascii_uppercase)}"
    city = rng.choice(["Aurora", "Meridian", "Cinderfall", "Eldergate", "Dawnport"])
    answer = rng.randint(0, 9)
    hop1 = f" {person} lives in the city of {city}. "
    hop2 = f" The secret code of the city of {city} is {answer}. "
    content = _insert_needle(content, _bytes_to_ids(hop1), depth_frac)
    content = _insert_needle(content, _bytes_to_ids(hop2), min(1.0, depth_frac + 0.35))
    # Distractors
    for _ in range(3):
        other_city = rng.choice(["Northbay", "Southfen", "Westmoor", "Eastmere"])
        content = _insert_needle(
            content,
            _bytes_to_ids(f" The secret code of the city of {other_city} is {rng.randint(0, 9)}. "),
            rng.uniform(0.05, 0.95),
        )
    cue = f" What is the secret code of the city where {person} lives? "
    content = _insert_needle(content, _bytes_to_ids(cue), 0.98)
    input_ids, attn = _pad_ids(content, seq_len)
    return input_ids, attn, answer


# --------------------------------------------------------------------------------------
# Dispatch / public API
# --------------------------------------------------------------------------------------

def _build_example(task: str, rng: random.Random, seq_len: int, depth_frac: float):
    task = _canonical(task)
    if task in _NIAH_SPECS:
        return _niah_core(rng, seq_len, depth_frac, **_NIAH_SPECS[task])
    if task == "vt":
        return _vt_example(rng, seq_len, depth_frac)
    if task == "cwe":
        return _cwe_example(rng, seq_len)
    if task == "fwe":
        return _fwe_example(rng, seq_len)
    if task == "qa_1":
        return _qa1_example(rng, seq_len, depth_frac)
    if task == "qa_2":
        return _qa2_example(rng, seq_len, depth_frac)
    raise ValueError(f"No builder for RULER task {task!r}")


def _build_split(task, seq_len, depth_frac, n_samples, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n_samples):
        ids, attn, label = _build_example(task, rng, seq_len, depth_frac)
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
    """Build train/validation splits for a RULER task (official name or alias).

    Returns:
        dict with ``train``, ``validation``, ``vocab_size``, ``num_labels``, ``pair``,
        ``needle_depth``, ``canonical_task``.
    """
    if task not in TASK_INFO:
        raise ValueError(
            f"Unknown RULER task {task!r}; choose from {list(TASK_INFO)} "
            f"(official: {OFFICIAL_TASKS})"
        )

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
        "canonical_task": _canonical(task),
    }
