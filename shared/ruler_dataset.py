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

Backward-compatible aliases: ``niah`` → ``niah_single_1``.
``mq_niah`` is a dedicated two-key selective-retrieval task (KEY_ALPHA / KEY_BETA).
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
    # Aliases / legacy
    "niah": {"num_labels": _NIAH_LABELS, "pair": False, "uses_depth": True},
    "mq_niah": {"num_labels": _NIAH_LABELS, "pair": False, "uses_depth": True},
}

# Resolve aliases to canonical builders (mq_niah has its own builder).
_TASK_ALIAS = {
    "niah": "niah_single_1",
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
    """UUID-like string whose classification label is an embedded decimal digit 0..9."""
    digit = rng.randint(0, 9)
    u = uuid.UUID(int=rng.getrandbits(128))
    # Replace the last hex char with a decimal digit so the label is unambiguous.
    s = str(u)[:-1] + str(digit)
    return s, digit


def _build_haystack(rng: random.Random, content_budget: int, kind: str = "noise") -> list:
    if kind == "essay":
        pool = _ESSAY_FILLER
    elif kind == "needle":
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


def _overlaps(start: int, length: int, occupied: list[tuple[int, int]]) -> bool:
    end = start + length
    for a, b in occupied:
        if not (end <= a or start >= b):
            return True
    return False


def _place_nonoverlapping(
    content_ids: list,
    items: list[tuple[list, float]],
    *,
    tail_ids: list | None = None,
) -> list:
    """Place needles at preferred depths without overwriting each other.

    ``items`` is a list of ``(needle_ids, depth_frac)``. If ``tail_ids`` is set, that
    span is reserved at the end of the window (e.g. a query cue) and written last.
    """
    out = list(content_ids)
    n = len(out)
    reserved_tail = len(tail_ids) if tail_ids else 0
    usable = max(0, n - reserved_tail)
    occupied: list[tuple[int, int]] = []

    order = sorted(range(len(items)), key=lambda i: -len(items[i][0]))
    for idx in order:
        needle_ids, depth_frac = items[idx]
        if not needle_ids:
            continue
        length = len(needle_ids)
        if length > usable:
            length = usable
            needle_ids = needle_ids[:length]
        max_start = max(0, usable - length)
        preferred = int(max(0.0, min(1.0, depth_frac)) * max_start)
        placed = False
        for delta in range(0, max_start + 1):
            for start in (preferred + delta, preferred - delta):
                if start < 0 or start > max_start:
                    continue
                if _overlaps(start, length, occupied):
                    continue
                out[start : start + length] = needle_ids
                occupied.append((start, start + length))
                placed = True
                break
            if placed:
                break
        if not placed and length > 0 and usable > 0:
            for start in range(0, max_start + 1):
                if not _overlaps(start, length, occupied):
                    out[start : start + length] = needle_ids
                    occupied.append((start, start + length))
                    placed = True
                    break
    if tail_ids:
        tlen = min(len(tail_ids), n)
        out[n - tlen :] = tail_ids[:tlen]
    return out


def _insert_needle(content_ids: list, needle_ids: list, depth_frac: float) -> list:
    """Single-needle convenience wrapper."""
    return _place_nonoverlapping(content_ids, [(needle_ids, depth_frac)])


def _spread_depths(base: float, n: int) -> list[float]:
    """Spread n depths across [0, 1]; index 0 stays near ``base`` (queried needle)."""
    if n <= 1:
        return [base]
    depths = [base]
    # Remaining needles: uniform grid, skipping a slot near ``base`` to reduce collision.
    others = [i / (n - 1) for i in range(n - 1)] if n > 1 else []
    # Rotate so the first "other" is farthest from base.
    others = sorted(others, key=lambda d: -abs(d - base))
    depths.extend(others[: n - 1])
    return [min(1.0, max(0.0, d)) for d in depths]


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

    keys = []
    kv = {}
    for _ in range(max(num_keys, num_queries)):
        k = make_key()
        while k in kv:
            k = make_key()
        vals = [make_val() for _ in range(num_values)]
        kv[k] = vals
        keys.append(k)

    query_keys = keys[:num_queries]
    if num_queries == 1 and num_values == 1:
        cue = f" What is the special magic {query_keys[0]}? "
    elif num_values > 1:
        cue = f" What are all values for the special magic {query_keys[0]}? "
    else:
        joined = ", ".join(query_keys)
        cue = f" What are the special magics for: {joined}? "
    cue_ids = _bytes_to_ids(cue)

    key_depths = _spread_depths(depth_frac, len(keys))
    items: list[tuple[list, float]] = []
    for k, d in zip(keys, key_depths):
        for vi, (vstr, _) in enumerate(kv[k]):
            # Spread multi-values evenly in a local band rather than 0.02 offsets.
            if num_values <= 1:
                vd = d
            else:
                band = 0.15
                vd = min(1.0, max(0.0, d - band / 2 + band * vi / max(1, num_values - 1)))
            needle = f" The special magic {k} is: {vstr}. "
            items.append((_bytes_to_ids(needle), vd))
    content = _place_nonoverlapping(content, items, tail_ids=cue_ids)

    digits = []
    for k in query_keys:
        for _, dig in kv[k]:
            digits.append(dig)
    label = digits[0] if len(digits) == 1 else (sum(digits) % 10)

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
# Variable tracking (VT) + legacy mq_niah
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
    cue = f" What is the value of {names[-1]}? "
    cue_ids = _bytes_to_ids(cue)
    depths = _spread_depths(depth_frac, len(stmts))
    items = [(_bytes_to_ids(s), d) for s, d in zip(stmts, depths)]
    content = _place_nonoverlapping(content, items, tail_ids=cue_ids)
    input_ids, attn = _pad_ids(content, seq_len)
    return input_ids, attn, root


def _mq_niah_example(rng: random.Random, seq_len: int, depth_frac: float):
    """Two needles; label is KEY_ALPHA's digit (selective retrieval)."""
    budget = seq_len - 1
    content = _build_haystack(rng, budget, kind="noise")
    alpha = rng.randint(0, 9)
    beta = rng.randint(0, 9)
    while beta == alpha:
        beta = rng.randint(0, 9)
    needle_a = _bytes_to_ids(f" KEY_ALPHA holds: {alpha}. ")
    needle_b = _bytes_to_ids(f" KEY_BETA holds: {beta}. ")
    cue = _bytes_to_ids(" What does KEY_ALPHA hold? ")
    beta_depth = min(1.0, max(0.0, depth_frac + 0.35 if depth_frac < 0.65 else depth_frac - 0.35))
    content = _place_nonoverlapping(
        content,
        [(needle_a, depth_frac), (needle_b, beta_depth)],
        tail_ids=cue,
    )
    input_ids, attn = _pad_ids(content, seq_len)
    return input_ids, attn, alpha


# --------------------------------------------------------------------------------------
# Aggregation: CWE / FWE
# --------------------------------------------------------------------------------------

_SYN_WORDS = [f"WORD{i}" for i in range(_WORD_VOCAB)]


def _encode_with_reserved_cue(bag_text: str, cue: str, seq_len: int) -> tuple[list, list]:
    """Encode bag + cue, always preserving the cue at the end of the content window."""
    budget = seq_len - 1
    cue_ids = _bytes_to_ids(cue)
    reserve = min(len(cue_ids), budget)
    bag_budget = max(0, budget - reserve)
    bag_ids = _bytes_to_ids(bag_text)[:bag_budget]
    if len(bag_ids) < bag_budget:
        pad = _bytes_to_ids(" .")
        while len(bag_ids) < bag_budget:
            bag_ids.extend(pad)
        bag_ids = bag_ids[:bag_budget]
    content = bag_ids + cue_ids[:reserve]
    content = content[:budget]
    if len(content) < budget:
        space_id = NUM_SPECIAL + ord(" ")
        content = content + [space_id] * (budget - len(content))
    return _pad_ids(content, seq_len)


def _cwe_example(rng: random.Random, seq_len: int, freq_cw: int = 30, freq_ucw: int = 3, num_cw: int = 3):
    """Common-words extraction without leaking the answer in the cue.

    The bag has common (high-freq) and uncommon (low-freq) words. The cue lists four
    candidate words (exactly one common); the label is that common word's id.
    """
    common = rng.sample(range(_WORD_VOCAB), k=min(num_cw, _WORD_VOCAB))
    uncommon = [i for i in range(_WORD_VOCAB) if i not in common]
    bag = []
    for i in common:
        bag.extend([_SYN_WORDS[i]] * freq_cw)
    # Fill with uncommon words at low frequency.
    fills = 0
    while uncommon and fills < 40:
        u = rng.choice(uncommon)
        bag.extend([_SYN_WORDS[u]] * freq_ucw)
        fills += 1
    rng.shuffle(bag)

    answer = rng.choice(common)
    # Three distractor candidates from uncommon (fall back to other commons if needed).
    pool = list(uncommon) if len(uncommon) >= 3 else [i for i in range(_WORD_VOCAB) if i != answer]
    distractors = rng.sample(pool, k=min(3, len(pool)))
    candidates = [answer] + distractors
    while len(candidates) < 4:
        candidates.append(rng.randint(0, _WORD_VOCAB - 1))
    rng.shuffle(candidates)
    cand_str = ", ".join(_SYN_WORDS[c] for c in candidates[:4])
    cue = f" Which of these words is common in the list: {cand_str}? "
    bag_text = " " + " ".join(bag) + " "
    input_ids, attn = _encode_with_reserved_cue(bag_text, cue, seq_len)
    return input_ids, attn, answer


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
    """Frequent-words extraction: label is the empirical top-1 word id.

    Each example uses a random permutation of word ranks so Zeta mass is not stuck
    on WORD0.
    """
    budget = seq_len - 1
    perm = list(range(_WORD_VOCAB))
    rng.shuffle(perm)
    tokens = []
    cue = " What is the most frequent word? "
    cue_ids = _bytes_to_ids(cue)
    reserve = min(len(cue_ids), budget)
    target_chars = max(16, int((budget - reserve) * 0.9))
    while sum(len(t) + 1 for t in tokens) < target_chars:
        rank = _zeta_sample(rng, alpha, _WORD_VOCAB)
        tokens.append(_SYN_WORDS[perm[rank]])
    counts = Counter(tokens)
    top_word, _ = counts.most_common(1)[0]
    label = _SYN_WORDS.index(top_word)
    bag_text = " " + " ".join(tokens) + " "
    input_ids, attn = _encode_with_reserved_cue(bag_text, cue, seq_len)
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
    cue = f" What is the {attr.replace('_', ' ')} of {entity}? "
    cue_ids = _bytes_to_ids(cue)
    items = [(_bytes_to_ids(fact), depth_frac)]
    for _ in range(3):
        e2, a2, _ = rng.choice(_QA_FACTS)
        if e2 == entity and a2 == attr:
            continue
        items.append((
            _bytes_to_ids(f" The {a2.replace('_', ' ')} of {e2} is {rng.randint(0, 9)}. "),
            rng.uniform(0.05, 0.95),
        ))
    content = _place_nonoverlapping(content, items, tail_ids=cue_ids)
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
    cue = f" What is the secret code of the city where {person} lives? "
    cue_ids = _bytes_to_ids(cue)
    hop2_depth = min(1.0, depth_frac + 0.35) if depth_frac < 0.65 else max(0.0, depth_frac - 0.35)
    items = [
        (_bytes_to_ids(hop1), depth_frac),
        (_bytes_to_ids(hop2), hop2_depth),
    ]
    for _ in range(3):
        other_city = rng.choice(["Northbay", "Southfen", "Westmoor", "Eastmere"])
        items.append((
            _bytes_to_ids(f" The secret code of the city of {other_city} is {rng.randint(0, 9)}. "),
            rng.uniform(0.05, 0.95),
        ))
    content = _place_nonoverlapping(content, items, tail_ids=cue_ids)
    input_ids, attn = _pad_ids(content, seq_len)
    return input_ids, attn, answer


# --------------------------------------------------------------------------------------
# Dispatch / public API
# --------------------------------------------------------------------------------------

def _build_example(task: str, rng: random.Random, seq_len: int, depth_frac: float):
    if task == "mq_niah":
        return _mq_niah_example(rng, seq_len, depth_frac)
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
        "canonical_task": _canonical(task) if task != "mq_niah" else "mq_niah",
        "protocol": "ruler-adapted-classification",
    }
