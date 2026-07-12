"""Long Range Arena (LRA) datasets for the long-context evaluation track.

Three tasks, all reduced to (optionally paired) fixed-length integer-id classification
so they can be hosted by a from-scratch BART-shaped encoder and trained with the same
Hugging Face Trainer machinery as the IMDb experiments:

- ``listops``  : 10-way classification of nested MAX/MIN/MED/SUM_MOD expressions
                 (self-contained generator following Nangia & Bowman, 2018).
- ``text``     : byte-level IMDb sentiment (binary). Reuses ``stanfordnlp/imdb`` so no
                 extra download is required -- the long-range signal comes from reading
                 the review one *byte* at a time at long ``seq_len``.
- ``retrieval``: byte-level document matching (binary, dual-tower). Built from the LRA
                 ACL-Anthology (AAN) ``label paper1_id paper2_id`` id files plus the
                 original AAN texts; raises a clear error if the data is not present.

Every split is returned as a ``datasets.Dataset`` already padded to ``seq_len`` and in
torch format, so the default data collator can stack rows directly.
"""

import os
import random
from statistics import median

from datasets import Dataset, load_dataset


# Shared special tokens (kept identical across tasks so the encoder config is uniform).
PAD_ID = 0
CLS_ID = 1
EOS_ID = 2
UNK_ID = 3
NUM_SPECIAL = 4

TASK_INFO = {
    "listops": {"num_labels": 10, "pair": False},
    "text": {"num_labels": 2, "pair": False},
    "retrieval": {"num_labels": 2, "pair": True},
}

# Byte-level tasks (text, retrieval) use one id per byte value (0..255) above the specials.
BYTE_VOCAB_SIZE = NUM_SPECIAL + 256


def _pad_ids(ids, seq_len):
    """Prepend [CLS], truncate to seq_len, right-pad, and build the attention mask."""
    ids = [CLS_ID] + list(ids)
    ids = ids[:seq_len]
    attn = [1] * len(ids) + [0] * (seq_len - len(ids))
    ids = ids + [PAD_ID] * (seq_len - len(ids))
    return ids, attn


def _encode_bytes(text, seq_len):
    raw = text.encode("utf-8", errors="ignore")[: seq_len - 1]
    return _pad_ids([NUM_SPECIAL + b for b in raw], seq_len)


# --------------------------------------------------------------------------------------
# ListOps
# --------------------------------------------------------------------------------------

_LISTOPS_OPS = ["[MAX", "[MIN", "[MED", "[SM"]
_LISTOPS_CLOSE = "]"
_LISTOPS_TOKENS = _LISTOPS_OPS + [_LISTOPS_CLOSE] + [str(d) for d in range(10)]
# id 4.. for listops vocab tokens
_LISTOPS_VOCAB = {tok: NUM_SPECIAL + i for i, tok in enumerate(_LISTOPS_TOKENS)}
LISTOPS_VOCAB_SIZE = NUM_SPECIAL + len(_LISTOPS_TOKENS)


def _listops_value(op, vals):
    if op == "[MAX":
        return max(vals)
    if op == "[MIN":
        return min(vals)
    if op == "[MED":
        return int(median(vals))  # floor of the median, stays in 0..9
    return sum(vals) % 10  # [SM = SUM_MOD


def _listops_tree(rng, max_depth, max_args, prob_op):
    """Return (token_list, value). Recurses to build a nested expression."""
    if max_depth <= 1 or rng.random() > prob_op:
        v = rng.randint(0, 9)
        return [str(v)], v
    op = rng.choice(_LISTOPS_OPS)
    n_args = rng.randint(2, max_args)
    toks, vals = [op], []
    for _ in range(n_args):
        sub_toks, sub_val = _listops_tree(rng, max_depth - 1, max_args, prob_op)
        toks.extend(sub_toks)
        vals.append(sub_val)
    toks.append(_LISTOPS_CLOSE)
    return toks, _listops_value(op, vals)


def _listops_example(rng, seq_len, max_args=5):
    """Grow a tree until its length is a reasonable fraction of seq_len."""
    target = max(8, int(0.6 * seq_len))
    depth = 4
    best = None
    for _ in range(40):
        toks, val = _listops_tree(rng, depth, max_args, prob_op=0.75)
        # account for the [CLS] slot when comparing to seq_len
        if len(toks) + 1 > seq_len:
            depth = max(2, depth - 1)
            continue
        best = (toks, val)
        if len(toks) >= target:
            break
        depth += 1
    if best is None:
        best = _listops_tree(rng, 2, max_args, prob_op=0.0)
    toks, val = best
    ids = [_LISTOPS_VOCAB[t] for t in toks]
    input_ids, attn = _pad_ids(ids, seq_len)
    return input_ids, attn, val


def _build_listops(seq_len, train_samples, eval_samples, seed):
    rng = random.Random(seed)

    def make(n):
        rows = {"input_ids": [], "attention_mask": [], "labels": []}
        for _ in range(n):
            ids, attn, label = _listops_example(rng, seq_len)
            rows["input_ids"].append(ids)
            rows["attention_mask"].append(attn)
            rows["labels"].append(label)
        return Dataset.from_dict(rows)

    train = make(train_samples)
    val = make(eval_samples)
    return train, val, LISTOPS_VOCAB_SIZE


# --------------------------------------------------------------------------------------
# Text (byte-level IMDb)
# --------------------------------------------------------------------------------------

def _build_text(seq_len, train_samples, eval_samples, seed):
    ds = load_dataset("stanfordnlp/imdb")
    train_raw = ds["train"].shuffle(seed=seed).select(range(min(train_samples, len(ds["train"]))))
    test_raw = ds["test"].shuffle(seed=seed).select(range(min(eval_samples, len(ds["test"]))))

    def convert(split):
        rows = {"input_ids": [], "attention_mask": [], "labels": []}
        for ex in split:
            ids, attn = _encode_bytes(ex["text"], seq_len)
            rows["input_ids"].append(ids)
            rows["attention_mask"].append(attn)
            rows["labels"].append(int(ex["label"]))
        return Dataset.from_dict(rows)

    return convert(train_raw), convert(test_raw), BYTE_VOCAB_SIZE


# --------------------------------------------------------------------------------------
# Retrieval (byte-level AAN document matching, dual-tower)
# --------------------------------------------------------------------------------------

def _load_aan_texts(data_dir):
    """Load a {paper_id: text} map from an AAN dump under ``data_dir``.

    Supported layouts:
      - ``<data_dir>/papers/<id>.txt`` (one file per paper), or
      - ``<data_dir>/aan_texts.tsv`` with ``id<TAB>text`` rows.
    """
    papers = {}
    tsv = os.path.join(data_dir, "aan_texts.tsv")
    papers_dir = os.path.join(data_dir, "papers")
    if os.path.isfile(tsv):
        with open(tsv, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    papers[parts[0]] = parts[1]
    elif os.path.isdir(papers_dir):
        for fname in os.listdir(papers_dir):
            if fname.endswith(".txt"):
                with open(os.path.join(papers_dir, fname), encoding="utf-8", errors="ignore") as f:
                    papers[fname[:-4]] = f.read()
    return papers


def _read_id_pairs(path, limit):
    pairs = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                pairs.append((int(parts[0]), parts[1], parts[2]))
            if limit and len(pairs) >= limit:
                break
    return pairs


def _build_retrieval(seq_len, train_samples, eval_samples, seed, data_dir):
    if not data_dir or not os.path.isdir(data_dir):
        raise FileNotFoundError(
            "Retrieval (AAN) data not found. Set --data-dir to an LRA retrieval directory "
            "containing 'new_aan_pairs.train.tsv'/'.eval.tsv' (id pairs) and either "
            "'papers/<id>.txt' files or 'aan_texts.tsv'. See scripts/get_lra_data.sh."
        )
    papers = _load_aan_texts(data_dir)
    if not papers:
        raise FileNotFoundError(
            f"No AAN paper texts found under {data_dir} (expected 'papers/*.txt' or 'aan_texts.tsv')."
        )

    def find_pairs(*names):
        for n in names:
            p = os.path.join(data_dir, n)
            if os.path.isfile(p):
                return p
        return None

    train_path = find_pairs("new_aan_pairs.train.tsv", "train.tsv", "retrieval.train.tsv")
    eval_path = find_pairs("new_aan_pairs.eval.tsv", "new_aan_pairs.test.tsv", "test.tsv", "retrieval.test.tsv")
    if train_path is None or eval_path is None:
        raise FileNotFoundError(f"Could not find AAN id-pair tsv files under {data_dir}.")

    def build(path, limit):
        rows = {
            "input_ids_a": [], "attention_mask_a": [],
            "input_ids_b": [], "attention_mask_b": [], "labels": [],
        }
        for label, id_a, id_b in _read_id_pairs(path, limit):
            if id_a not in papers or id_b not in papers:
                continue
            ia, aa = _encode_bytes(papers[id_a], seq_len)
            ib, ab = _encode_bytes(papers[id_b], seq_len)
            rows["input_ids_a"].append(ia)
            rows["attention_mask_a"].append(aa)
            rows["input_ids_b"].append(ib)
            rows["attention_mask_b"].append(ab)
            rows["labels"].append(int(label))
        if not rows["labels"]:
            raise FileNotFoundError(
                f"No usable retrieval pairs in {path} (paper ids did not match available texts)."
            )
        return Dataset.from_dict(rows)

    train = build(train_path, train_samples)
    val = build(eval_path, eval_samples)
    return train, val, BYTE_VOCAB_SIZE


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def build_lra_dataset(task, seq_len, train_samples, eval_samples, seed=42, data_dir=None):
    """Build an LRA task dataset.

    Returns a dict with keys: ``train``, ``validation`` (datasets.Dataset, torch format),
    ``vocab_size``, ``num_labels``, ``pair`` (True for dual-tower retrieval).
    """
    if task not in TASK_INFO:
        raise ValueError(f"Unknown LRA task '{task}'. Choose from {list(TASK_INFO)}.")

    if task == "listops":
        train, val, vocab_size = _build_listops(seq_len, train_samples, eval_samples, seed)
        cols = ["input_ids", "attention_mask", "labels"]
    elif task == "text":
        train, val, vocab_size = _build_text(seq_len, train_samples, eval_samples, seed)
        cols = ["input_ids", "attention_mask", "labels"]
    else:  # retrieval
        train, val, vocab_size = _build_retrieval(seq_len, train_samples, eval_samples, seed, data_dir)
        cols = ["input_ids_a", "attention_mask_a", "input_ids_b", "attention_mask_b", "labels"]

    train.set_format(type="torch", columns=cols)
    val.set_format(type="torch", columns=cols)
    return {
        "train": train,
        "validation": val,
        "vocab_size": vocab_size,
        "num_labels": TASK_INFO[task]["num_labels"],
        "pair": TASK_INFO[task]["pair"],
    }
