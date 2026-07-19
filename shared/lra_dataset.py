"""Long Range Arena (LRA) datasets for the long-context evaluation track.

Full LRA suite, reduced to (optionally paired) fixed-length integer-id classification
so they can be hosted by a from-scratch BART-shaped encoder and trained with the same
Hugging Face Trainer machinery as the IMDb experiments:

- ``listops``      : 10-way nested MAX/MIN/MED/SUM_MOD (Nangia & Bowman–style generator).
- ``text``         : byte-level IMDb sentiment (binary) via ``stanfordnlp/imdb``.
- ``retrieval``    : byte-level AAN document matching (binary, dual-tower).
- ``image``        : CIFAR-10 grayscale pixel sequences (10-way).
- ``pathfinder``   : synthetic 32×32 path-connectivity (binary), generated on the fly.
- ``pathfinder_x`` : extreme-length Pathfinder (128×128 / ~16K), generated on the fly.

Every split is returned as a ``datasets.Dataset`` already padded to ``seq_len`` and in
torch format, so the default data collator can stack rows directly.
"""

import math
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
    "image": {"num_labels": 10, "pair": False},
    "pathfinder": {"num_labels": 2, "pair": False},
    "pathfinder_x": {"num_labels": 2, "pair": False},
}

# Byte-level / pixel-level tasks use one id per byte/intensity (0..255) above the specials.
BYTE_VOCAB_SIZE = NUM_SPECIAL + 256
PIXEL_VOCAB_SIZE = BYTE_VOCAB_SIZE


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
# Image (CIFAR-10 grayscale pixel sequence)
# --------------------------------------------------------------------------------------

def _rgb_to_gray(r, g, b):
    return int(0.299 * r + 0.587 * g + 0.114 * b)


def _resize_gray_nearest(pixels, src_side, dst_side):
    """Nearest-neighbor resize of a flat grayscale image."""
    if src_side == dst_side:
        return pixels
    out = []
    for y in range(dst_side):
        sy = min(src_side - 1, (y * src_side) // dst_side)
        for x in range(dst_side):
            sx = min(src_side - 1, (x * src_side) // dst_side)
            out.append(pixels[sy * src_side + sx])
    return out


def _encode_pixels(gray_pixels, seq_len):
    """Map 0..255 intensities to token ids and pad to ``seq_len`` (with [CLS])."""
    content_budget = max(1, seq_len - 1)
    raw = list(gray_pixels)[:content_budget]
    return _pad_ids([NUM_SPECIAL + int(p) for p in raw], seq_len)


def _image_side_for_seq(seq_len):
    """Largest square that fits in seq_len-1 content slots (LRA image is 32x32 by default)."""
    return max(1, int(math.isqrt(max(1, seq_len - 1))))


def _build_image(seq_len, train_samples, eval_samples, seed):
    """CIFAR-10 as a flattened grayscale pixel sequence (LRA Image task)."""
    ds = load_dataset("cifar10")
    side, _ = _canonical_visual_seq(seq_len, side_default=32)

    def convert(split, n):
        raw = split.shuffle(seed=seed).select(range(min(n, len(split))))
        rows = {"input_ids": [], "attention_mask": [], "labels": []}
        for ex in raw:
            img = ex["img"] if "img" in ex else ex["image"]
            if hasattr(img, "convert"):
                img = img.convert("RGB")
                w, h = img.size
                pix = list(img.getdata())
                gray = [_rgb_to_gray(r, g, b) for (r, g, b) in pix]
                src_side = w
            else:
                flat = []
                for row in img:
                    for r, g, b in row:
                        flat.append(_rgb_to_gray(r, g, b))
                src_side = int(len(flat) ** 0.5)
                gray = flat
            gray = _resize_gray_nearest(gray, src_side, side)
            ids, attn = _encode_pixels(gray, seq_len)
            rows["input_ids"].append(ids)
            rows["attention_mask"].append(attn)
            rows["labels"].append(int(ex["label"]))
        return Dataset.from_dict(rows)

    train = convert(ds["train"], train_samples)
    val = convert(ds["test"], eval_samples)
    return train, val, PIXEL_VOCAB_SIZE


# --------------------------------------------------------------------------------------
# Pathfinder / Pathfinder-X (synthetic long-range spatial connectivity)
# --------------------------------------------------------------------------------------

def _draw_disk(grid, cy, cx, radius, value=255):
    n = len(grid)
    r2 = radius * radius
    for y in range(max(0, cy - radius), min(n, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(n, cx + radius + 1)):
            if (y - cy) * (y - cy) + (x - cx) * (x - cx) <= r2:
                grid[y][x] = value


def _draw_dash(grid, y0, x0, y1, x1, thickness=1, value=220):
    """Draw a short thick segment (a 'paddle') between two points."""
    n = len(grid)
    steps = max(1, int(max(abs(y1 - y0), abs(x1 - x0))))
    for t in range(steps + 1):
        y = int(round(y0 + (y1 - y0) * t / steps))
        x = int(round(x0 + (x1 - x0) * t / steps))
        for dy in range(-thickness, thickness + 1):
            for dx in range(-thickness, thickness + 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < n and 0 <= xx < n:
                    grid[yy][xx] = value


def _random_walk_path(rng, n, length, start=None, gap=2):
    """Return a list of (y, x) waypoints forming a dashed contour."""
    margin = max(2, n // 10)
    if start is None:
        y = rng.randint(margin, n - 1 - margin)
        x = rng.randint(margin, n - 1 - margin)
    else:
        y, x = start
    pts = [(y, x)]
    angle = rng.uniform(0, 2 * math.pi)
    step = max(2, gap + 1)
    for _ in range(length - 1):
        angle += rng.uniform(-0.7, 0.7)
        ny = int(round(y + step * math.sin(angle)))
        nx = int(round(x + step * math.cos(angle)))
        ny = min(n - 1 - margin, max(margin, ny))
        nx = min(n - 1 - margin, max(margin, nx))
        if (ny, nx) == (y, x):
            angle += 1.2
            continue
        pts.append((ny, nx))
        y, x = ny, nx
    return pts


def _paint_dashed_path(grid, pts, thickness=1, dashed=True):
    """Paint a polyline; if ``dashed``, skip every other segment (paddle gaps)."""
    for i in range(len(pts) - 1):
        if dashed and (i % 2 == 1):
            continue
        _draw_dash(grid, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], thickness=thickness)


def _markers_connected(grid, p1, p2, thresh=80):
    """BFS connectivity on bright pixels (8-connected)."""
    n = len(grid)
    sy, sx = p1
    ty, tx = p2
    if grid[sy][sx] < thresh or grid[ty][tx] < thresh:
        return False
    seen = {(sy, sx)}
    stack = [(sy, sx)]
    while stack:
        y, x = stack.pop()
        if (y, x) == (ty, tx):
            return True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < n and 0 <= nx < n and (ny, nx) not in seen and grid[ny][nx] >= thresh:
                    seen.add((ny, nx))
                    stack.append((ny, nx))
    return False


def _pathfinder_example(rng, side, connected, n_distractors=None, contour_len=None):
    """Generate one Pathfinder-style binary image and label (connectivity-verified)."""
    if contour_len is None:
        contour_len = max(8, side // 2)
    if n_distractors is None:
        n_distractors = max(3, side // 8)
    thickness = 1 if side <= 64 else 2
    marker_r = 1 if side <= 32 else (2 if side <= 64 else 3)
    gap = 2 if side <= 64 else 3
    margin = max(3, side // 8)

    for _attempt in range(40):
        grid = [[0 for _ in range(side)] for _ in range(side)]
        p1 = (rng.randint(margin, side - 1 - margin), rng.randint(margin, side - 1 - margin))
        p2 = p1
        for _ in range(40):
            p2 = (rng.randint(margin, side - 1 - margin), rng.randint(margin, side - 1 - margin))
            if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) >= side * 0.35:
                break

        if connected:
            # Continuous target path so markers stay connected under BFS.
            path = _random_walk_path(rng, side, contour_len, start=p1, gap=gap)
            path[-1] = p2
            mid = len(path) // 2
            path[mid] = ((path[mid][0] + p2[0]) // 2, (path[mid][1] + p2[1]) // 2)
            _paint_dashed_path(grid, path, thickness=thickness, dashed=False)
        else:
            path_a = _random_walk_path(rng, side, contour_len // 2 + 1, start=p1, gap=gap)
            path_b = _random_walk_path(rng, side, contour_len // 2 + 1, start=p2, gap=gap)
            _paint_dashed_path(grid, path_a, thickness=thickness, dashed=True)
            _paint_dashed_path(grid, path_b, thickness=thickness, dashed=True)

        _draw_disk(grid, p1[0], p1[1], marker_r, value=255)
        _draw_disk(grid, p2[0], p2[1], marker_r, value=255)

        for _ in range(n_distractors):
            dpath = _random_walk_path(rng, side, max(4, contour_len // 3), gap=gap)
            _paint_dashed_path(grid, dpath, thickness=thickness, dashed=True)

        is_conn = _markers_connected(grid, p1, p2)
        if is_conn == bool(connected):
            pixels = [grid[y][x] for y in range(side) for x in range(side)]
            return pixels, int(connected)

    # Fallback: empty canvas with markers only (disconnected) or a straight link.
    grid = [[0 for _ in range(side)] for _ in range(side)]
    p1 = (margin, margin)
    p2 = (side - 1 - margin, side - 1 - margin)
    if connected:
        _paint_dashed_path(grid, [p1, p2], thickness=thickness, dashed=False)
    _draw_disk(grid, p1[0], p1[1], marker_r, value=255)
    _draw_disk(grid, p2[0], p2[1], marker_r, value=255)
    pixels = [grid[y][x] for y in range(side) for x in range(side)]
    return pixels, int(_markers_connected(grid, p1, p2))


def _canonical_visual_seq(seq_len, side_default):
    """Snap seq_len to side^2+1 for visual tasks (prefer ``side_default`` when it fits)."""
    if side_default * side_default + 1 <= seq_len:
        side = side_default
    else:
        side = max(8, _image_side_for_seq(seq_len))
    return side, side * side + 1


def _build_pathfinder(seq_len, train_samples, eval_samples, seed, side_default=32):
    """On-the-fly Pathfinder (or Path-X) pixel-sequence dataset."""
    side, _ = _canonical_visual_seq(seq_len, side_default)

    def make(n, base_seed):
        rng = random.Random(base_seed)
        rows = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(n):
            connected = (i % 2 == 0)
            local = random.Random(rng.randint(0, 2**31 - 1))
            pixels, label = _pathfinder_example(
                local,
                side,
                connected=connected,
                n_distractors=max(3, side // 8),
                contour_len=max(8, side // 2),
            )
            ids, attn = _encode_pixels(pixels, seq_len)
            rows["input_ids"].append(ids)
            rows["attention_mask"].append(attn)
            rows["labels"].append(label)
        return Dataset.from_dict(rows)

    train = make(train_samples, seed)
    val = make(eval_samples, seed + 1_000_003)
    return train, val, PIXEL_VOCAB_SIZE


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
    elif task == "retrieval":
        train, val, vocab_size = _build_retrieval(seq_len, train_samples, eval_samples, seed, data_dir)
        cols = ["input_ids_a", "attention_mask_a", "input_ids_b", "attention_mask_b", "labels"]
    elif task == "image":
        train, val, vocab_size = _build_image(seq_len, train_samples, eval_samples, seed)
        cols = ["input_ids", "attention_mask", "labels"]
    elif task == "pathfinder":
        train, val, vocab_size = _build_pathfinder(
            seq_len, train_samples, eval_samples, seed, side_default=32
        )
        cols = ["input_ids", "attention_mask", "labels"]
    else:  # pathfinder_x
        train, val, vocab_size = _build_pathfinder(
            seq_len, train_samples, eval_samples, seed, side_default=128
        )
        cols = ["input_ids", "attention_mask", "labels"]

    train.set_format(type="torch", columns=cols)
    val.set_format(type="torch", columns=cols)
    return {
        "train": train,
        "validation": val,
        "vocab_size": vocab_size,
        "num_labels": TASK_INFO[task]["num_labels"],
        "pair": TASK_INFO[task]["pair"],
    }
