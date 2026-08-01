#!/usr/bin/env python3
"""Build report_data.js (consumed by report.html) from benchmarks/**/eval_*.json.

Unlike build_dashboard.py, this emits one flat run table and lets the page derive
every view from it. Runs keep their compute budget (preset / train size / epochs)
so the page can refuse to compare runs trained on different budgets.

    python scripts/build_report.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_data.js"

# Legacy hand-named experiment dirs that predate the exp_N convention.
_CUSTOM_EXP_MAP = {
    "bigbird": 15,
    "bigbird_8000": 16,
    "biggerbird_globals_ver1": 17,
    "biggerbird_topk_mmr": 18,
    "biggerbird_topk_MMR_ver2": 19,
    "biggerbird_topkMMR_globals_ver1": 20,
    "biggerbird_topkMMR_globals_8000": 21,
    "biggerbird_topkMMR_globals_randoms_8000": 22,
    "biggerbird_topkMMR_globals_teleports_8000": 23,
}

EXP_LABELS = {
    0: "Baseline (dense)",
    1: "DeepSeek top-k",
    2: "Lightning hybrid",
    3: "Dynamic globals",
    4: "PBS attention",
    5: "Bigger Bird",
    6: "DeepSeek + PBS",
    7: "Layer adaptive",
    8: "Token drop",
    9: "Attention speculation",
    10: "GQA sparse",
    11: "NSA",
    12: "S2 / HHST",
    13: "Dynamic context",
    14: "Token drop + DeepSeek",
    15: "Bigger Bird (proper)",
}

# Architectural primitives each experiment is built from. Tagging runs this way
# lets the report attribute wins to mechanisms instead of whole experiments,
# which is what a new architecture actually gets designed out of.
MECHANISM_LABELS = {
    "dense": "Full dense attention",
    "window": "Sliding local window",
    "block": "Block-sparse selection",
    "topk": "Content-based top-k",
    "lowrank": "Low-rank scoring proxy",
    "globals": "Global tokens",
    "random": "Random / teleport links",
    "drop": "Token dropping",
    "layeradapt": "Per-layer budget schedule",
    "gqa": "Grouped KV heads",
    "specul": "Speculative verification",
    "sink": "Attention sink",
    "stride": "Strided / dilated blocks",
    "mmr": "MMR diversity selection",
    "chunk": "Chunked processing",
}

EXP_MECHANISMS = {
    0: ["dense"],
    1: ["topk", "lowrank"],
    2: ["block"],
    3: ["window", "globals"],
    4: ["block"],
    5: ["window", "topk", "globals", "random", "mmr"],
    6: ["topk", "lowrank", "block"],
    7: ["topk", "lowrank", "layeradapt"],
    8: ["drop"],
    9: ["window", "globals", "specul"],
    10: ["topk", "lowrank", "gqa"],
    11: ["block", "stride", "topk", "window"],
    12: ["block", "stride", "sink", "window"],
    13: ["drop", "chunk"],
    14: ["drop", "topk", "lowrank"],
    15: ["block", "topk", "globals", "random", "mmr"],
}

_NAME_MECHANISM_HINTS = [
    ("topkmmr", ["topk", "mmr"]),
    ("topk_mmr", ["topk", "mmr"]),
    ("topk", ["topk"]),
    ("mmr", ["mmr"]),
    ("globals", ["globals"]),
    ("teleport", ["random"]),
    ("random", ["random"]),
    ("bigbird", ["window", "block", "globals", "random"]),
]


def mechanisms_for(name: str, n: int, model_config: dict) -> list[str]:
    if n in EXP_MECHANISMS:
        return list(EXP_MECHANISMS[n])

    tags: set[str] = set()
    lowered = name.lower()
    for needle, mechs in _NAME_MECHANISM_HINTS:
        if needle in lowered:
            tags.update(mechs)

    # Fall back to whatever hyperparameters the run actually recorded.
    keys = set(model_config or {})
    if keys & {"window_size", "local_k"}:
        tags.add("window")
    if keys & {"block_size", "num_blocks", "topk_blocks", "shard_size", "local_blocks"}:
        tags.add("block")
    if keys & {"top_k", "max_k"}:
        tags.add("topk")
    if "low_rank_dim" in keys:
        tags.add("lowrank")
    if keys & {"num_globals", "globals_per_head"}:
        tags.add("globals")
    if keys & {"num_teleports", "teleports_per_head", "teleport_bias"}:
        tags.add("random")
    if keys & {"drop_after_layer", "drop_ratio", "target_budget"}:
        tags.add("drop")
    if keys & {"k_early", "k_mid", "k_late"}:
        tags.add("layeradapt")
    if "kv_groups" in keys:
        tags.add("gqa")
    if keys & {"verify_every", "verify_kl_weight", "num_anchors"}:
        tags.add("specul")
    if "use_sink" in keys:
        tags.add("sink")
    if keys & {"stride", "stride_blocks"}:
        tags.add("stride")
    if keys & {"diversity_lambda", "gamma_diversity", "use_topk_mmr"}:
        tags.add("mmr")
    if "chunk_size" in keys:
        tags.add("chunk")
    if model_config.get("attention") == "full_dense":
        tags.add("dense")

    return sorted(tags)


def exp_num(name: str) -> int:
    if name in _CUSTOM_EXP_MAP:
        return _CUSTOM_EXP_MAP[name]
    m = re.search(r"exp_(\d+)", name)
    return int(m.group(1)) if m else -1


def track_of(path: Path, meta: dict) -> str:
    parent = path.parent.name
    for track in ("lra", "ruler", "nolima"):
        if parent.startswith(f"{track}_"):
            return track
    task = meta.get("task") or ""
    for track in ("lra", "ruler", "nolima"):
        if isinstance(task, str) and task.startswith(f"{track}_"):
            return track
    return "imdb"


def task_of(path: Path, meta: dict, track: str) -> str:
    mc = meta.get("model_config", {})
    task = meta.get("task") or mc.get("canonical_task") or mc.get("benchmark")
    if not task:
        # e.g. "lra_listops_exp_5_bigger_bird" -> "lra_listops"
        task = re.sub(r"_exp_\d+.*$", "", path.parent.name)
    task = str(task)
    if track != "imdb" and task.startswith(f"{track}_"):
        task = task[len(track) + 1 :]
    return task or track


_KNOB_KEYS = {
    "window_size", "local_k", "block_size", "num_blocks", "topk_blocks", "top_k",
    "low_rank_dim", "num_globals", "num_teleports", "diversity_lambda", "teleport_bias",
    "drop_after_layer", "drop_ratio", "target_budget", "chunk_size", "k_early", "k_mid",
    "k_late", "kv_groups", "num_anchors", "verify_every", "verify_kl_weight", "stride",
    "stride_blocks", "shard_size", "local_blocks", "use_sink", "dense_layers",
}


def rnd(value, places=4):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), places)
    return None


def load_runs() -> list[dict]:
    runs = []
    for path in sorted(ROOT.glob("benchmarks/**/eval_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        meta = data.get("experiment_metadata", {})
        perf = data.get("performance_metrics", {})
        ev = perf.get("eval", {})
        if ev.get("eval_accuracy") is None:
            continue

        mc = meta.get("model_config", {})
        ds = meta.get("dataset_info", {})
        env = meta.get("environment", {})
        tc = meta.get("training_config", {})

        track = track_of(path, meta)
        name = meta.get("name") or path.parent.name
        seq = mc.get("seq_length") or meta.get("seq_length") or ds.get("max_seq_len")
        train_time = perf.get("training_time_seconds")

        gpu_hours = perf.get("gpu_hours")
        if gpu_hours is None and train_time and env.get("gpu_name"):
            gpu_hours = train_time * max(1, int(env.get("gpu_count") or 1)) / 3600.0

        trajectory = [
            point.get("eval_accuracy")
            for point in (perf.get("trajectory") or [])
            if point.get("eval_accuracy") is not None
        ]

        runs.append(
            {
                "track": track,
                "task": task_of(path, meta, track),
                "exp": name,
                "n": exp_num(name),
                "ts": meta.get("timestamp") or path.stem.replace("eval_", ""),
                "seq": int(seq) if seq else None,
                "depth": rnd(mc.get("needle_depth"), 3),
                "preset": mc.get("compute_preset") or "legacy",
                "trainN": ds.get("train_size"),
                "evalN": ds.get("eval_size"),
                "labels": ds.get("num_labels"),
                "epochs": tc.get("epochs"),
                "acc": rnd(ev.get("eval_accuracy")),
                "f1": rnd(ev.get("eval_f1")),
                "loss": rnd(ev.get("eval_loss")),
                "secs": rnd(train_time, 1),
                "gpuh": rnd(gpu_hours, 5),
                "mem": rnd(perf.get("peak_memory_mb"), 1),
                "lat": rnd(perf.get("inference_latency_ms"), 3),
                "sm": perf.get("softmax_comparisons"),
                "gpu": env.get("gpu_name"),
                "cluster": env.get("cluster") or env.get("compute_resource"),
                "host": env.get("hostname"),
                "job": env.get("slurm_job_id"),
                "mech": mechanisms_for(name, exp_num(name), mc),
                "knobs": {k: v for k, v in mc.items() if k in _KNOB_KEYS},
                "traj": [rnd(x, 4) for x in trajectory],
                "src": str(path.relative_to(ROOT)),
            }
        )
    return runs


def main() -> None:
    runs = load_runs()
    runs.sort(key=lambda r: (r["track"], r["task"], r["n"], r["ts"]))

    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "expLabels": {str(k): v for k, v in EXP_LABELS.items()},
        "mechLabels": MECHANISM_LABELS,
        "runs": runs,
    }

    with open(OUT, "w") as f:
        f.write("// Generated by scripts/build_report.py — do not edit by hand.\n")
        f.write("const REPORT = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")

    tracks = sorted({r["track"] for r in runs})
    presets = sorted({r["preset"] for r in runs})
    tagged = sum(1 for r in runs if r["gpu"])
    print(f"Wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"  runs: {len(runs)}")
    print(f"  tracks: {tracks}")
    print(f"  budgets: {presets}")
    print(f"  compute-tagged: {tagged}/{len(runs)}")


if __name__ == "__main__":
    main()
