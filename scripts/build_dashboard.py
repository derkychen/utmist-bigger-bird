#!/usr/bin/env python3
"""Build the self-contained benchmark dashboard and its diagnostics.

The benchmark directory contains several generations of result files:

* flat R1 generative ``results_*.json`` files;
* nested ``eval_*.json`` / ``results_*.json`` files;
* sweep payloads with a top-level ``results`` list; and
* older aggregate efficiency/complexity files.

This builder never silently throws those formats into one ambiguous cell. It
keeps every recognized source observation, emits deterministic preferred rows
for charts, records malformed/unrecognized files, and writes a durable build
report that can be inspected without opening the browser.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard.html"
BUILD_LOG = ROOT / "benchmarks" / "dashboard_build.log"
BUILD_REPORT = ROOT / "benchmarks" / "dashboard_build_report.json"

DATA_START = "<!-- DASHBOARD_DATA_START -->"
DATA_END = "<!-- DASHBOARD_DATA_END -->"

EXP_LABELS = {
    0: "Dense baseline",
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
    16: "Free NSA (param-free)",
    17: "Coarse-to-fine",
}

CUSTOM_EXP_MAP = {
    "bigbird": 15,
    "bigbird_8000": 16,
    "biggerbird_globals_ver1": 17,
    "biggerbird_topk_mmr": 18,
    "biggerbird_topk_mmr_ver2": 19,
    "biggerbird_topkmmr_globals_ver1": 20,
    "biggerbird_topkmmr_globals_8000": 21,
    "biggerbird_topkmmr_globals_randoms_8000": 22,
    "biggerbird_topkmmr_globals_teleports_8000": 23,
}

SKIP_NAMES = {
    "complexity_results.json",
    "complexity_results_backup_4096.json",
    BUILD_REPORT.name,
}

SOURCE_PRIORITY = {
    "direct_results": 100,
    "nested_results": 90,
    "nested_eval": 80,
    "sweep": 70,
    "efficiency_summary": 60,
}

# R1/Llama runs before this project-wide sparse-only correction may have used
# dense causal fallbacks. They remain visible as historical source runs but are
# excluded from current sparse-analysis charts.
SPARSE_FIX_CUTOFF = "20260813_000000"


# ---------------------------------------------------------------------------
# Small normalization helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def as_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        if not math.isfinite(parsed):
            return None
        return int(parsed) if parsed.is_integer() else parsed
    return None


def as_int(value: Any) -> int | None:
    number = as_number(value)
    return int(number) if number is not None else None


def rounded(value: Any, places: int = 4) -> float | int | None:
    number = as_number(value)
    if number is None:
        return None
    if isinstance(number, int):
        return number
    return round(number, places)


def first_value(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def timestamp_key(value: Any) -> str:
    return str(value or "")


def sparse_validity(exp_num: int, model: str, timestamp: Any) -> tuple[str, bool]:
    if exp_num == 0:
        return "dense_baseline", True
    if model != "r1-llama-8b":
        return "not_r1", True
    digits = re.sub(r"\D", "", str(timestamp or ""))
    if digits and digits < SPARSE_FIX_CUTOFF.replace("_", ""):
        return "pre_sparse_fix", False
    return "sparse_only", True


def pretty_variant(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = re.sub(r"[_+\-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def experiment_number(*values: Any) -> int:
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
        text = str(value or "")
        match = re.search(r"(?:^|[_\-])exp[_\-]?(\d+)(?:$|[_\-])", text.lower())
        if match:
            return int(match.group(1))
        normalized = _normalized_name(text)
        if normalized in CUSTOM_EXP_MAP:
            return CUSTOM_EXP_MAP[normalized]
    return -1


def parent_variant(path: Path, exp_num: int) -> str:
    parent = path.parent.name
    if parent == "benchmarks":
        return ""
    if exp_num >= 0:
        match = re.match(rf"exp[_\-]?{exp_num}[_\-](.*)$", parent, re.I)
        if match:
            suffix = re.split(r"_generative_|_lra_|_ruler_|_nolima_|_seq\d+", match.group(1), maxsplit=1, flags=re.I)[0]
            if suffix:
                return pretty_variant(suffix)
    return pretty_variant(parent)


def experiment_fields(path: Path, data: dict, meta: dict, exp_override: Any = None) -> dict:
    exp_num = experiment_number(
        exp_override,
        data.get("exp"),
        data.get("exp_num"),
        data.get("experiment"),
        data.get("exp_name"),
        meta.get("name"),
        path.parent.name,
    )
    raw_name = first_value(data.get("exp_name"), meta.get("name"))
    variant = parent_variant(path, exp_num)
    if not variant and raw_name:
        variant = pretty_variant(str(raw_name))
    label = EXP_LABELS.get(exp_num) or variant or (f"Experiment {exp_num}" if exp_num >= 0 else "Unnumbered run")
    experiment = f"exp_{exp_num} · {label}" if exp_num >= 0 else label
    return {
        "exp_num": exp_num,
        "experiment": experiment,
        "experiment_id": f"exp_{exp_num}" if exp_num >= 0 else _normalized_name(label),
        "experiment_label": label,
        "variant": variant,
    }


def track_from(path: Path, data: dict, meta: dict, fallback: str | None = None) -> str:
    mc = as_dict(meta.get("model_config"))
    explicit = first_value(data.get("track"), meta.get("track"), mc.get("track"))
    if explicit:
        explicit = str(explicit).lower()
        for track in ("ruler", "lra", "nolima", "imdb"):
            if track in explicit:
                return track

    text = " ".join(
        str(value or "").lower()
        for value in (
            path.parent.name,
            path.name,
            data.get("task"),
            meta.get("task"),
            data.get("benchmark"),
        )
    )
    for track in ("ruler", "lra", "nolima"):
        if re.search(rf"(?:^|[_\-]){track}(?:[_\-]|$)", text) or track in text:
            return track

    task = str(first_value(data.get("task"), meta.get("task"), "")).lower()
    if task in {"niah", "mq_niah", "ruler"}:
        return "ruler"
    if task in {"text", "listops", "lra_text", "lra_listops"}:
        return "lra"
    if fallback:
        return fallback
    # Legacy nested IMDb files often have no track field at all.
    if "experiment_metadata" in data or path.name.startswith("eval_"):
        return "imdb"
    return "unknown"


def task_from(path: Path, data: dict, meta: dict, track: str, fallback: Any = None) -> str:
    mc = as_dict(meta.get("model_config"))
    task = first_value(data.get("task"), meta.get("task"), mc.get("canonical_task"), mc.get("benchmark"), fallback)
    if task is None:
        return "imdb" if track == "imdb" else track
    task = str(task)
    for prefix in ("ruler_", "lra_", "nolima_"):
        if task.lower().startswith(prefix):
            task = task[len(prefix):]
    return task or ("imdb" if track == "imdb" else track)


def sequence_from(path: Path, data: dict, meta: dict, row: dict | None = None) -> int | None:
    row = row or {}
    mc = as_dict(meta.get("model_config"))
    ds = as_dict(meta.get("dataset_info"))
    candidates = (
        data.get("seq_len"),
        data.get("seq_length"),
        row.get("seq"),
        row.get("seq_length"),
        mc.get("seq_length"),
        meta.get("seq_length"),
        ds.get("max_seq_len"),
    )
    for candidate in candidates:
        parsed = as_int(candidate)
        if parsed:
            return parsed
    match = re.search(r"(?:^|[_\-])(?:seq|length|len)[_\-]?(\d+)(?:$|[_\-])", path.parent.name.lower())
    return int(match.group(1)) if match else None


def model_from(path: Path, data: dict, meta: dict, fallback: str | None = None) -> str:
    raw = first_value(
        data.get("model"),
        data.get("base_model"),
        meta.get("base_model"),
        as_dict(meta.get("model_config")).get("base_model"),
    )
    text = str(raw or "").lower()
    if not text:
        text = " ".join((path.parent.name.lower(), path.name.lower()))
    if any(token in text for token in ("llama", "deepseek", "r1-distill", "r1_")):
        return "r1-llama-8b"
    if "bart" in text:
        return "bart-base"
    if data.get("experiment_metadata") or path.name.startswith("eval_"):
        return fallback or "bart-base"
    if "_generative_" in path.parent.name.lower() or "generative" in path.name.lower():
        return fallback or "r1-llama-8b"
    return fallback or "unknown"


def source_kind(path: Path, data: dict, forced: str | None = None) -> str:
    if forced:
        return forced
    if data.get("experiment_metadata"):
        return "nested_eval" if path.name.startswith("eval_") else "nested_results"
    if "task" in data and ("accuracy" in data or "n_examples" in data):
        return "direct_results"
    if isinstance(data.get("results"), list):
        return "sweep"
    return "unknown"


def metric_sources(data: dict) -> tuple[dict, dict, dict, dict]:
    meta = as_dict(data.get("experiment_metadata"))
    perf = as_dict(data.get("performance_metrics"))
    ev = as_dict(perf.get("eval"))
    train = as_dict(perf.get("train"))
    return meta, perf, ev, train


def status_from(data: dict, accuracy: Any, expected: int | None, observed: int | None) -> tuple[str, bool]:
    raw = str(data.get("status") or data.get("state") or "").lower()
    if bool(data.get("oom")) or raw in {"oom", "out_of_memory"}:
        return "oom", False
    if raw in {"error", "failed", "failure", "timeout", "cancelled", "canceled"}:
        return raw, False
    if accuracy is None:
        return (raw or "incomplete"), False
    if expected and observed is not None and observed < expected:
        return "partial", False
    if raw and raw not in {"ok", "success", "complete", "completed"}:
        return raw, True
    return "ok", True


def make_run(
    path: Path,
    data: dict,
    diagnostics: dict,
    *,
    forced_kind: str | None = None,
    exp_override: Any = None,
    task_fallback: Any = None,
    track_fallback: str | None = None,
    model_fallback: str | None = None,
) -> dict | None:
    meta, perf, ev, train = metric_sources(data)
    track = track_from(path, data, meta, track_fallback)
    task = task_from(path, data, meta, track, task_fallback)
    exp = experiment_fields(path, data, meta, exp_override)
    seq = sequence_from(path, data, meta, data)
    if seq is None:
        diagnostics["unrecognized_files"].append({"file": relative(path), "reason": "no sequence length"})
        return None

    kind = source_kind(path, data, forced_kind)
    model = model_from(path, data, meta, model_fallback)
    accuracy = first_value(
        data.get("accuracy"),
        data.get("acc"),
        ev.get("eval_accuracy"),
        data.get("eval_accuracy"),
    )
    f1 = first_value(data.get("f1"), data.get("eval_f1"), ev.get("eval_f1"), accuracy)
    expected = first_value(data.get("n_examples"), data.get("expected_examples"), data.get("eval_samples"))
    expected = as_int(first_value(expected, as_dict(meta.get("dataset_info")).get("eval_size")))
    examples = data.get("examples")
    stored_examples = len(examples) if isinstance(examples, list) else None
    observed = as_int(data.get("completed_examples"))
    if observed is None and expected is not None and accuracy is not None:
        # Flat generative files intentionally store only a preview of examples;
        # the declared n_examples is the authoritative evaluation count.
        observed = expected
    if observed is None:
        observed = stored_examples

    if accuracy is None and isinstance(examples, list) and examples:
        correct = sum(1 for item in examples if isinstance(item, dict) and item.get("correct") is True)
        accuracy = correct / len(examples)
        observed = stored_examples
    accuracy = rounded(accuracy, 6)
    f1 = rounded(f1, 6)

    eval_time = first_value(
        data.get("time_seconds"),
        data.get("eval_time_s"),
        data.get("eval_runtime"),
        ev.get("eval_runtime"),
        perf.get("eval_time_seconds"),
    )
    train_time = first_value(data.get("train_time_s"), data.get("training_time_seconds"), perf.get("training_time_seconds"))
    latency = first_value(
        data.get("inference_latency_ms"),
        data.get("inference_ms"),
        perf.get("inference_latency_ms"),
    )
    if latency is None and as_number(eval_time) is not None and expected:
        latency = float(eval_time) * 1000.0 / expected

    peak_memory = first_value(data.get("peak_memory_mb"), data.get("peak_mem_mb"), perf.get("peak_memory_mb"), as_dict(meta.get("environment")).get("peak_memory_mb"))
    if peak_memory is None:
        peak_gb = first_value(data.get("peak_memory_gb"), data.get("peak_mem_gb"))
        if as_number(peak_gb) is not None:
            peak_memory = float(peak_gb) * 1024.0

    status, complete = status_from(data, accuracy, expected, observed)
    mc = as_dict(meta.get("model_config"))
    timestamp = first_value(data.get("timestamp"), meta.get("timestamp"), path.stem.replace("results_", "").replace("eval_", ""), "")
    validity, analysis_eligible = sparse_validity(exp["exp_num"], model, timestamp)
    causal = first_value(data.get("is_causal"), mc.get("is_causal"))
    if causal is None and track == "ruler":
        causal = True

    run = {
        **exp,
        "track": track,
        "task": task,
        "model": model,
        "source_kind": kind,
        "source_file": relative(path),
        "timestamp": str(timestamp),
        "seq_length": seq,
        "depth": rounded(first_value(data.get("depth"), data.get("needle_depth"), mc.get("needle_depth")), 3),
        "accuracy": accuracy,
        "f1": f1,
        "eval_time_s": rounded(eval_time, 3),
        "train_time_s": rounded(train_time, 3),
        "latency_ms": rounded(latency, 3),
        "peak_memory_mb": rounded(peak_memory, 2),
        "softmax_comparisons": as_number(first_value(data.get("softmax_comparisons"), perf.get("softmax_comparisons"))),
        "n_examples": expected,
        "observed_examples": observed,
        "stored_examples": stored_examples,
        "status": status,
        "complete": complete,
        "causal": causal,
        "attention_mode": "dense_baseline" if exp["exp_num"] == 0 else "sparse_experiment",
        "sparse_validity": validity,
        "analysis_eligible": analysis_eligible,
        "random_baseline": rounded(data.get("random_baseline"), 4),
        "gpu": first_value(data.get("gpu"), as_dict(meta.get("environment")).get("gpu_name")),
        "cluster": first_value(as_dict(meta.get("environment")).get("cluster"), as_dict(meta.get("environment")).get("compute_resource")),
        "host": as_dict(meta.get("environment")).get("hostname"),
        "slurm_job": first_value(data.get("slurm_job"), as_dict(meta.get("environment")).get("slurm_job_id")),
    }
    diagnostics["source_kind_counts"][kind] += 1
    diagnostics["track_counts"][track] += 1
    diagnostics["model_counts"][model] += 1
    return run


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def load_json(path: Path, diagnostics: dict) -> dict | None:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        diagnostics["parse_errors"].append({"file": relative(path), "error": str(exc)})
        return None
    if not isinstance(value, dict):
        diagnostics["unrecognized_files"].append({"file": relative(path), "reason": "top-level JSON is not an object"})
        return None
    return value


def aggregate_rows(path: Path, payload: dict, diagnostics: dict) -> list[dict]:
    config = as_dict(payload.get("config"))
    rows = []
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            diagnostics["unrecognized_files"].append({"file": relative(path), "reason": "non-object row in results list"})
            continue
        merged = dict(item)
        merged.setdefault("timestamp", config.get("timestamp", path.stem))
        merged.setdefault("task", config.get("task"))
        if merged.get("task") is None and isinstance(config.get("tasks"), list) and len(config["tasks"]) == 1:
            merged["task"] = config["tasks"][0]
        merged.setdefault("model", config.get("model"))
        path_text = str(path).lower()
        if "lra_" in path_text:
            aggregate_track = "lra"
        elif "ruler_" in path_text:
            aggregate_track = "ruler"
        elif "nolima_" in path_text:
            aggregate_track = "nolima"
        else:
            aggregate_track = "imdb"
        row = make_run(
            path,
            merged,
            diagnostics,
            forced_kind="sweep",
            exp_override=merged.get("exp", merged.get("exp_num")),
            task_fallback=merged.get("task"),
            track_fallback=aggregate_track,
            model_fallback="bart-base",
        )
        if row:
            rows.append(row)
    return rows


def load_runs(diagnostics: dict) -> list[dict]:
    runs: list[dict] = []
    for path in sorted((ROOT / "benchmarks").glob("**/*.json")):
        if not path.is_file():
            continue
        diagnostics["scanned_files"] += 1
        if path.name in SKIP_NAMES or "backup" in path.stem.lower():
            diagnostics["ignored_files"].append({"file": relative(path), "reason": "aggregate/backup source"})
            continue
        if "weights_" in str(path) or path.name in {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json"}:
            diagnostics["ignored_files"].append({"file": relative(path), "reason": "model artifact, not an evaluation result"})
            continue
        data = load_json(path, diagnostics)
        if data is None:
            continue
        diagnostics["parsed_files"] += 1

        if path.name == "complexity_results.json":
            continue
        if path.name == "efficiency_results.json":
            for item in data.get("results", []):
                if not isinstance(item, dict):
                    continue
                row = make_run(
                    path,
                    item,
                    diagnostics,
                    forced_kind="efficiency_summary",
                    exp_override=item.get("exp_num"),
                    task_fallback="imdb",
                    track_fallback="imdb",
                    model_fallback="bart-base",
                )
                if row:
                    runs.append(row)
            continue
        if isinstance(data.get("results"), list) and not data.get("experiment_metadata"):
            runs.extend(aggregate_rows(path, data, diagnostics))
            continue

        if data.get("experiment_metadata") or (path.name.startswith("eval_") and "performance_metrics" in data):
            row = make_run(path, data, diagnostics)
        elif "task" in data and ("accuracy" in data or "n_examples" in data or "time_seconds" in data):
            row = make_run(path, data, diagnostics)
        else:
            diagnostics["unrecognized_files"].append({"file": relative(path), "reason": "no supported benchmark schema"})
            row = None
        if row:
            runs.append(row)
    return runs


def load_complexity(diagnostics: dict) -> dict:
    path = ROOT / "benchmarks" / "complexity_results.json"
    data = load_json(path, diagnostics) if path.is_file() else {}
    if not data:
        return {"metadata": {}, "results": []}
    results = []
    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "exp_num": as_int(item.get("exp_num")),
                "exp_name": item.get("exp_name") or f"exp_{item.get('exp_num', -1)}",
                "seq_length": as_int(item.get("seq_length")),
                "batch_size": item.get("batch_size"),
                "time_ms": rounded(item.get("time_ms"), 4),
                "oom": bool(item.get("oom")),
            }
        )
    return {"metadata": data.get("metadata", {}), "results": results}


# ---------------------------------------------------------------------------
# Aggregation, audit, and report generation
# ---------------------------------------------------------------------------


def preferred_rows(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in runs:
        if not row.get("analysis_eligible", True):
            continue
        key = (
            row["track"],
            row["task"],
            row["exp_num"],
            row["seq_length"],
            row["model"],
            row.get("depth"),
        )
        groups[key].append(row)

    selected = []
    for group in groups.values():
        selected.append(
            max(
                group,
                key=lambda row: (
                    1 if row["complete"] else 0,
                    SOURCE_PRIORITY.get(row["source_kind"], 0),
                    timestamp_key(row.get("timestamp")),
                    row.get("n_examples") or 0,
                ),
            )
        )
    return sorted(selected, key=lambda row: (row["track"], row["task"], row["seq_length"], row["exp_num"], row.get("depth") or -1, row["model"]))


def mean(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if as_number(value) is not None]
    return rounded(sum(numbers) / len(numbers), 6) if numbers else None


def aggregate_runs(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in runs:
        if not row.get("analysis_eligible", True):
            continue
        key = (
            row["track"],
            row["task"],
            row["exp_num"],
            row["seq_length"],
            row["model"],
            row.get("depth"),
        )
        groups[key].append(row)

    output = []
    for key, group in groups.items():
        statuses = Counter(row["status"] for row in group)
        complete = [row for row in group if row["complete"]]
        latest = max(group, key=lambda row: timestamp_key(row.get("timestamp")))
        output.append(
            {
                "track": key[0],
                "task": key[1],
                "exp_num": key[2],
                "experiment": latest["experiment"],
                "experiment_label": latest["experiment_label"],
                "variant": latest["variant"],
                "seq_length": key[3],
                "model": key[4],
                "depth": key[5],
                "n_runs": len(group),
                "n_complete": len(complete),
                "n_examples": max((row.get("n_examples") or 0 for row in group), default=None) or None,
                "accuracy": mean([row.get("accuracy") for row in complete or group]),
                "accuracy_min": min((float(row["accuracy"]) for row in complete or group if as_number(row.get("accuracy")) is not None), default=None),
                "accuracy_max": max((float(row["accuracy"]) for row in complete or group if as_number(row.get("accuracy")) is not None), default=None),
                "f1": mean([row.get("f1") for row in complete or group]),
                "eval_time_s": mean([row.get("eval_time_s") for row in complete or group]),
                "latency_ms": mean([row.get("latency_ms") for row in complete or group]),
                "peak_memory_mb": mean([row.get("peak_memory_mb") for row in complete or group]),
                "softmax_comparisons": mean([row.get("softmax_comparisons") for row in complete or group]),
                "status_counts": dict(statuses),
                "status": "ok" if complete else ("oom" if statuses.get("oom") else "incomplete"),
                "source_files": sorted({row["source_file"] for row in group}),
            }
        )
    return sorted(output, key=lambda row: (row["track"], row["task"], row["seq_length"], row["exp_num"], row.get("depth") or -1, row["model"]))


def coverage_summaries(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    by_track_model: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in runs:
        if not row.get("analysis_eligible", True):
            continue
        by_track_model[(row["track"], row["model"])].append(row)
    coverage = []
    for (track, model), group in sorted(by_track_model.items()):
        coverage.append({
            "track": track,
            "model": model,
            "observations": len(group),
            "cells": len({(r["task"], r["exp_num"], r["seq_length"], r.get("depth")) for r in group}),
            "experiments": sorted({r["exp_num"] for r in group}),
            "tasks": sorted({r["task"] for r in group}),
            "seq_lengths": sorted({r["seq_length"] for r in group}),
            "complete": sum(1 for r in group if r["complete"]),
        })

    preferred = preferred_rows(runs)
    focus = [
        row for row in preferred
        if row["track"] == "ruler"
        and row["model"] == "r1-llama-8b"
        and row["task"] == "niah"
        and row.get("depth") == 0.5
    ]
    focus_by_exp: dict[int, list[dict]] = defaultdict(list)
    for row in focus:
        focus_by_exp[row["exp_num"]].append(row)
    focus_summary = []
    for exp_num, group in sorted(focus_by_exp.items()):
        focus_summary.append({
            "exp_num": exp_num,
            "experiment": group[0]["experiment"],
            "results": [
                {
                    "seq_length": row["seq_length"],
                    "accuracy": row["accuracy"],
                    "status": row["status"],
                    "complete": row["complete"],
                    "source_file": row["source_file"],
                }
                for row in sorted(group, key=lambda item: item["seq_length"])
            ],
        })
    return coverage, focus_summary


def audit_sparse_models() -> dict:
    violations = []
    files = []
    forbidden = (
        "dense_self_attention",
        "DenseLlamaAttention",
        "sdpa_dense_or_none",
        "F.scaled_dot_product_attention",
        "full_scores = torch.bmm(Q, K.transpose(1, 2))",
    )
    for path in sorted((ROOT / "experiments").glob("**/model_llama.py")):
        text = path.read_text(encoding="utf-8")
        exp_num = experiment_number(path.parent.name)
        found = [needle for needle in forbidden if needle in text]
        if exp_num != 0 and found:
            violations.append({"experiment": f"exp_{exp_num}", "file": relative(path), "forbidden_patterns": found})
        files.append({
            "experiment": f"exp_{exp_num}",
            "file": relative(path),
            "mode": "dense_baseline" if exp_num == 0 else ("sparse_only" if not found else "violation"),
            "forbidden_patterns": found,
        })
    utils_text = (ROOT / "sparse_attn_utils.py").read_text(encoding="utf-8")
    start = utils_text.find("def causal_sparse_attention(")
    end = utils_text.find("def last_query_topk_indices(", start)
    causal_body = utils_text[start:end] if start >= 0 and end >= 0 else ""
    if "dense_self_attention" in causal_body:
        violations.append({
            "experiment": "shared helper",
            "file": "sparse_attn_utils.py",
            "forbidden_patterns": ["causal_sparse_attention dense shortcut"],
        })
    return {
        "status": "pass" if not violations else "fail",
        "expected_dense_experiment": "exp_0",
        "short_sequence_helper": "sparse_only" if "dense_self_attention" not in causal_body else "dense_fallback",
        "files": files,
        "violations": violations,
    }


def build_data() -> dict:
    diagnostics = {
        "scanned_files": 0,
        "parsed_files": 0,
        "parse_errors": [],
        "unrecognized_files": [],
        "ignored_files": [],
        "source_kind_counts": Counter(),
        "track_counts": Counter(),
        "model_counts": Counter(),
    }
    runs = load_runs(diagnostics)
    analysis_runs = [row for row in runs if row.get("analysis_eligible", True)]
    complexity = load_complexity(diagnostics)
    sparse_audit = audit_sparse_models()
    coverage_summary, focus_summary = coverage_summaries(analysis_runs)
    diagnostics["source_kind_counts"] = dict(diagnostics["source_kind_counts"])
    diagnostics["track_counts"] = dict(diagnostics["track_counts"])
    diagnostics["model_counts"] = dict(diagnostics["model_counts"])
    diagnostics["run_count"] = len(runs)
    diagnostics["analysis_run_count"] = len(analysis_runs)
    diagnostics["excluded_pre_sparse_fix"] = len(runs) - len(analysis_runs)
    diagnostics["complete_run_count"] = sum(1 for row in analysis_runs if row["complete"])
    diagnostics["status_counts"] = dict(Counter(row["status"] for row in runs))
    diagnostics["coverage_keys"] = len({(r["track"], r["task"], r["exp_num"], r["seq_length"], r["model"], r.get("depth")) for r in analysis_runs})

    return {
        "schema_version": 2,
        "generated_at": now_iso(),
        "experiment_labels": {str(key): value for key, value in EXP_LABELS.items()},
        "runs": sorted(runs, key=lambda row: (row["track"], row["task"], row["model"], row["seq_length"], row["exp_num"], timestamp_key(row.get("timestamp")))),
        "preferred": preferred_rows(analysis_runs),
        "aggregates": aggregate_runs(analysis_runs),
        "coverage_summary": coverage_summary,
        "focus_summary": focus_summary,
        "complexity": complexity,
        "diagnostics": diagnostics,
        "sparse_audit": sparse_audit,
    }


def log_text(data: dict) -> str:
    diagnostics = data["diagnostics"]
    audit = data["sparse_audit"]
    lines = [
        f"Dashboard build: {data['generated_at']}",
        f"Schema: {data['schema_version']}",
        "",
        "SOURCE SCAN",
        f"  scanned JSON files: {diagnostics['scanned_files']}",
        f"  parsed JSON files: {diagnostics['parsed_files']}",
        f"  recognized observations: {diagnostics['run_count']}",
        f"  sparse-analysis observations: {diagnostics['analysis_run_count']}",
        f"  excluded pre-sparse-fix R1 observations: {diagnostics['excluded_pre_sparse_fix']}",
        f"  preferred chart rows: {len(data['preferred'])}",
        f"  aggregate cells: {len(data['aggregates'])}",
        f"  complete observations: {diagnostics['complete_run_count']}",
        f"  coverage keys: {diagnostics['coverage_keys']}",
        f"  parse errors: {len(diagnostics['parse_errors'])}",
        f"  unrecognized files/rows: {len(diagnostics['unrecognized_files'])}",
        f"  ignored aggregate/backup files: {len(diagnostics['ignored_files'])}",
        "",
        "SOURCE KINDS",
    ]
    lines.extend(f"  {key}: {value}" for key, value in sorted(diagnostics["source_kind_counts"].items()))
    lines.extend(["", "TRACKS", *[f"  {key}: {value}" for key, value in sorted(diagnostics["track_counts"].items())]])
    lines.extend(["", "MODELS", *[f"  {key}: {value}" for key, value in sorted(diagnostics["model_counts"].items())]])
    lines.extend(["", "RUN STATUS", *[f"  {key}: {value}" for key, value in sorted(diagnostics["status_counts"].items())]])
    lines.append("")
    lines.append("TRACK / MODEL COVERAGE")
    lines.extend(
        f"  {item['track']} / {item['model']}: observations={item['observations']} cells={item['cells']} complete={item['complete']} seqs={','.join(str(seq) for seq in item['seq_lengths'])}"
        for item in data["coverage_summary"]
    )
    lines.append("")
    lines.append("R1 RULER NIAH DEPTH=0.5")
    for item in data["focus_summary"]:
        formatted = []
        for result in item["results"]:
            accuracy_text = "—" if result["accuracy"] is None else f"{float(result['accuracy']) * 100:.1f}%"
            formatted.append(f"{result['seq_length']}={accuracy_text}[{result['status']}]")
        lines.append(f"  {item['experiment']}: {', '.join(formatted)}")
    lines.extend(["", "SPARSE AUDIT", f"  status: {audit['status']}", f"  expected dense baseline: {audit['expected_dense_experiment']}", f"  short-sequence helper: {audit['short_sequence_helper']}", f"  historical exclusion cutoff: {SPARSE_FIX_CUTOFF}"])
    if audit["violations"]:
        lines.extend(f"  VIOLATION: {json.dumps(item, sort_keys=True)}" for item in audit["violations"])
    else:
        lines.append("  no non-baseline dense attention patterns found")
    if diagnostics["parse_errors"]:
        lines.extend(["", "PARSE ERRORS"])
        lines.extend(f"  {item['file']}: {item['error']}" for item in diagnostics["parse_errors"])
    if diagnostics["unrecognized_files"]:
        lines.extend(["", "UNRECOGNIZED"])
        lines.extend(f"  {item['file']}: {item['reason']}" for item in diagnostics["unrecognized_files"])
    return "\n".join(lines) + "\n"


def write_dashboard(data: dict, dashboard_path: Path) -> None:
    html = dashboard_path.read_text(encoding="utf-8")
    if DATA_START not in html or DATA_END not in html:
        raise RuntimeError(f"dashboard markers missing in {dashboard_path}")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")
    block = f"{DATA_START}\nconst DASHBOARD_DATA = {payload};\n{DATA_END}"
    html = re.sub(re.escape(DATA_START) + r".*?" + re.escape(DATA_END), block, html, count=1, flags=re.DOTALL)
    dashboard_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=DASHBOARD)
    parser.add_argument("--log", type=Path, default=BUILD_LOG)
    parser.add_argument("--report", type=Path, default=BUILD_REPORT)
    args = parser.parse_args()

    data = build_data()
    args.report.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.log.write_text(log_text(data), encoding="utf-8")
    write_dashboard(data, args.dashboard)

    diagnostics = data["diagnostics"]
    print(f"Updated {args.dashboard}")
    print(f"Wrote {args.log}")
    print(f"Wrote {args.report}")
    print(f"  scanned={diagnostics['scanned_files']} parsed={diagnostics['parsed_files']} observations={diagnostics['run_count']}")
    print(f"  preferred={len(data['preferred'])} aggregates={len(data['aggregates'])} parse_errors={len(diagnostics['parse_errors'])}")
    print(f"  sparse_audit={data['sparse_audit']['status']}")


if __name__ == "__main__":
    main()
