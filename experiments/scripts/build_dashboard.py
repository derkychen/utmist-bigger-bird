#!/usr/bin/env python3
"""Regenerate embedded benchmark data in dashboard.html from repo JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard.html"


def exp_num(name: str) -> int:
    m = re.search(r"exp_(\d+)", name)
    return int(m.group(1)) if m else -1


def round_val(value, places=3):
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, places) if places else value
    return value


def seq_from_eval(meta: dict) -> int | None:
    mc = meta.get("model_config", {})
    ds = meta.get("dataset_info", {})
    seq = mc.get("seq_length") or ds.get("max_seq_len")
    return int(seq) if seq else None


def track_from_path(path: Path, meta: dict) -> str:
    parent = path.parent.name
    if parent.startswith("lra_"):
        return "lra"
    if parent.startswith("ruler_"):
        return "ruler"
    task = meta.get("task", "")
    if isinstance(task, str):
        if task.startswith("lra_"):
            return "lra"
        if task.startswith("ruler_"):
            return "ruler"
    return "imdb"


def load_csv_rows() -> list[dict]:
    rows = []
    for path in sorted(ROOT.glob("benchmarks/**/eval_*.json")):
        with open(path) as f:
            data = json.load(f)
        meta = data.get("experiment_metadata", {})
        perf = data.get("performance_metrics", {})
        ev = perf.get("eval", {})
        train = perf.get("train", {})
        ds = meta.get("dataset_info", {})
        env = meta.get("environment", {})
        mc = meta.get("model_config", {})
        name = meta.get("name", path.parent.name)
        ts = meta.get("timestamp", path.stem.replace("eval_", ""))
        train_samples = mc.get("train_samples") or ds.get("train_size")
        track = track_from_path(path, meta)
        rows.append(
            {
                "Track": track,
                "Task": meta.get("task", ""),
                "Experiment": name,
                "Timestamp": ts,
                "Seq_Length": seq_from_eval(meta),
                "Needle_Depth": round_val(mc.get("needle_depth"), 2),
                "Train_Samples": train_samples,
                "F1": round_val(ev.get("eval_f1")),
                "Accuracy": round_val(ev.get("eval_accuracy")),
                "Train_Time_s": round_val(perf.get("training_time_seconds"), 1),
                "Eval_Time_s": round_val(ev.get("eval_runtime"), 2),
                "Epochs": meta.get("training_config", {}).get("epochs"),
                "Train_Loss": round_val(train.get("train_loss"), 3),
                "Eval_Loss": round_val(ev.get("eval_loss"), 3),
                "Peak_Memory_MB": round_val(
                    perf.get("peak_memory_mb") or env.get("peak_memory_mb"), 2
                ),
                "Inference_Latency_ms": round_val(perf.get("inference_latency_ms"), 2),
                "Softmax_Comparisons": perf.get("softmax_comparisons"),
            }
        )
    return rows


def _efficiency_row(
    *,
    track: str,
    exp_name: str,
    exp_n: int,
    seq_length: int,
    f1,
    accuracy,
    train_time_s,
    peak_memory_mb,
    inference_latency_ms,
    softmax_comparisons,
    oom: bool,
) -> dict:
    return {
        "track": track,
        "exp_name": exp_name,
        "exp_num": exp_n,
        "seq_length": seq_length,
        "f1": round_val(f1),
        "accuracy": round_val(accuracy),
        "train_time_s": round_val(train_time_s, 2),
        "peak_memory_mb": round_val(peak_memory_mb, 2),
        "inference_latency_ms": round_val(inference_latency_ms, 2),
        "softmax_comparisons": softmax_comparisons,
        "oom": oom,
    }


def load_efficiency() -> list[dict]:
    """Merge efficiency_results, eval JSONs, and sweep files — keyed by (track, exp, seq)."""
    merged: dict[tuple[str, int, int], tuple[dict, int, str]] = {}

    def put(row: dict, priority: int, ts: str = "") -> None:
        key = (row["track"], row["exp_num"], row["seq_length"])
        cur = merged.get(key)
        if cur is None:
            merged[key] = (row, priority, ts)
            return
        cur_row, cur_pri, cur_ts = cur
        if priority < cur_pri:
            merged[key] = (row, priority, ts)
            return
        if priority > cur_pri:
            return
        # Same priority: prefer successful runs, then newer timestamp.
        if cur_row.get("oom") and not row.get("oom"):
            merged[key] = (row, priority, ts)
            return
        if row.get("oom") and not cur_row.get("oom"):
            return
        if ts > cur_ts:
            merged[key] = (row, priority, ts)

    eff_path = ROOT / "benchmarks/efficiency_results.json"
    if eff_path.is_file():
        with open(eff_path) as f:
            for row in json.load(f)["results"]:
                put(
                    _efficiency_row(
                        track="imdb",
                        exp_name=row["exp_name"],
                        exp_n=row["exp_num"],
                        seq_length=row["seq_length"],
                        f1=row.get("f1"),
                        accuracy=row.get("accuracy"),
                        train_time_s=row.get("train_time_s"),
                        peak_memory_mb=row.get("peak_memory_mb"),
                        inference_latency_ms=row.get("inference_latency_ms"),
                        softmax_comparisons=row.get("softmax_comparisons"),
                        oom=row.get("oom", False),
                    ),
                    priority=2,
                )

    for path in ROOT.glob("benchmarks/**/eval_*.json"):
        with open(path) as f:
            data = json.load(f)
        meta = data["experiment_metadata"]
        perf = data.get("performance_metrics", {})
        ev = perf.get("eval", {})
        seq = seq_from_eval(meta)
        if not seq:
            continue
        name = meta["name"]
        num = exp_num(name)
        ts = meta.get("timestamp", path.stem.replace("eval_", ""))
        put(
            _efficiency_row(
                track=track_from_path(path, meta),
                exp_name=name,
                exp_n=num,
                seq_length=seq,
                f1=ev.get("eval_f1"),
                accuracy=ev.get("eval_accuracy"),
                train_time_s=perf.get("training_time_seconds"),
                peak_memory_mb=perf.get("peak_memory_mb")
                or meta.get("environment", {}).get("peak_memory_mb"),
                inference_latency_ms=perf.get("inference_latency_ms"),
                softmax_comparisons=perf.get("softmax_comparisons"),
                oom=False,
            ),
            priority=1,
            ts=ts,
        )

    for pattern, track in (
        ("benchmarks/long_context_sweep*.json", "imdb"),
        ("benchmarks/lra_sweep*.json", "lra"),
        ("benchmarks/lra_oom*.json", "lra"),
        ("benchmarks/ruler_sweep*.json", "ruler"),
    ):
        for path in sorted(ROOT.glob(pattern)):
            with open(path) as f:
                payload = json.load(f)
            if not isinstance(payload, dict) or "results" not in payload:
                continue
            ts = payload.get("config", {}).get("timestamp", path.stem)[:19]
            ts = ts.replace("-", "").replace(":", "").replace("T", "_")
            for row in payload["results"]:
                seq = row.get("seq") or row.get("seq_length")
                if not seq:
                    continue
                exp_n = row.get("exp", row.get("exp_num"))
                exp_name = row.get("exp_name") or f"exp_{exp_n}"
                oom = row.get("oom", False) or row.get("status") == "oom"
                put(
                    _efficiency_row(
                        track=track,
                        exp_name=exp_name,
                        exp_n=exp_n,
                        seq_length=int(seq),
                        f1=row.get("f1"),
                        accuracy=row.get("accuracy"),
                        train_time_s=row.get("train_time_s"),
                        peak_memory_mb=row.get("peak_mem_mb") or row.get("peak_memory_mb"),
                        inference_latency_ms=row.get("inference_ms") or row.get("inference_latency_ms"),
                        softmax_comparisons=row.get("softmax_comparisons"),
                        oom=oom,
                    ),
                    priority=3,
                    ts=ts,
                )

    efficiency = [entry[0] for entry in merged.values()]
    efficiency.sort(key=lambda x: (x["track"], x["seq_length"], x["exp_num"]))
    return efficiency


def load_complexity() -> dict:
    with open(ROOT / "benchmarks/complexity_results.json") as f:
        cx = json.load(f)

    return {
        "metadata": {
            "timestamp": cx["metadata"]["timestamp"][:19],
            "device": cx["metadata"]["device"],
            "batch_size": "mixed",
            "trials": cx["metadata"]["trials"],
        },
        "results": [
            {
                "exp_num": row["exp_num"],
                "exp_name": row["exp_name"],
                "seq_length": row["seq_length"],
                "time_ms": round(row["time_ms"], 3) if row.get("time_ms") else None,
                "oom": row["oom"],
            }
            for row in cx["results"]
        ],
    }


def js_object(value) -> str:
    return json.dumps(value, separators=(",", ":"))


def js_rows(rows: list[dict]) -> str:
    lines = ["const CSV_ROWS = ["]
    for row in rows:
        parts = []
        for key, val in row.items():
            if val is None:
                parts.append(f"{key}:null")
            elif isinstance(val, str):
                parts.append(f'{key}:"{val}"')
            elif isinstance(val, bool):
                parts.append(f"{key}:{str(val).lower()}")
            else:
                parts.append(f"{key}:{val}")
        lines.append("  {" + ",".join(parts) + "},")
    lines.append("];")
    return "\n".join(lines)


def patch_dashboard(html: str, complexity: dict, efficiency: list[dict], rows: list[dict]) -> str:
    max_exp = max(
        max(exp_num(r["Experiment"]) for r in rows),
        max(e["exp_num"] for e in efficiency),
        max(r["exp_num"] for r in complexity["results"]),
    )
    eff_seqs = sorted({e["seq_length"] for e in efficiency})
    tracks = sorted({e["track"] for e in efficiency})

    data_block = "\n".join(
        [
            f"const COMPLEXITY = {js_object(complexity)};",
            "",
            f"const EFFICIENCY = {js_object(efficiency)};",
            "",
            js_rows(rows),
            "",
            f"const MAX_EXP = {max_exp};",
            f"const EFFICIENCY_SEQS = {js_object(eff_seqs)};",
            f"const TRACKS = {js_object(tracks)};",
        ]
    )

    html = re.sub(
        r"const COMPLEXITY = \{.*?const CSV_ROWS = \[.*?\];\n*(?:const MAX_EXP = \d+;\n*)?(?:const EFFICIENCY_SEQS = \[.*?\];\n*)?(?:const TRACKS = \[.*?\];\n*)?",
        data_block + "\n",
        html,
        count=1,
        flags=re.DOTALL,
    )

    html = html.replace("for(let e=0;e<=10;e++)", "for(let e=0;e<=MAX_EXP;e++)")

    return html


def main() -> None:
    rows = load_csv_rows()
    efficiency = load_efficiency()
    complexity = load_complexity()
    html = DASHBOARD.read_text()
    html = patch_dashboard(html, complexity, efficiency, rows)
    DASHBOARD.write_text(html)
    seqs = sorted({e["seq_length"] for e in efficiency})
    tracks = sorted({e["track"] for e in efficiency})
    print(f"Updated {DASHBOARD}")
    print(f"  CSV rows: {len(rows)}")
    print(f"  Efficiency rows: {len(efficiency)}")
    print(f"  Tracks: {tracks}")
    print(f"  Efficiency seq lengths: {seqs}")
    print(f"  Complexity rows: {len(complexity['results'])}")


if __name__ == "__main__":
    main()
