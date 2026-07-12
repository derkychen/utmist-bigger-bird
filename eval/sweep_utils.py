"""Shared helpers for LRA / RULER sweep orchestration."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"


def parse_csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def is_oom(blob: str) -> bool:
    b = blob.lower()
    return "out of memory" in b or "cuda out of memory" in b


def bench_dir(track: str, task: str, exp_name: str) -> Path:
    return BENCH / f"{track}_{task}_{exp_name}"


def load_matching_eval(track: str, task: str, exp_name: str, match_fn) -> dict | None:
    d = bench_dir(track, task, exp_name)
    if not d.is_dir():
        return None
    best = None
    for path in d.glob("eval_*.json"):
        with open(path) as f:
            data = json.load(f)
        if match_fn(data):
            ts = data.get("experiment_metadata", {}).get("timestamp", path.stem)
            if best is None or ts > best[0]:
                best = (ts, data)
    return best[1] if best else None


def save_sweep(payload: dict, prefix: str) -> tuple[Path, Path]:
    BENCH.mkdir(parents=True, exist_ok=True)
    ts_path = BENCH / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    latest_path = BENCH / f"{prefix}_results.json"
    for path in (ts_path, latest_path):
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    return ts_path, latest_path


def rebuild_dashboard() -> None:
    script = ROOT / "scripts" / "build_dashboard.py"
    if script.is_file():
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=False)


def run_module(module: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
