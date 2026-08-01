#!/usr/bin/env python3
"""Append compute/cluster metadata onto existing eval_*.json files.

Use this when a run already started with an older runner that only saved
use_mps / fp16 / peak_memory_mb. Safe to re-run; skips files that already
have gpu_name (unless --force).

Examples (on the cluster, after/during the job):
  # Tag all missing files as this Trillium H100 job
  python scripts/backfill_compute_env.py \\
    --profile trillium-h100 \\
    --slurm-job-id 669255 \\
    --hostname trig0004 \\
    --only-missing

  # Live enrich as new results appear (Ctrl-C to stop)
  python scripts/backfill_compute_env.py --profile trillium-h100 \\
    --slurm-job-id $SLURM_JOB_ID --watch 60

  # Dry run
  python scripts/backfill_compute_env.py --profile trillium-h100 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROFILES = {
    "trillium-h100": {
        "cluster": "trillium",
        "compute_resource": "trillium-h100",
        "device": "cuda",
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_count": 1,
        "gpu_memory_total_mb": 81559.0,
        "cuda_capability": "9.0",
        "cuda_available": True,
        "use_mps": False,
    },
    "local-gtx1660": {
        "cluster": "local",
        "compute_resource": "local-gtx1660",
        "device": "cuda",
        "gpu_name": "NVIDIA GeForce GTX 1660 SUPER",
        "gpu_count": 1,
        "gpu_memory_total_mb": 6144.0,
        "cuda_capability": "7.5",
        "cuda_available": True,
        "use_mps": False,
    },
}


def has_compute_fields(env: dict) -> bool:
    return bool(env.get("gpu_name") or env.get("cluster") or env.get("hostname"))


def build_patch(args: argparse.Namespace) -> dict:
    patch = dict(PROFILES.get(args.profile, {})) if args.profile else {}
    overrides = {
        "cluster": args.cluster,
        "compute_resource": args.compute_resource,
        "hostname": args.hostname,
        "gpu_name": args.gpu_name,
        "gpu_count": args.gpu_count,
        "gpu_memory_total_mb": args.gpu_mem_mb,
        "cuda_capability": args.cuda_capability,
        "slurm_job_id": args.slurm_job_id,
        "slurm_job_name": args.slurm_job_name,
        "slurm_nodelist": args.slurm_nodelist,
        "device": args.device,
        "torch_version": args.torch_version,
        "cuda_version": args.cuda_version,
        "platform": args.platform,
    }
    for key, val in overrides.items():
        if val is not None and val != "":
            patch[key] = val

    # Fill from live Slurm env if not provided
    if not patch.get("slurm_job_id"):
        patch["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    if not patch.get("slurm_job_name"):
        patch["slurm_job_name"] = os.environ.get("SLURM_JOB_NAME")
    if not patch.get("slurm_nodelist"):
        patch["slurm_nodelist"] = os.environ.get("SLURM_NODELIST") or os.environ.get(
            "SLURM_JOB_NODELIST"
        )
    if not patch.get("hostname") and patch.get("slurm_nodelist"):
        # Often a single node name like trig0004
        nodelist = patch["slurm_nodelist"]
        if "[" not in nodelist:
            patch["hostname"] = nodelist

    if args.profile and "cuda_available" not in patch:
        patch["cuda_available"] = patch.get("device") == "cuda"

    return {k: v for k, v in patch.items() if v is not None}


def gpu_hours(train_time_s, gpu_count) -> float | None:
    if train_time_s is None or not gpu_count:
        return None
    try:
        return round(float(train_time_s) * int(gpu_count) / 3600.0, 6)
    except (TypeError, ValueError):
        return None


def enrich_file(path: Path, patch: dict, *, force: bool, dry_run: bool) -> str:
    with open(path) as f:
        data = json.load(f)

    meta = data.setdefault("experiment_metadata", {})
    env = meta.setdefault("environment", {})
    perf = data.setdefault("performance_metrics", {})

    if has_compute_fields(env) and not force:
        return "skip"

    # Preserve existing peak_memory / fp16 / use_mps when patch lacks them
    merged = {**env, **{k: v for k, v in patch.items() if v is not None}}
    # Prefer already-recorded peak memory
    if env.get("peak_memory_mb") is not None:
        merged["peak_memory_mb"] = env["peak_memory_mb"]
    elif perf.get("peak_memory_mb") is not None:
        merged["peak_memory_mb"] = perf["peak_memory_mb"]

    meta["environment"] = merged

    gh = gpu_hours(perf.get("training_time_seconds"), merged.get("gpu_count") or 1)
    if gh is not None and (force or perf.get("gpu_hours") is None):
        perf["gpu_hours"] = gh

    if dry_run:
        return "dry"

    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    return "updated"


def iter_eval_jsons(root: Path, glob: str) -> list[Path]:
    return sorted(root.glob(glob))


def run_once(args: argparse.Namespace, patch: dict) -> tuple[int, int, int]:
    updated = skipped = dry = 0
    for path in iter_eval_jsons(ROOT, args.glob):
        if args.mtime_after:
            if path.stat().st_mtime < args.mtime_after:
                continue
        status = enrich_file(path, patch, force=args.force, dry_run=args.dry_run)
        if status == "updated":
            updated += 1
            print(f"  updated {path.relative_to(ROOT)}")
        elif status == "dry":
            dry += 1
            print(f"  would update {path.relative_to(ROOT)}")
        else:
            skipped += 1
    return updated, skipped, dry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        help="Named hardware profile to merge into environment",
    )
    p.add_argument("--glob", default="benchmarks/**/eval_*.json", help="Path glob under repo root")
    p.add_argument("--force", action="store_true", help="Overwrite files that already have compute fields")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--watch", type=float, default=0, help="Seconds between passes (0 = once)")
    p.add_argument(
        "--mtime-after",
        type=float,
        default=0,
        help="Only touch files with mtime >= this unix timestamp",
    )
    p.add_argument("--only-missing", action="store_true", help="Alias: do not use --force (default)")
    p.add_argument("--cluster")
    p.add_argument("--compute-resource")
    p.add_argument("--hostname")
    p.add_argument("--gpu-name")
    p.add_argument("--gpu-count", type=int)
    p.add_argument("--gpu-mem-mb", type=float)
    p.add_argument("--cuda-capability")
    p.add_argument("--slurm-job-id")
    p.add_argument("--slurm-job-name")
    p.add_argument("--slurm-nodelist")
    p.add_argument("--device")
    p.add_argument("--torch-version")
    p.add_argument("--cuda-version")
    p.add_argument("--platform")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    patch = build_patch(args)
    if not patch:
        print(
            "ERROR: provide --profile and/or field overrides "
            "(e.g. --profile trillium-h100 --slurm-job-id …)",
            file=sys.stderr,
        )
        return 1

    print(f"Repo: {ROOT}")
    print(f"Patch: {json.dumps(patch, indent=2)}")

    while True:
        updated, skipped, dry = run_once(args, patch)
        print(f"Done: updated={updated} skipped={skipped} dry_run={dry}")
        if not args.watch:
            break
        time.sleep(args.watch)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
