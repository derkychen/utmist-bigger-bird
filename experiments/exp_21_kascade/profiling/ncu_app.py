"""Nsight Compute on a Modal H100 for the compressed-cache decode kernel (separate image with ncu).

  modal run --env engineering_team_bigger_bird experiments/exp_21_kascade/profiling/ncu_app.py
"""
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/root/repo")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "numpy")
    .env({"PYTHONPATH": "/root/repo"})
    .add_local_dir(REPO, "/root/repo", ignore=[".git", "**/__pycache__", "benchmarks", "viz", "context", "*.html",
                                               "report_data.js", "venv", ".venv", "modal_rd/logs", "modal_rd/results"])
)
app = modal.App("bigger-bird-ncu", image=image)


@app.function(gpu="H100!", timeout=1800, cpu=8, memory=65536)
def profile():
    ncu = next(p for p in ("/usr/local/cuda/bin/ncu", "/opt/nvidia/nsight-compute/ncu") if Path(p).exists())
    sections = ["SpeedOfLight", "Occupancy", "LaunchStats", "WarpStateStats", "ComputeWorkloadAnalysis",
                "MemoryWorkloadAnalysis", "SchedulerStats", "InstructionStats"]
    cmd = [ncu, "--clock-control", "none", "--nvtx", "--nvtx-include", "profiled_lloyd/", "--nvtx-include", "profiled_uniform/",
           "-k", "regex:_partial", "-c", "2",
           *[x for s in sections for x in ("--section", s)],
           "python", "-m", "experiments.exp_21_kascade.profiling.profile_target"]
    r = subprocess.run(cmd, cwd="/root/repo", capture_output=True, text=True)
    print(r.stdout[-60000:])
    print("STDERR", r.stderr[-4000:])


@app.local_entrypoint()
def main():
    profile.remote()
