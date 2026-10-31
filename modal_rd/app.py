"""Modal harness for Bigger Bird long-context R&D on full H100s.

    modal run modal_rd/app.py::download
    modal run modal_rd/app.py --module modal_rd.eval_niah --args "--arm dense --bytes 512000 --n 6"
    modal run --detach modal_rd/app.py --module ... --args "..."   # keep running after local exit

The repo is copied into the image at /root/repo; results and model weights
live on the persistent `bb-vol` volume (/vol/results, /vol/models).
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
from modal_rd.model_config import MODEL_ID, MODEL_DIR, MODEL_REVISION, PROMPT_FORMAT

vol = modal.Volume.from_name("bb-vol", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "accelerate", "datasets",
                 "omegaconf", "huggingface_hub", "safetensors", "tokenizers", "numpy")
    .env({"SCRATCH": "/vol", "PYTHONPATH": "/root/repo", "HF_HOME": "/vol/hf-cache",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "BB_MODEL": MODEL_ID, "BB_MODEL_REVISION": MODEL_REVISION, "BB_PROMPT_FORMAT": PROMPT_FORMAT})
    .add_local_dir(REPO, "/root/repo", ignore=[
        ".git", "**/__pycache__", "benchmarks", "viz", "context", "*.html",
        "report_data.js", "venv", ".venv", "modal_rd/logs", "modal_rd/results"])
)
app = modal.App("bigger-bird-rd", image=image)


@app.function(volumes={"/vol": vol}, timeout=3600, cpu=4)
def download():
    import json
    from huggingface_hub import HfApi, snapshot_download
    resolved = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION).sha
    snapshot_download(MODEL_ID, revision=resolved, local_dir=MODEL_DIR)
    (Path(MODEL_DIR) / "bb_checkpoint.json").write_text(json.dumps(
        {"model_id": MODEL_ID, "resolved_revision": resolved}, indent=2))
    vol.commit()
    print(sorted(p.name for p in Path(MODEL_DIR).iterdir()))


@app.function(volumes={"/vol": vol}, timeout=3600, cpu=8, memory=32768)
def run_cpu(module: str, args: str = ""):
    try:
        subprocess.run([sys.executable, "-m", module, *shlex.split(args)], cwd="/root/repo", check=True)
    finally:
        vol.commit()


GPU = os.environ.get("BB_GPU", "H100!")
MEMORY_MB = int(os.environ.get("BB_MEMORY_MB", "65536"))     # container RAM; ~200000 for a CPU-offloaded 1M cache


def run(module: str, args: str = ""):
    try:
        subprocess.run([sys.executable, "-m", module, *shlex.split(args)], cwd="/root/repo", check=True)
    finally:
        vol.commit()


# BB_GPU=none registers only CPU functions (e.g. download before GPU billing is enabled).
if GPU != "none":
    run = app.function(gpu=GPU, volumes={"/vol": vol}, timeout=6 * 3600, cpu=8, memory=MEMORY_MB)(run)


@app.local_entrypoint()
def main(module: str, args: str = ""):
    run.remote(module, args)
