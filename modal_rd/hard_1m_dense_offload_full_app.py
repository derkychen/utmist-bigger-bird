"""Matched dense 1M hard-task run with CPU-offloaded BF16 KV cache."""
import subprocess
import sys
from modal_rd.app import app, vol


@app.function(gpu='B200', volumes={'/vol': vol}, timeout=72000, cpu=8, memory=262144)
def dense_offload_full():
    try:
        subprocess.run([sys.executable, '-m', 'modal_rd.hard_1m_dense_offload'], cwd='/root/repo', check=True)
    finally:
        vol.commit()
