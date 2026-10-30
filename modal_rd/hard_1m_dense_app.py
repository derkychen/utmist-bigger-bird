"""Detached B200 runner with enough time for full dense 1M prefill."""
import subprocess
import sys
import modal
from modal_rd.app import app, vol


@app.function(gpu='B200', volumes={'/vol': vol}, timeout=43200, cpu=8, memory=65536)
def dense_hard():
    try:
        subprocess.run([sys.executable, '-m', 'modal_rd.hard_1m_dense'], cwd='/root/repo', check=True)
    finally:
        vol.commit()
