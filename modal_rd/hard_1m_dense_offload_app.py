"""One-example B200 feasibility probe for CPU-offloaded dense KV cache."""
import subprocess
import sys
from modal_rd.app import app, vol


@app.function(gpu='B200', volumes={'/vol': vol}, timeout=14400, cpu=8, memory=262144)
def dense_offload_probe():
    try:
        subprocess.run([sys.executable, '-m', 'modal_rd.hard_1m_dense_offload_probe'], cwd='/root/repo', check=True)
    finally:
        vol.commit()
