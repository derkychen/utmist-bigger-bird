"""Run CPU prompt calibration before the dependent H100 pilot."""
from modal_rd.app import app, run_cpu, run


@app.function(cpu=1, timeout=21600)
def controller():
    print('Starting prompt calibration', flush=True)
    run_cpu.remote('modal_rd.hard_1m_pilot', '--calibrate')
    print('Calibration complete; starting the H100 pilot', flush=True)
    run.remote('modal_rd.hard_1m_pilot', '')
    print('Hard 1M pilot complete', flush=True)
