"""Dense full-cache counterpart to the six-task sparse 1M pilot."""
import gc
import hashlib
import json
import traceback
from pathlib import Path

import modal
import torch
from modal_rd.hard_1m_pilot import TASKS  # installs equivalent fast byte decoder
from modal_rd.model_config import MODEL_ID, NEMOTRON_ID, model_metadata
from modal_rd.niah import load_tokenizer, prepare_task
from modal_rd.eval_niah import build, run_one
from modal_rd.mlp_chunk import install_chunked_mlp
from modal_rd.cpu_embed import install_cpu_embedding

ROOT = Path('/vol/results')
PLAN = ROOT / 'nemotron_hard_1m_pilot_20261029_plan.json'
SPARSE = ROOT / 'nemotron_hard_1m_pilot_20261029.json'
OUT = ROOT / 'nemotron_hard_1m_dense_b200_20261030.json'


def main():
    if MODEL_ID != NEMOTRON_ID:
        raise ValueError('Set BB_MODEL=nemotron')
    if OUT.exists():
        raise FileExistsError(OUT)
    plan, sparse = json.loads(PLAN.read_text()), json.loads(SPARSE.read_text())
    meta = model_metadata()
    if plan['status'] != 'complete' or sparse['status'] != 'complete':
        raise ValueError('Sparse pilot or prompt plan incomplete')
    if plan['model']['source_sha256'] != meta['source_sha256'] or sparse['model']['resolved_revision'] != meta['resolved_revision']:
        raise ValueError('Source hashes or checkpoint revision changed')
    gpu = torch.cuda.get_device_name(0)
    if 'B200' not in gpu:
        raise RuntimeError(f'Expected B200; got {gpu}')
    report = {'status': 'running', 'model': meta, 'gpu': gpu, 'arm': 'dense',
              'precision': 'bfloat16', 'cache': True, 'mlp_chunk': 8192,
              'cpu_embed': True, 'warm': 'all', 'max_new': 64, 'count': 3,
              'sparse_source': SPARSE.name, 'plan_source': PLAN.name,
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'tasks': {}}
    volume = modal.Volume.from_name('bb-vol')
    def save():
        OUT.write_text(json.dumps(report, indent=2))
        volume.commit()
    save()
    try:
        tok = load_tokenizer()
        model = build('dense', {}, [])
        install_chunked_mlp(model, 8192)
        install_cpu_embedding(model)
        for task in TASKS:
            spec = plan['tasks'][task]
            rows, skipped = prepare_task(tok, task, spec['bytes'], 3, depth=.5, seed=42)
            manifest = [{k: v for k, v in r.items() if k != 'text'} for r in rows]
            if manifest != spec['manifest'] or manifest != sparse['tasks'][task]['manifest']:
                raise ValueError(f'Prompt mismatch: {task}')
            phase = {'status': 'running', 'bytes': spec['bytes'], 'skipped': skipped,
                     'manifest': manifest, 'warm': [], 'timed': []}
            report['tasks'][task] = phase
            save()
            for stage in ('warm', 'timed'):
                for row in rows:
                    result = run_one(model, tok, row, 'dense', 64)
                    phase[stage].append(result)
                    save()
                    print(stage, task, row['idx'], row['tokens'], result['exact'],
                          result['prefill_s'], result['peak_gb'], flush=True)
            phase['warm_timed_equal'] = all(a['generated_ids'] == b['generated_ids']
                                             for a, b in zip(phase['warm'], phase['timed']))
            phase['status'] = 'complete'
            save()
            del rows
            gc.collect()
        report['status'] = 'complete'
    except Exception as e:
        report.update(status='failed', error=str(e), traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
