"""One matched hard-task dense generation using a CPU-offloaded BF16 KV cache."""
import hashlib
import json
import traceback
from pathlib import Path

import modal
import torch
from modal_rd.hard_1m_pilot import TASKS
from modal_rd.model_config import MODEL_ID, NEMOTRON_ID, model_metadata
from modal_rd.niah import load_tokenizer, prepare_task
from modal_rd.eval_niah import build, run_one
from modal_rd.mlp_chunk import install_chunked_mlp
from modal_rd.cpu_embed import install_cpu_embedding

ROOT = Path('/vol/results')
OUT = ROOT / 'nemotron_hard_1m_dense_offload_probe_20261030.json'


def main():
    if MODEL_ID != NEMOTRON_ID or OUT.exists():
        raise RuntimeError('Expected Nemotron and an unused output path')
    plan = json.loads((ROOT / 'nemotron_hard_1m_pilot_20261029_plan.json').read_text())
    sparse = json.loads((ROOT / 'nemotron_hard_1m_pilot_20261029.json').read_text())
    meta = model_metadata()
    if plan['status'] != 'complete' or sparse['status'] != 'complete':
        raise ValueError('Sparse pilot or prompt plan incomplete')
    if plan['model']['source_sha256'] != meta['source_sha256'] or sparse['model']['resolved_revision'] != meta['resolved_revision']:
        raise ValueError('Source hashes or checkpoint revision changed')
    task = TASKS[0]
    report = {'status': 'running', 'task': task, 'model': meta,
              'gpu': torch.cuda.get_device_name(0), 'cache': 'offloaded BF16',
              'max_new': 64, 'warm': [], 'timed': [],
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    volume = modal.Volume.from_name('bb-vol')

    def save():
        OUT.write_text(json.dumps(report, indent=2))
        volume.commit()

    save()
    try:
        tok = load_tokenizer()
        spec = plan['tasks'][task]
        rows, skipped = prepare_task(tok, task, spec['bytes'], 3, depth=.5, seed=42)
        manifest = [{k: v for k, v in r.items() if k != 'text'} for r in rows]
        if manifest != spec['manifest'] or manifest != sparse['tasks'][task]['manifest']:
            raise ValueError('Prompt manifest mismatch')
        row = rows[0]
        report.update(manifest=manifest[:1], skipped=skipped)
        save()
        model = build('dense', {}, [])
        install_chunked_mlp(model, 8192)
        install_cpu_embedding(model)
        original_generate = model.generate

        def offloaded_generate(*args, **kwargs):
            kwargs['cache_implementation'] = 'offloaded'
            return original_generate(*args, **kwargs)

        model.generate = offloaded_generate
        for stage in ('warm', 'timed'):
            report['phase'] = stage
            save()
            result = run_one(model, tok, row, 'dense', 64)
            report[stage].append(result)
            save()
            print(stage, result['exact'], result['prefill_s'], result['wall_s'], result['peak_gb'], flush=True)
        report['warm_timed_equal'] = report['warm'][0]['generated_ids'] == report['timed'][0]['generated_ids']
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
