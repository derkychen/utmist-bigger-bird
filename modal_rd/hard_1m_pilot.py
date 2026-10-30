"""Calibrate and run small, fully warmed Nemotron hard-NIAH pilots."""
import argparse
import gc
import hashlib
import json
import re
import traceback
from pathlib import Path

from modal_rd.model_config import MODEL_ID, NEMOTRON_ID, model_metadata
from modal_rd.niah import load_tokenizer, prepare_task
import modal_rd.niah as niah

# The dataset exposes tensors; iterating over millions of scalar tensors in the
# legacy byte decoder is extremely slow. Convert to ordinary integers first.
_original_ids_to_text = niah._ids_to_text
def _fast_ids_to_text(ids):
    return _original_ids_to_text(ids.tolist() if hasattr(ids, 'tolist') else ids)
niah._ids_to_text = _fast_ids_to_text

TASKS = ('niah_single_2', 'niah_single_3', 'niah_multikey_1',
         'niah_multikey_3', 'niah_multivalue', 'niah_multiquery')
ROOT = Path('/vol/results')
TAG = 'nemotron_hard_1m_pilot_20261029'


def save(path, data):
    path.write_text(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrate', action='store_true')
    args = parser.parse_args()
    if MODEL_ID != NEMOTRON_ID:
        raise ValueError('Set BB_MODEL=nemotron')
    ROOT.mkdir(parents=True, exist_ok=True)
    plan_path = ROOT / f'{TAG}_plan.json'
    tok = load_tokenizer()
    if args.calibrate:
        plan = {'status': 'running', 'model': model_metadata(), 'target_tokens': 980000,
                'count': 3, 'tasks': {}}
        save(plan_path, plan)
        for task in TASKS:
            nbytes = 3000000
            attempts = []
            for _ in range(5):
                try:
                    print('CALIBRATING', task, nbytes, flush=True)
                    rows, skipped = prepare_task(tok, task, nbytes, 3, depth=.5, seed=42)
                    lens = [r['tokens'] for r in rows]
                    attempts.append({'bytes': nbytes, 'tokens': lens, 'skipped': skipped})
                    if len(rows) == 3 and min(lens) >= 940000 and max(lens) <= 1020000:
                        plan['tasks'][task] = {'bytes': nbytes, 'attempts': attempts,
                            'manifest': [{k: v for k, v in r.items() if k != 'text'} for r in rows]}
                        save(plan_path, plan)
                        print('CALIBRATED', task, nbytes, lens, flush=True)
                        break
                    nbytes = int(nbytes * 980000 / max(lens))
                    del rows
                    gc.collect()
                except ValueError as e:
                    match = re.search(r'is (\d+) tokens; use fewer bytes', str(e))
                    if not match:
                        raise
                    attempts.append({'bytes': nbytes, 'error': str(e)})
                    nbytes = int(nbytes * 970000 / int(match.group(1)))
            else:
                raise RuntimeError(f'Calibration failed for {task}: {attempts}')
        plan['status'] = 'complete'
        save(plan_path, plan)
        return

    import torch
    from modal_rd.eval_niah import build, run_one
    from modal_rd.mlp_chunk import install_chunked_mlp
    from modal_rd.cpu_embed import install_cpu_embedding
    from modal_rd.nemotron_scaling import CONFIG
    plan = json.loads(plan_path.read_text())
    if plan['status'] != 'complete' or plan['model']['source_sha256'] != model_metadata()['source_sha256']:
        raise ValueError('Incomplete plan or source hashes changed')
    out = ROOT / f'{TAG}.json'
    if out.exists():
        raise FileExistsError(out)
    report = {'status': 'running', 'model': model_metadata(), 'gpu': torch.cuda.get_device_name(0),
              'config': CONFIG, 'warm': 'all', 'max_new': 64, 'count': 3,
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'tasks': {}}
    save(out, report)
    try:
        model = build('vs', CONFIG, [])
        install_chunked_mlp(model, 8192)
        install_cpu_embedding(model)
        for task in TASKS:
            spec = plan['tasks'][task]
            rows, skipped = prepare_task(tok, task, spec['bytes'], 3, depth=.5, seed=42)
            assert [r['token_sha256'] for r in rows] == [r['token_sha256'] for r in spec['manifest']]
            phase = {'status': 'running', 'bytes': spec['bytes'], 'skipped': skipped,
                     'manifest': spec['manifest'], 'warm': [], 'timed': []}
            report['tasks'][task] = phase
            save(out, report)
            for name in ('warm', 'timed'):
                for row in rows:
                    result = run_one(model, tok, row, 'vs', 64)
                    phase[name].append(result)
                    save(out, report)
                    print(name, task, row['idx'], row['tokens'], result['exact'], result['prefill_s'], flush=True)
            phase['warm_timed_equal'] = all(a['generated_ids'] == b['generated_ids']
                                            for a, b in zip(phase['warm'], phase['timed']))
            phase['status'] = 'complete'
            save(out, report)
            del rows
            gc.collect()
        report['status'] = 'complete'
    except Exception as e:
        report.update(status='failed', error=str(e), traceback=traceback.format_exc())
        raise
    finally:
        save(out, report)


if __name__ == '__main__':
    main()
