"""Eviction quality gate, then bounded single-prompt H100 scaling probes. No retuning."""
import argparse
import hashlib
import json
import traceback
from pathlib import Path

import torch
from modal_rd.eval_niah import build, run_one
from modal_rd.mlp_chunk import install_chunked_mlp
from modal_rd.cpu_embed import install_cpu_embedding
from modal_rd.model_config import model_metadata
from modal_rd.niah import MANIFEST_KEYS, load_tokenizer, prepare

ROOT = Path('/vol/results')
CONFIG = {'sink': 64, 'window': 16384, 'vertical': 512, 'exact_tail': 256,
          'kv_evict': True, 'evict_pool': 8, 'evict_keep': 16384}


def prefix_check(tok):
    control = json.loads((ROOT / 'nemotron_chat64_pilot_20261028_vs.json').read_text())
    manifest = json.loads((ROOT / 'nemotron_chat64_pilot_20261028_vs_bytes512000_depth0.5_prompts.json').read_text())
    assert control['model_id'] == model_metadata()['model_id']
    assert control['resolved_revision'] == model_metadata()['resolved_revision']
    for path in ('modal_rd/attention.py', 'kernels/bigger_bird_flash.py'):
        assert control['source_sha256'][path] == model_metadata()['source_sha256'][path]
    rows = prepare(tok, 512000, 6, minimal=True)
    assert len(rows) == len(manifest['rows']) == 6
    for r, old in zip(rows, manifest['rows']):
        assert {k: r[k] for k in MANIFEST_KEYS} == {k: old[k] for k in MANIFEST_KEYS}
    return rows


def calibrate(tok, target):
    nbytes, attempts = target * 4, []
    for _ in range(3):
        row = prepare(tok, nbytes, 1, minimal=True)[0]
        attempts.append({'bytes': nbytes, 'tokens': row['tokens']})
        if abs(row['tokens'] - target) / target <= .005:
            return nbytes, [row], attempts
        nbytes = max(1000, round(nbytes * target / row['tokens']))
    raise ValueError(f'Token calibration did not converge: {attempts}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--cpu-only', action='store_true')
    a = p.parse_args()
    tok = load_tokenizer()
    gate_rows = prefix_check(tok)
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / f'{a.tag}.json'
    report = {**model_metadata(), 'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'prefix_hash_check': True, 'config': CONFIG, 'warm': 'all', 'max_new': 64,
              'mlp_chunk': 8192, 'cpu_embed': True, 'int8': False, 'cache': True,
              'kind': 'six-prompt eviction gate + single-prompt feasibility; not a long-context benchmark',
              'phases': [], 'status': 'cpu_checked'}
    def save():
        out.write_text(json.dumps(report, indent=2))
    save()
    if a.cpu_only:
        print('PREFIX_HASH_CHECK_PASS', flush=True)
        return
    report.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    model = build('vs', CONFIG, [])
    install_chunked_mlp(model, 8192)
    install_cpu_embedding(model)
    try:
        for target in (128000, 262144, 524288, 1000000):
            phase = {'target_tokens': target, 'warm': [], 'timed': [], 'status': 'preparing'}
            report['phases'].append(phase)
            report['status'] = 'running'
            save()
            if target == 128000:
                nbytes, rows, attempts = 512000, gate_rows, []
            else:
                nbytes, rows, attempts = calibrate(tok, target)
            phase.update(bytes=nbytes, calibration=attempts,
                         manifest=[{k: v for k, v in r.items() if k != 'text'} for r in rows])
            save()
            for name in ('warm', 'timed'):
                for row in rows:
                    phase['status'] = f'{name}_{row["idx"]}'
                    save()
                    result = run_one(model, tok, row, 'vs', 64)
                    result['peak_reserved_gb'] = torch.cuda.max_memory_reserved() / 1e9
                    result['cache_kept_per_layer'] = [int(l.self_attn.kept.shape[-1]) for l in model.model.layers]
                    phase[name].append(result)
                    save()
                    print(f"{name.upper()} target={target} tokens={row['tokens']} idx={row['idx']} "
                          f"exact={result['exact']} prefill={result['prefill_s']:.3f} peak={result['peak_gb']:.3f}", flush=True)
            assert [r['token_sha256'] for r in phase['warm']] == [r['token_sha256'] for r in phase['timed']]
            phase['warm_timed_equal'] = all(w['generated_ids'] == t['generated_ids'] for w, t in zip(phase['warm'], phase['timed']))
            assert phase['warm_timed_equal'], 'Warm/timed predictions differ'
            phase.update(status='complete', exact=sum(r['exact'] for r in phase['timed']), n=len(rows))
            save()
            if target == 128000 and phase['exact'] != 6:
                report['status'] = 'stopped_eviction_quality_gate'
                save()
                print('EVICTION_GATE_FAILED: longer probes skipped', flush=True)
                return
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if report['phases']:
            report['phases'][-1]['status'] = 'failed'
        save()
        raise
    finally:
        save()
    print('SCALING_PROBE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
