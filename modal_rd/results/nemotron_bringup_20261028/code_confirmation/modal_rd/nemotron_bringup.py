"""Bounded GPU gate: kernel + model parity, then six paired ~128K prompts. No tuning."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from modal_rd.model_config import model_metadata


def run(module, *args):
    subprocess.run([sys.executable, '-m', module, *args], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--n', type=int, default=6)
    p.add_argument('--check-bytes', type=int, default=64000)
    p.add_argument('--max-new', type=int, default=64)
    a = p.parse_args()
    run('modal_rd.model_preflight')
    run('modal_rd.selfcheck', '--bytes', str(a.check_bytes), '--tag', a.tag + '_selfcheck')
    for arm in ('dense_rd', 'vs'):
        args = ['--arm', arm, '--bytes', '512000', '--n', str(a.n), '--max-new', str(a.max_new), '--warm', 'all',
                '--mlp-chunk', '8192', '--cpu-embed', '--tag', f'{a.tag}_{arm}']
        if arm == 'vs':
            args += ['--attn', json.dumps({'sink': 64, 'window': 16384, 'vertical': 512, 'exact_tail': 256})]
        run('modal_rd.eval_niah', *args)
    root = Path('/vol/results')
    dense, sparse = [json.loads((root / f'{a.tag}_{arm}.json').read_text()) for arm in ('dense_rd', 'vs')]
    d, s = dense['runs'][0], sparse['runs'][0]
    assert [r['token_sha256'] for r in d['examples']] == [r['token_sha256'] for r in s['examples']]
    assert d['warm_timed_equal'] and s['warm_timed_equal'], 'Warm/timed predictions changed'
    assert d['n'] == s['n'] == a.n
    summary = {**model_metadata(), 'verified': True, 'n': a.n, 'max_new': a.max_new,
               'dense_exact': d['exact'], 'sparse_exact': s['exact'],
               'dense_prefill_s': d['mean_prefill_s'], 'sparse_prefill_s': s['mean_prefill_s'],
               'prefill_speedup': d['mean_prefill_s'] / s['mean_prefill_s'],
               'min_tokens': min(r['tokens'] for r in d['examples']),
               'max_tokens': max(r['tokens'] for r in d['examples']),
               'stage': 'matched 30-prompt confirmation' if a.n >= 30 else 'six-prompt pilot; no tuning or final benchmark claim',
               'untuned_indices_6_onward': {arm: {'n': sum(r['idx'] >= 6 for r in report['examples']),
                   'exact': sum(r['exact'] for r in report['examples'] if r['idx'] >= 6)}
                   for arm, report in (('dense_rd', d), ('vs', s))}}
    (root / f'{a.tag}_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
