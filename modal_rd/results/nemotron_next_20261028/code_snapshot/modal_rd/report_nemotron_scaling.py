"""Independent scores and a readable report from a copied scaling probe JSON."""
import argparse
import json
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('raw', type=Path)
    a = p.parse_args()
    d = json.loads(a.raw.read_text())
    assert d['prefix_hash_check']
    report = ['# Nemotron eviction and scaling probe', '',
              f"Status: {d['status']}. Checkpoint: `{d['resolved_revision']}`.", '',
              'Single H100, BF16, native chat, greedy max64, cache eviction keep16384/pool8, '
              'sink64/window16384/routes512/exact-tail256, MLP chunk8192 and CPU embeddings.', '',
              '| Target | Actual tokens | Completed | Exact | Prefill s | Task s | Peak allocated GB |',
              '|---|---|---:|---:|---:|---:|---:|']
    checks = []
    for phase in d['phases']:
        timed, warm = phase['timed'], phase['warm']
        for r in timed + warm:
            m = re.search(r'\d+(?:,\d{3})*', r['generated'])
            exact = bool(m and m[0].replace(',', '') == r['answer'])
            assert exact == r['exact'], ('score', r['idx'])
            assert r['prefill_s'] > 0 and r['peak_gb'] > 0
        for w, t in zip(warm, timed):
            assert w['token_sha256'] == t['token_sha256']
            assert w['generated_ids'] == t['generated_ids']
        if phase['status'] == 'complete':
            assert len(warm) == len(timed) == phase['n']
            assert sum(r['exact'] for r in timed) == phase['exact']
            checks.append(phase['target_tokens'])
        if timed:
            tokens = [r['tokens'] for r in timed]
            avg = lambda k: sum(r[k] for r in timed) / len(timed)
            report.append(f"| {phase['target_tokens']} | {min(tokens)}–{max(tokens)} | {len(timed)} | "
                          f"{sum(r['exact'] for r in timed)}/{len(timed)} | {avg('prefill_s'):.3f} | "
                          f"{avg('wall_s'):.3f} | {max(r['peak_gb'] for r in timed):.3f} |")
        else:
            report.append(f"| {phase['target_tokens']} | pending | 0 | — | — | — | — |")
    report += ['', 'The six-example 128K phase checks eviction quality against a prior matched control. '
               'Longer phases have one prompt each; their scores are feasibility evidence, not a benchmark. '
               'There is no matched dense baseline beyond 128K, no tuned budget selection, and no '
               'general long-generation or multi-question cache claim.']
    if d.get('error'):
        report += ['', f"Failure: {d['error_type']}: {d['error']}"]
    a.raw.with_name('verification.json').write_text(json.dumps({'independent_scores_pass': True,
        'job_status': d['status'], 'complete_phases': checks}, indent=2))
    out = a.raw.with_name('results-report.md')
    out.write_text('\n'.join(report) + '\n')
    print(out)


if __name__ == '__main__':
    main()
