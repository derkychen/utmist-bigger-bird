import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
tag = 'nemotron_chat64_pilot_20261028'
reports = {a: json.loads((root / f'{tag}_{a}.json').read_text()) for a in ('dense_rd', 'vs')}
checks, stats = {}, {}
for arm, report in reports.items():
    run = report['runs'][0]
    rows, warm = run['examples'], run['warmed_examples']
    manifest = json.loads((root / f'{tag}_{arm}_bytes512000_depth0.5_prompts.json').read_text())['rows']
    assert len(rows) == len(warm) == len(manifest) == run['n'] == 6
    assert report['max_new'] == 64 and report['warm'] == 'all'
    assert report['resolved_revision'] == 'fa3a10b9c6ad02f42c81e1d68dcf6967afb230ef'
    exact = 0
    for r, w, m in zip(rows, warm, manifest):
        match = re.search(r'\d+(?:,\d{3})*', r['generated'])
        score = bool(match and match[0].replace(',', '') == r['answer'])
        assert score == r['exact']
        exact += score
        assert r['idx'] == w['idx'] == m['idx']
        assert r['token_sha256'] == w['token_sha256'] == m['token_sha256']
        assert r['tokens'] == m['tokens']
        assert r['generated_ids'] == w['generated_ids']
        assert r['forwards'] == len(r['generated_ids']) <= report['max_new']
        assert r['prefill_s'] > 0 and r['wall_s'] > 0 and r['peak_gb'] > 0
    assert exact == run['exact']
    stats[arm] = {'exact': exact, 'n': len(rows), 'prefill_s': run['mean_prefill_s'],
                  'wall_s': sum(r['wall_s'] for r in rows) / len(rows), 'peak_gb': run['peak_gb'],
                  'min_tokens': min(r['tokens'] for r in rows), 'max_tokens': max(r['tokens'] for r in rows)}
    checks[arm] = True
assert reports['dense_rd']['source_sha256'] == reports['vs']['source_sha256']
assert reports['dense_rd']['gpu'] == reports['vs']['gpu']
assert [r['token_sha256'] for r in reports['dense_rd']['runs'][0]['examples']] == [r['token_sha256'] for r in reports['vs']['runs'][0]['examples']]
selfcheck = json.loads((root / f'{tag}_selfcheck.json').read_text())
assert selfcheck['pass']
speed = stats['dense_rd']['prefill_s'] / stats['vs']['prefill_s']
(root / 'verification.json').write_text(json.dumps({'pass': True, 'checks': checks, 'stats': stats, 'prefill_speedup': speed}, indent=2))
lines = ['# Nemotron bring-up — 2026-10-28', '', 'Six-prompt pilot, synthetic NIAH at depth 0.5 and seed 42. Native Nemotron chat, BF16, greedy max64, full KV cache; all six prompts warmed then timed. Same H100, prompt hashes and source hashes across arms. Exact full-number scoring independently checked.', '', '| Arm | Exact | Prefill s | Task s | Peak allocated GB | Actual token range |', '|---|---:|---:|---:|---:|---|']
for arm, x in stats.items():
    lines.append(f"| {arm} | {x['exact']}/{x['n']} | {x['prefill_s']:.3f} | {x['wall_s']:.3f} | {x['peak_gb']:.2f} | {x['min_tokens']}–{x['max_tokens']} |")
lines += ['', f'Prefill speedup: {speed:.3f}×. Native-HF versus dense-wrapper last-logit max error: {selfcheck["last_logit_max_error"]}. Greedy output IDs matched.', '', 'The original max10 attempt was stopped after its first dense warm prediction was truncated at `760` for gold `7606921`. It is preserved and excluded from the comparison. Both arms in the replacement use max64.', '', 'This pilot establishes bring-up only. It does not establish 30-prompt accuracy, harder-task quality, 1M feasibility, memory savings, or asymptotic scaling. No eviction or budget retuning was used.']
(root / 'results-report.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(stats, indent=2))
