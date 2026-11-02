"""Audit LongCodeQA's 1M bucket using the actual checkpoint tokenizer."""
import array
import hashlib
import json
from pathlib import Path

import modal
from experiments.exp_20_sota_long_context.run import Prompter
from experiments.exp_20_sota_long_context.tasks import load_task
from modal_rd.model_config import MODEL_ID, NEMOTRON_ID, model_metadata
from modal_rd.niah import load_tokenizer


def main():
    if MODEL_ID != NEMOTRON_ID:
        raise ValueError('Use BB_MODEL=nemotron')
    path = Path('/vol/results/final_bigger_bird_20261101/longcodeqa_1m_manifest.json')
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name('bb-vol')
    tok = load_tokenizer()
    prompter = Prompter(tok, 1_000_000)
    rows = load_task('lcqa1m')
    result = {'status': 'running', 'model': model_metadata(), 'max_tokens': 1_000_000,
              'task': 'lcqa1m', 'expected_count': len(rows), 'rows': []}
    for row in rows:
        ids, info = prompter.build(row)
        digest = hashlib.sha256(array.array('I', ids).tobytes()).hexdigest()
        result['rows'].append({'id': row['id'], 'repo': row['repo'], 'gold': row['gold'],
                               'tokens': len(ids), 'token_sha256_u32_native': digest, **info})
        path.write_text(json.dumps(result, indent=2))
        if len(result['rows']) % 5 == 0:
            volume.commit()
        print(row['id'], len(ids), info['truncated'], flush=True)
    result['status'] = 'complete'
    path.write_text(json.dumps(result, indent=2))
    volume.commit()


if __name__ == '__main__':
    main()
