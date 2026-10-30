"""Matched dense 1M single-forward prefill probe (no KV cache / no decode)."""
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from modal_rd.cpu_embed import install_cpu_embedding
from modal_rd.eval_niah import build, tokenize_prompt
from modal_rd.mlp_chunk import install_chunked_mlp
from modal_rd.niah import load_tokenizer, prepare
from modal_rd.model_config import model_metadata

ROOT = Path('/vol/results')
TAG = 'nemotron_dense_1m_prefill_20261029'


def main():
    source = json.loads((ROOT / 'nemotron_int64_1m_six_v2_20261028.json').read_text())
    sparse = source['phases'][0]['timed'][0]
    row = source['phases'][0]['manifest'][0]
    tokenizer = load_tokenizer()
    prompts = prepare(tokenizer, 4_000_000, 1, minimal=True)
    prompt = prompts[0]
    assert prompt['idx'] == sparse['idx'] == 0
    assert prompt['tokens'] == sparse['tokens']
    assert prompt['token_sha256'] == sparse['token_sha256'] == row['token_sha256']
    assert prompt['source_sha256'] == row['source_sha256']

    model = build('dense', {}, [])
    install_chunked_mlp(model, 8192)
    install_cpu_embedding(model)
    ids = tokenize_prompt(tokenizer, prompt['text'], return_tensors='pt', truncation=False)['input_ids'].cuda()
    assert ids.shape[1] == prompt['tokens']

    def forward():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                                                  SDPBackend.EFFICIENT_ATTENTION,
                                                  SDPBackend.MATH]):
            start.record()
            result = model(input_ids=ids, use_cache=False, logits_to_keep=1)
            end.record()
        torch.cuda.synchronize()
        return {'prefill_s': start.elapsed_time(end) / 1000,
                'peak_gb': torch.cuda.max_memory_allocated() / 1e9,
                'next_token_id': int(result.logits[0, -1].argmax())}

    warm = forward()
    torch.cuda.empty_cache()
    timed = forward()
    report = {**model_metadata(), 'tag': TAG, 'status': 'complete',
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'arm': 'dense SDPA', 'attention': 'causal dense; PyTorch SDPA backend selection',
              'cache': False, 'decode_tokens': 0, 'precision': 'bfloat16',
              'mlp_chunk': 8192, 'cpu_embed': True, 'warm': warm, 'timed': timed,
              'prompt': {k: prompt[k] for k in ('idx', 'tokens', 'token_sha256',
                                                  'source_sha256', 'answer', 'depth', 'seed')},
              'sparse_matched': {'prefill_s': sparse['prefill_s'],
                                 'wall_s': sparse['wall_s'],
                                 'peak_gb': sparse['peak_gb'],
                                 'generated': sparse['generated'],
                                 'exact': sparse['exact']},
              'source_probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out = ROOT / f'{TAG}.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
