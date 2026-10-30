"""Matched no-cache first-token diagnostics. Single forwards, not latency benchmarks or accuracy."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from modal_rd import attention as rd
from modal_rd.cpu_embed import install_cpu_embedding
from modal_rd.eval_niah import build
from modal_rd.mlp_chunk import install_chunked_mlp
from modal_rd.model_config import model_metadata, tokenize_prompt
from modal_rd.niah import load_tokenizer, prepare

CONFIG = {'sink': 64, 'window': 16384, 'vertical': 512, 'exact_tail': 256}
ROOT = Path('/vol/results')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    a = p.parse_args()
    reference = json.loads((ROOT / 'nemotron_evict_scale_20261028.json').read_text())
    tok = load_tokenizer()
    report = {**model_metadata(), 'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cache': False,
              'dtype': 'bfloat16', 'config': CONFIG, 'mlp_chunk': 8192, 'cpu_embed': True,
              'protocol': 'one first-forward per arm/context, no warmup; no full-answer accuracy or speedup claim',
              'status': 'running', 'phases': []}
    out = ROOT / f'{a.tag}.json'
    def save():
        out.write_text(json.dumps(report, indent=2))
    save()
    model = build('dense_rd', {}, [])
    install_chunked_mlp(model, 8192)
    install_cpu_embedding(model)
    try:
        # Check in-place RoPE at the actual large absolute positions, independently of dense cost.
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        q = torch.randn(1, 2, 4, 128, device='cuda', dtype=torch.bfloat16)
        pos = torch.tensor([[0, 127667, 522505, 995992]], device='cuda')
        with torch.inference_mode():
            cos, sin = model.model.rotary_emb(q, pos)
            expected, _ = apply_rotary_pos_emb(q, q, cos, sin)
            got = rd.rope_inplace(q.clone(), cos, sin)
        report['large_position_rope_identical'] = torch.equal(expected, got)
        assert report['large_position_rope_identical']
        save()
        for target in (128000, 524288, 1000000):
            old = next(p for p in reference['phases'] if p['target_tokens'] == target)
            row = prepare(tok, old['bytes'], 1, minimal=True)[0]
            assert row['token_sha256'] == old['manifest'][0]['token_sha256']
            phase = {'target': target, 'tokens': row['tokens'], 'token_sha256': row['token_sha256'], 'arms': {}, 'status': 'running'}
            report['phases'].append(phase)
            save()
            inputs = tokenize_prompt(tok, row['text'], return_tensors='pt', truncation=False).to('cuda')
            logits = {}
            for arm in ('vs', 'dense_rd'):
                phase['status'] = arm
                save()
                rd.install(model, arm, **(CONFIG if arm == 'vs' else {}))
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    values = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                assert torch.isfinite(values).all()
                logits[arm] = values
                path = ROOT / f'{a.tag}_{target}_{arm}_logits.pt'
                torch.save(values, path)
                v, ix = values.topk(10)
                phase['arms'][arm] = {'forward_s': elapsed, 'peak_gb': torch.cuda.max_memory_allocated()/1e9,
                    'top_ids': ix.tolist(), 'top_tokens': [tok.decode([i]) for i in ix.tolist()],
                    'top_logits': v.tolist(), 'logits_file': path.name,
                    'logits_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                save()
                print(f"FORWARD target={target} tokens={row['tokens']} arm={arm} seconds={elapsed:.3f} "
                      f"peak={phase['arms'][arm]['peak_gb']:.3f} top={phase['arms'][arm]['top_tokens'][:3]}", flush=True)
            phase['top1_equal'] = logits['vs'].argmax().item() == logits['dense_rd'].argmax().item()
            phase['logit_max_abs_diff'] = (logits['vs']-logits['dense_rd']).abs().max().item()
            phase['status'] = 'complete'
            save()
            del inputs, logits
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error=str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc())
        save()
        raise
    finally:
        save()
    print('PREFILL_DIAGNOSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
