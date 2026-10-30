"""Retest prior exact prompts after 64-bit kernel indexing; budgets stay fixed."""
import argparse
import hashlib
import json
import traceback
from pathlib import Path
import torch
from modal_rd.cpu_embed import install_cpu_embedding
from modal_rd.eval_niah import build, run_one
from modal_rd.kernelcheck import check
from modal_rd.mlp_chunk import install_chunked_mlp
from modal_rd.model_config import model_metadata
from modal_rd.niah import load_tokenizer, prepare
from modal_rd.nemotron_scaling import CONFIG

ROOT = Path('/vol/results')


def main():
    p = argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    gate=json.loads((ROOT/'long_indexing_check_20261028.json').read_text())
    assert gate['status']=='pass'
    assert gate['kernel_sha256']==hashlib.sha256(Path('/root/repo/kernels/bigger_bird_flash.py').read_bytes()).hexdigest()
    refs=[]
    for n,w,k in ((4096,512,256),(131072,4096,2048)):
        err=check(n,w,k,rows=16);assert err<.03,(n,err);refs.append({'n':n,'max_error':err})
    torch.cuda.empty_cache()
    old=json.loads((ROOT/'nemotron_evict_scale_20261028.json').read_text())
    assert old['resolved_revision']==model_metadata()['resolved_revision']
    tok=load_tokenizer()
    model=build('vs',CONFIG,[]);install_chunked_mlp(model,8192);install_cpu_embedding(model)
    report={**model_metadata(),'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'gpu':torch.cuda.get_device_name(0),'config':CONFIG,'warm':'all','max_new':64,
        'kernel_checks':refs,'status':'running','phases':[]}
    path=ROOT/f'{a.tag}.json'
    def save():path.write_text(json.dumps(report,indent=2))
    # Fail before token selection if a model forward produces NaN/Inf logits.
    def finite_logits(m,args,output):
        if not torch.isfinite(output.logits).all():
            raise FloatingPointError('Model logits contain NaN/Inf')
    guard=model.register_forward_hook(finite_logits)
    try:
        for target in (128000,524288,1000000):
            original=next(x for x in old['phases'] if x['target_tokens']==target)
            rows=prepare(tok,original['bytes'],len(original['manifest']),minimal=True)
            assert [r['token_sha256'] for r in rows]==[r['token_sha256'] for r in original['manifest']]
            phase={'target_tokens':target,'status':'running','manifest':original['manifest'],'warm':[],'timed':[]}
            report['phases'].append(phase);save()
            for name in ('warm','timed'):
                for row in rows:
                    phase['status']=f'{name}_{row["idx"]}';save()
                    r=run_one(model,tok,row,'vs',64)
                    phase[name].append(r);save()
                    print(f"{name.upper()} target={target} tokens={row['tokens']} idx={row['idx']} exact={r['exact']} "
                          f"prefill={r['prefill_s']:.3f} peak={r['peak_gb']:.3f} gen={r['generated']!r}",flush=True)
            assert all(w['generated_ids']==t['generated_ids'] for w,t in zip(phase['warm'],phase['timed']))
            phase.update(status='complete',exact=sum(r['exact'] for r in phase['timed']),n=len(rows));save()
            if target==128000 and phase['exact']!=6:
                raise RuntimeError('128K regression gate failed')
        report['status']='complete'
    except Exception as exc:
        report.update(status='failed',error=str(exc),error_type=type(exc).__name__,traceback=traceback.format_exc());save();raise
    finally:
        guard.remove();save()
    print('INT64_MODEL_RERUN_COMPLETE',flush=True)


if __name__=='__main__':main()
