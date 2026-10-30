"""Safe int32 overflow reproduction plus long strided kernel identity/reference checks."""
import hashlib
import json
from pathlib import Path
import torch
import triton
import triton.language as tl
from kernels.bigger_bird_flash import bigger_bird_flash
from kernels._bigger_bird_flash_int32 import _attention as old_attention


@triton.jit
def offsets(OUT32, OUT64):
    p = 524286 + tl.arange(0, 8)
    tl.store(OUT32 + tl.arange(0, 8), (p * 4096).to(tl.int64))
    tl.store(OUT64 + tl.arange(0, 8), p.to(tl.int64) * 4096)


def main():
    root = Path('/vol/results');path = root / 'long_indexing_check_20261028.json'
    result = {'status': 'running', 'checks': [], 'gpu': torch.cuda.get_device_name(0)}
    def save():
        path.write_text(json.dumps(result, indent=2))
    save()
    x = torch.empty(8, device='cuda', dtype=torch.int64);y = torch.empty_like(x)
    offsets[(1,)](x,y)
    want = torch.arange(524286,524294,device='cuda',dtype=torch.int64)*4096
    result['old_offsets'] = x.tolist();result['wide_offsets'] = y.tolist()
    assert not torch.equal(x,want) and torch.equal(y,want)
    result['safe_arithmetic_reproduction'] = True
    save()
    for n in (524224,524288,524352,995993):
        h,hk,d=32,8,128
        q = torch.zeros((n,h,d),device='cuda',dtype=torch.bfloat16).permute(1,0,2)
        k = torch.zeros((n,hk,d),device='cuda',dtype=torch.bfloat16).permute(1,0,2)
        v = torch.empty_like(k)
        for head in range(hk):
            v[head].copy_(((torch.arange(n,device='cuda')%97).float()+head*.25)[:,None].expand(n,d))
        idx = torch.empty((h,1,0),device='cuda',dtype=torch.int32)
        if n == 524352:
            # Compile the original production kernel without launching its unsafe memory writes.
            out = torch.empty((n,h,d),device='cuda',dtype=q.dtype).permute(1,0,2)
            compiled = old_attention.warmup(q,k,v,idx,q,out,
                *q.stride(),*k.stride(),*v.stride(),*idx.stride(),0,0,*out.stride(),
                n,n,0,d,h,4,0,1,0,triton.next_power_of_2(n),False,1.,64,64,128,
                num_warps=4,num_stages=2,grid=(triton.cdiv(n,64),h))
            (root/'long_indexing_old_kernel.ttir').write_text(compiled.asm['ttir'])
            result['old_kernel_compiled_without_launch'] = True
            del out
        out = bigger_bird_flash(q,k,v,idx,front=0,window=1,num_heads=h,kv_groups=4)
        torch.cuda.synchronize()
        # Identity attention: each query has only its own key. No quadratic reference required.
        for head in range(h):
            assert torch.equal(out[head],v[head//4]), (n,head)
        result['checks'].append({'tokens':n,'identity_all_elements':True,'query_stride':list(q.stride()),'output_stride':list(out.stride())})
        save();print(f'IDENTITY_PASS tokens={n}',flush=True)
        del q,k,v,idx,out
        torch.cuda.empty_cache()
    result['kernel_sha256'] = hashlib.sha256(Path('/root/repo/kernels/bigger_bird_flash.py').read_bytes()).hexdigest()
    result['status'] = 'pass';save();print('LONG_INDEXING_PASS',flush=True)


if __name__ == '__main__':
    main()
