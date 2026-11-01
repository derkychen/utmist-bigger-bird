"""One compressed-cache decode launch per codebook at 1M tokens, for Nsight Compute (ncu) to profile."""
import torch

from kernels import turbo_decode as td
from modal_rd.turbo_kv import TurboQuantizer

n = 1048576
torch.manual_seed(0)
k = torch.randn(8, n, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
q = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
for uniform in (False, True):
    tq = TurboQuantizer(128, 4, "cuda", uniform=uniform)
    (kc, kn), (vc, vn) = tq.encode(k), tq.encode(v)
    for _ in range(3):                      # autotune + warm up (ncu skips these launches)
        td.turbo_decode(q, kc, kn, vc, vn, tq.c_list, tq.rot, scale=128 ** -0.5, variant="dot")
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("profiled_" + ("uniform" if uniform else "lloyd"))
    td.turbo_decode(q, kc, kn, vc, vn, tq.c_list, tq.rot, scale=128 ** -0.5, variant="dot")
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print("done", "uniform" if uniform else "lloyd", flush=True)
