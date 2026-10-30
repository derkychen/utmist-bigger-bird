"""Kernel vs explicit masked softmax at long context / large budgets (sampled rows)."""
import torch

from kernels.bigger_bird_flash import bigger_bird_flash


def check(n, window, budget, sink=64, h=32, d=128, rows=64):
    torch.manual_seed(0)
    Q, K, V = (torch.randn(h, n, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    Q = Q * d ** -0.5 * 3
    idx = torch.stack([torch.randperm(n - window - sink, device="cuda")[:budget] + sink
                       for _ in range(h)]).sort(-1).values.int()
    out = bigger_bird_flash(Q, K, V, idx.view(h, 1, -1), front=sink, window=window, num_heads=h, scale=1.0)
    qs = torch.cat([torch.linspace(0, n - 1, rows, device="cuda").long(), torch.arange(n - 8, n, device="cuda")])
    kp = torch.arange(n, device="cuda")[None, :]
    routed = torch.zeros(h, n, dtype=torch.bool, device="cuda").scatter_(1, idx.long(), True)
    worst = 0.0
    for h0 in range(0, h, 8):
        hs = slice(h0, h0 + 8)
        mask = (kp <= qs[:, None]) & ((kp < sink) | (kp > qs[:, None] - window))
        mask = mask[None] | (routed[hs][:, None, :] & (kp <= qs[:, None])[None])
        s = (Q[hs][:, qs].float() @ K[hs].float().transpose(-1, -2)).masked_fill(~mask, -float("inf"))
        ref = s.softmax(-1) @ V[hs].float()
        worst = max(worst, (out[hs][:, qs].float() - ref).abs().max().item())
    print(f"n={n} window={window} budget={budget} max_err={worst:.4g}", flush=True)
    return worst


if __name__ == "__main__":
    for n, w, b in ((4096, 512, 256), (131072, 4096, 2048), (131072, 4096, 4096),
                    (131072, 8192, 8192), (65536, 4096, 4096), (131072, 2048, 1024)):
        check(n, w, b)
