# Exp 21 and long-context decode: summary

Nemotron-UltraLong-8B-1M (`BB_MODEL=nemotron`), Modal H100 80 GB, Oct 30–31 2026. Details and raw numbers: `README.md` (this folder), `benchmarks/exp_21_kascade/phase4/` and `phase5/`.

## Which option to use

| Goal | Best option | Why |
|---|---|---|
| Long context (~1M) with follow-up questions, cheap | **v3 + eviction, keep 64K** (instead of 16K) | 85/100 follow-up facts vs 62/100, fast decode, fits one H100 with FP8 |
| Best fact retention at 1M | **Compressed full cache, 4-bit, evenly spaced codebook** | 91/100 follow-up facts (optimal codebook 92), 54 GB; attention kernel as fast as BF16 |
| Faster decode for every arm | **CUDA-graph decode + FP8 weights** | v3 at 128K: 25.5 → 12.5 ms/token, same answers |
| Exp 21 (route reuse) | Not recommended | Fastest decode at 128K, but worse than eviction at 1M and needs the full cache |

## Method comparison

| Method | Decode ms/token at 128K (graphs, FP8) | 128K single needle | 1M follow-up facts (of 100) | 1M needle (of 25) | 1M long answer (of 10) | Peak GPU memory at 1M | Notes |
|---|---|---|---|---|---|---|---|
| v3, dense decode (full cache) | 12.5 | 30/30 | — | — | — | doesn't fit | full BF16 cache at 1M is ~134 GB |
| v3 + eviction, 16K kept (current default) | 8.8 | — | **62** | 20 | 3 | not logged | loses most facts from the first third of the document |
| v3 + eviction, 64K kept | ~9 (not measured) | — | **85** | 19 | 4 | 65 GB | +23 vs default, p ≈ 2×10⁻⁷ |
| v3 + eviction, 128K kept | ~9 (not measured) | — | **86** | 21 | 5 | 74 GB | no gain over 64K |
| Compressed full cache, 4-bit (optimal codebook) | — | — | **92** | 21 | 3 | 54 GB | +31 vs default; decode kernel 2.9 ms/layer at 1M |
| Compressed full cache, 4-bit (evenly spaced codebook) | — | — | **91** | 19 | not run | 54.6 GB | kernel 1.26 ms/layer at 1M (BF16 attention: 1.38); ~20% more rounding error |
| Exp 21 (route reuse + sparse decode, full cache) | **7.7** | 30/30 | worse than eviction (5-prompt test) | — | 0/4 | needs CPU offload | 1M: follow-up fact 2/5 vs eviction 4/5, even with 8× budget |

1M follow-up test: 25 held-out ~979K-token prompts, needle question then 4 planted facts at 15/35/55/80% depth, paired
across methods (exact McNemar test). Facts kept by depth, 15% → 80%: eviction 16K 16/48/84/100%, 64K 60/80/100/100%,
compressed 84/88/100/96% (evenly spaced codebook 80/88/100/96%). Evenly spaced vs optimal codebook: +2/−3 facts, p = 1.

## GPU kernels

All timings on the H100 unless noted. The decode, CUDA-graph, FP8 and residual changes work for every arm, not just exp 21.

| Change | What it does | Result |
|---|---|---|
| CUDA-graph decode (`modal_rd/graph_decode.py`) | Captures one decode step once, replays it per token (no Python or launch overhead) | v3 at 128K: 25.5 → 15.1 ms/token (1.7×); same tokens as HF `generate` |
| Split-K decode attention (`kernels/bigger_bird_decode.py`) | One query over sink + window + routed keys, read in place; dense mode = flash-decoding | Sparse mode per layer: 0.029 vs 0.181 ms dense at 128K (6×), 0.028 vs 0.695 ms at 512K (24×). Dense mode on par with SDPA (3080) |
| Fused anchor routing (`kernels/bigger_bird_route.py`) | Scores the 4 query heads of a KV head in one pass, reading each key once | 1.9× faster than PyTorch routing (3080, 128K); exp 21 graphed 11.0 → 10.3 ms/token |
| Weight-only FP8 (`modal_rd/fp8.py`, `kernels/fp8_gemv.py`) | FP8 weights, FP8 GEMV for decode | Weights 16 → 9.9 GB, −2.6 ms/token; v3 at 128K 15.1 → 12.5 ms/token; needle 30/30 |
| In-place residual adds (`modal_rd/inplace_residual.py`) | Exact; avoids a copy of the hidden states per layer | ~2 GB less prefill peak at 128K (~16 GB est. at 1M) |
| Graph buffers sized from the kept cache | With eviction, decode buffers hold only the kept tokens | v3 + eviction at 128K: peak 41 → 32.6 GB |
| Compressed-cache decode (`kernels/turbo_decode.py`) | Fused 4-bit decode: rotate the query once, decode codes in registers, un-rotate the output once | Per layer at 1M: optimal codebook 3.11 → 2.92 ms after tuning; evenly spaced codebook 1.26 ms vs BF16 SDPA 1.38 ms, with a 4× smaller cache. At 128K: 0.43 / 0.21 vs 0.18 ms |

End to end at 128K (graphs + FP8): v3 12.5, v3 + eviction 8.8, exp 21 7.7 ms/token, all with the same answers.

Compressed-cache kernel profile (Nsight Compute, H100, 1M): the optimal codebook is compute-bound (the codebook
lookup); the evenly spaced one is latency-bound (25% occupancy, 27% of memory bandwidth). Capping registers sped up
the optimal codebook ~7% but not the evenly spaced one; more splits did not help either (3080 only). Even with the
decode step skipped (wrong-output ablation) the kernel is ~3× above the memory limit, so the remaining cost is
the kernel's structure (4 query heads padded to 16 rows for tensor cores), not the decode.

## What else changed

- Exp 21: route reuse in decode only (`decodeonly`), separate decode window / budget (`dw`, `dv`), span pooling (`sp`), CPU-offloaded cache (`offload_cache.py`).
- Evals: `eval_decode.py` (long answers, second questions), `eval_facts.py` (1M planted-fact follow-ups), `h100_plan.py` / `run_h100.sh` (H100 stages), `bench_decode.py` / `bench_turbo.py` (kernel benchmarks), `profiling/` (Nsight Compute on Modal).
- Fix: exp 20's decode loop gave evicted runs wrong positions.

## Findings
1. At 128K every method gives the same answers; differences are speed and memory.
2. At 1M the default eviction budget (16K per KV head, ~1.6% of the prompt) loses 38% of facts asked as follow-ups, mostly early in the document. 64K fixes most of it.
3. Keeping everything compressed (4-bit) beats dropping tokens: 92 vs 85 facts kept. TurboQuant's paper reports the same direction up to ~100K.
4. Exp 21's cross-layer route reuse works at 128K but not at 1M, and cannot be combined with per-layer eviction (layers keep different tokens).
5. At batch 1, reading the weights (~3 ms/token with FP8) is the main remaining decode cost at 128K. At 1M attention dominates (~40 ms/token over 32 layers with the compressed cache).

## Next steps (parked)
- Make the evenly spaced codebook the default for the compressed cache (1M facts 91 vs 92, same within noise, 2.3× faster kernel).
- Compressed-cache kernel: test more splits on the H100 (~$0.20); then restructure it so the tensor-core padding is not wasted (e.g. one block over several KV heads, or a non-tensor-core inner product with better reuse).
- Raise the eviction default to 64K for long-context runs (team decision).

## Ideas from DeepSeek-V4.1-Flash

V4.1 is built around shrinking the KV cache for 1M context. It shares the overall shape of v3 (local window +
top-k global attention in every layer) and also reuses selections across layers, like exp 21, but it is trained
for all of it. Experiments we could run without training:

| Idea | In V4.1 | What we would test |
|---|---|---|
| Hierarchical candidate pool | The first indexer scores the whole cache and keeps the top 2048 blocks of 8 tokens; later layers only score that pool | Exp 21 / v3 decode: a full scan in the first anchor only, later anchors re-score its pool. Bounds the anchor cost, our largest decode cost at long context |
| Standard FP4 KV format | Main KV in FP4 (E2M1, one FP8 scale per 16 channels), stored after RoPE | Same 1M fact test as our 4-bit cache; simpler than rotation + codebook and natively supported on Blackwell |
| Keep the window in higher precision | Sliding-window KV stays FP8; only the far context is FP4 | Compressed cache with the last 16K tokens in BF16: should recover most of the 4-bit rounding error where attention is strongest |
| Share the KV cache across layers | Reuse layers read an earlier layer's KV, not just its indices | Sharing indices alone failed at 1M for us; sharing storage needs training, so only worth a small zero-shot probe |
| Bounded window replay | On a prompt-cache hit, recompute only the last window's tokens instead of storing the window KV | Relevant if we serve multi-turn sessions with a persisted cache |
| Speculative decoding | A small drafter proposes 5 tokens per step | Batch-1 decode is limited by reading the weights; verifying several drafted tokens per step amortizes that. Needs a draft model for Llama |
