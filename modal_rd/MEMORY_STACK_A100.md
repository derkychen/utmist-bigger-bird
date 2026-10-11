# Sparse prefill memory stack (#17 → #18 → #19): A100 validation

Tested 2026-10-11 on one **NVIDIA A100-SXM4-40GB**: torch 2.13.0+cu130, triton 3.7.1, transformers 5.15.1
(the same versions as `modal_rd/app.py`), model `DeepSeek-R1-Distill-Llama-8B` in BF16.

| PR | Branch | Change |
|---|---|---|
| #17 A | `mem-a-gqa-kernel-tail` | GQA K/V read inside the kernel; exact tail runs through the flash kernel |
| #18 B | `mem-b-mlp-chunk` | `--mlp-chunk N`: sequence-chunked MLP |
| #19 C | `mem-c-kv-evict` | `kv_evict`: optional KV-cache eviction after sparse prefill (default off) |

## Fixes needed to run on GPU (in A, carried up the stack)

1. **Kernel dispatch.** `modal_rd/attention.py` fell back to the reference kernel only on `ImportError`.
   On any machine with Triton installed, CPU or FP32 inputs then raised
   `ValueError: Triton Bigger Bird requires CUDA FP16/BF16`.
   It now picks the kernel at call time: Triton for CUDA FP16/BF16, the pure-PyTorch reference otherwise.
2. **The tests never exercised Triton.** They built FP32 tensors and an FP32 model, which the Triton kernel
   rejects, so all of them failed on CUDA (26/28 for A, 29/36 for C).
   On CUDA they now run in BF16 through the real kernel. On CPU they stay FP32 with the original strict checks.
3. **BF16 tolerances (GPU only).**
   - Old-vs-new routing: old (repeated K) and new (grouped K) BF16 matmuls round differently, so near-ties at the
     top-k boundary flip. The test now requires ≥90% per-head overlap; the observed minimum was 92%, i.e. ≤2 of 24 columns.
     Logits agree to ≤0.034 on a scale of ~2 (atol 5e-2). CPU FP32 still requires exact equality.
   - `kv_evict` decode-equivalence scores: atol 2e-2 on GPU (observed 0.0098). Token sequences match exactly.
     Eviction itself moves scores by ~1.0, so the test still tells the two cases apart.

## Unit tests

| Branch | CUDA (BF16, Triton) | CPU (FP32, reference) |
|---|---|---|
| A | 28/28 pass | 28/28 pass |
| A+B | 33/33 pass | — |
| A+B+C | 36/36 pass | 36/36 pass |

The allocation test (kernel stubbed, n=2048) shows wrapper allocations falling from 27.9 to 19.9 MB, with no K/V-sized copies left.

## GPU checkers

- `modal_rd.kernelcheck`: kernel vs explicit masked softmax up to n=131072 and budget 8192.
  Max error ≤ 0.0079 at every size (BF16). **PASS**
- `modal_rd.selfcheck` with the real model: the full-budget kernel matches SDPA (err 0.0039) and the small budget matches the
  masked reference (err 0.0090). On a 15,943-token NIAH prompt, native HF, `dense_rd` and `vs` produce identical greedy
  output. **SELFCHECK_PASS**
- Grouped kernel at Llama-8B shapes (32 q / 8 kv heads, d=128):
  - `kv_groups=4` output is **bit-identical** to repeated K/V at n=32K and n=131K.
  - Exact tail via `window=n` vs masked SDPA: err ≤ 2.4e-4.
  - Speed concern from #17 resolved; the tail kernel is 6–10× faster than masked SDPA:

| n | r=64 | r=256 | r=512 |
|---|---|---|---|
| 131K | 9.4 vs 58 ms | 11.7 vs 100 ms | 17.1 vs 164 ms |

## End-to-end NIAH (real model)

Setup:
- `eval_niah`, `--bytes 261000` (~65K tokens), depth 0.5, `--warm first`, **n=4** (an n=3 sweep gave the same picture).
- Sparse arm: `{"sink":64,"window":4096,"exact_tail":256,"vertical":512,"vertical_scope":8192}`, the config of
  `results/b7_v512_scope8k.json`.
- The 127K config needs about 48 GB even for dense, so it does not fit on 40 GB.
- "base" is `c027c86`, the parent of A.

| Run | Code | peak_gb | exact | prefill | tokens identical to previous row |
|---|---|---:|---:|---:|---|
| dense | base | 32.39 | 4/4 | 10.14 s | — |
| dense `--mlp-chunk 8192` | A+B | 28.39 | 4/4 | 10.24 s | 4/4 vs dense base |
| vs | base | **36.46** | 4/4 | 8.11 s | — |
| vs | A | 32.40 | 4/4 | 6.98 s | **4/4** |
| vs `--mlp-chunk 8192` | A+B | 28.39 | 4/4 | 6.39 s | **4/4** |
| vs `--mlp-chunk 8192`, kv_evict off | A+B+C | 28.39 | 4/4 | 6.39 s | **4/4** |
| vs `--mlp-chunk 8192`, **kv_evict on** | A+B+C | **20.91** | **2/4** | 6.40 s | 0/4 |

## Does the stack change behaviour?

- **A: no.** Generated tokens are identical to base on every example, and the sparse arm's 4 GB excess over dense is gone
  (36.46 → 32.40 GB, now equal to dense). Routed-column statistics move by <0.01 because of BF16 near-ties; outputs do not change.
- **B: no.** The MLP op is exact, and tokens are identical for both dense and sparse, saving another 4 GB.
  It is not a sparse-specific win, so compare arms with the same `--mlp-chunk`.
- **C with the default (`kv_evict` off): no.** Tokens are identical and memory is unchanged.
- **C with `kv_evict` on: yes, decay as expected.** The KV cache shrinks (−7.5 GB peak at 65K), but NIAH drops from 4/4 to 2/4.
  Failures include truncated or off-by-one needles (`1560` for `1560400`, `7400576` for `7400577`).
  Decode only sees the retained keys, and with `window=4096` / `vertical=512` the needle's digits are not all retained.
  Do not enable it for accuracy runs without tuning the window and budget and running a larger eval.

## Tuning `kv_evict` (n=10 per task, 30 for single NIAH)

Same sparse arm and `--mlp-chunk 8192`, `--bytes 261000` (52-84K tokens), depth 0.5 unless noted. Multi-needle tasks use
`--max-new 64` (`multikey_*`: 40). "off" = sparse prefill with eviction disabled. Peak GB is for the multiquery run
(single NIAH for that row).

| Task | dense | off | E0: PR default | P1: `evict_pool=8` | **T1: P1 + `evict_keep=8192`, `evict_window=16384`** |
|---|---:|---:|---:|---:|---:|
| niah, depths .1/.5/.9 | 30/30 | 30/30 | 26/30 | 30/30 | **30/30** |
| niah_multikey_1 | — | 10/10 | 10/10 | 10/10 | **10/10** |
| niah_multikey_3 | 5/10 | 8/10 | 7/10 | 8/10 | **8/10** |
| niah_multivalue | 4/10 | 4/10 | 1/10 | 8/10 | **6/10** |
| niah_multiquery | 7/10 | 10/10 | 7/10 | 7/10 | **10/10** |
| peak GB (multiquery / niah) | — | 26.07 / 28.39 | 20.15 / 20.91 | 20.15 / 20.91 | 22.53 / 23.29 |

Recommended: **T1** `{"kv_evict":true,"evict_pool":8,"evict_keep":8192,"evict_window":16384}`. It matches or beats
"off" on every task and saves 3.5-5 GB at this length. P1 is the memory-first option, with no extra memory over the PR
default; it fixes single NIAH and multivalue but still drops one of the four values on 3/10 multiquery prompts.
Defaults are unchanged (all new knobs = 0 reproduce the PR's policy).

What was tried, on multiquery (the hardest case for eviction):

| Variant (on top of P1 unless noted) | kept / KV head | multiquery |
|---|---:|---:|
| `evict_keep=8192` | ~12.3K | 6/10 |
| `evict_pool_back=16` / `32` (also keep the text before a key) | ~6.2K | 7/10 / 6/10 |
| `evict_full_layers=2` (layers 0-1 unevicted) | ~6.2K | 6/10 |
| `evict_obs=256` (+`evict_obs_max`), probe on the 4 failing prompts | ~6.2K | no change |
| `evict_keep=4096, evict_window=8192` | ~12.3K | 6/10 |
| `evict_keep=8192, evict_window=8192` | ~16.4K | 8/10 |
| `evict_keep=16384, evict_window=8192` | ~24.6K | 9/10 |
| **`evict_keep=8192, evict_window=16384` (T1)** | ~24.6K | **10/10** |
| `evict_keep=16384, evict_window=16384` | ~32.8K | 10/10 |
| `evict_keep=32768` | ~36.9K | 10/10 |

Findings:
- `evict_pool` (keep the 7 tokens after each high-scoring key, i.e. the rest of a number) fixes the truncated-needle
  failures on single NIAH for free.
- Multiquery failures under E0/P1 drop one of the four values entirely and repeat another. Coverage is not the cause: the
  dropped value is kept by 75-100% of KV heads in the layers that read it, and in 2 of 3 cases it sits inside the
  always-kept sink. Better scoring (observation window, max-pooling, longer spans, full early layers) does not help.
  More cache does, and a longer recent window helps more than more routed keys at the same size.
- Decode after eviction is exact given the kept set (`test_decode_matches_masked_full_cache_reference`, RoPE positions
  included), so the gap comes from the eviction policy, not the implementation.
