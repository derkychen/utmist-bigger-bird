# Exp 20 results — Bigger Bird v3 vs dense on published long-context benchmarks

> **Status (2026-10-12): partial. The A100 pilot is done; the full run is NOT done.** Left to run: **294 Code.Debug + 403
> LongBench v2 items.** The plan is a full rerun of all 1,178 items on a Modal H100 80GB at 130K tokens
> (`bash experiments/exp_20_sota_long_context/run_h100.sh h100_130k`, est. 10–12 H100-hours). See [What is left](#what-is-left).

**Bottom line so far (492 paired examples, A100 40GB, prompts ≤120K tokens):**
- **Accuracy:** Bigger Bird v3 ties dense; no difference is significant anywhere. Overall it scores 46.1% vs 47.0%
  (McNemar p = 0.64; 19 items right only for Bigger Bird, 23 right only for dense). On LongCodeQA, the code benchmark
  the model can actually do, accuracy is identical at 32K, 64K and 128K.
- **Speed:** prefill is 1.47× faster at ~117K tokens and 1.14× at 64–96K. It is slower than FlashAttention below
  ~50K (0.63× at 16–32K).
- **Memory:** saved only with KV eviction, −30% peak at 96K+ (38.2 → 26.5 GB). Decode with eviction is a flat
  ~26 ms/token at every length, vs 52 ms for dense at 117K. Sparse prefill alone peaks exactly like dense.

## Headline

| benchmark | n | dense | bb_v3 | Δ pp | same answer | McNemar p (bb right only / dense right only) | median TV vs dense |
|---|---:|---:|---:|---:|---:|---|---:|
| LongCodeQA 32K | 113 / 113 | 68.1 | 68.1 | 0.0 | 100% | 1.00 (0/0) | 0.003 |
| LongCodeQA 64K | 76 / 76 | 63.2 | 63.2 | 0.0 | 94.7% | 1.00 (2/2) | 0.021 |
| LongCodeQA 128K | 92 / 92 | 54.3 | 54.3 | 0.0 | 88.0% | 1.00 (5/5) | 0.056 |
| LongBench v2 | 100 / 503 | 29.0 | 26.0 | −3.0 | 75.0% | 0.58 (5/8) | 0.106 |
| InfiniteBench Code.Debug | 111 / 394 | 24.3 | 23.4 | −0.9 | 74.8% | 1.00 (7/8) | 0.112 |
| **All** | 492 | 47.0 | 46.1 | −0.8 | 86.2% | 0.64 (19/23) | 0.043 |

| prompt length | n | dense prefill | bb_v3 prefill | speedup | peak GB: dense / bb_v3 / bb_v3_evict | decode ms/tok: dense / bb_v3_evict |
|---|---:|---:|---:|---:|---|---|
| ≤16K (bb = dense) | 41 | 1.08 s | 1.08 s | 1.00× | 18.8 / 18.8 / 18.8 | 26.8 / 25.8 |
| 16–32K | 99 | 2.52 s | 3.98 s | 0.63× | 20.9 / 20.9 / 20.8 | 27.1 / 25.7 |
| 32–64K | 88 | 7.25 s | 8.26 s | 0.88× | 25.7 / 25.7 / 22.4 | 33.2 / 25.7 |
| 64–96K | 131 | 15.27 s | 13.37 s | 1.14× | 31.7 / 31.7 / 24.4 | 42.3 / 25.8 |
| 96K+ (mean 117K) | 133 | 27.16 s | 18.42 s | **1.47×** | 38.2 / 38.2 / **26.5** | 51.9 / **25.7** |

`bb_v3_evict` has the same prefill as `bb_v3`, so it gives the same answers; its accuracy columns match `bb_v3` exactly.

## What we did

1. **Picked benchmarks that can tell sparse from dense.**
   - Chosen: LongBench v2 (503 items, 6 domains incl. code-repo QA), InfiniteBench Code.Debug (394 items, planted
     bug in a real Python repo) and LongCodeBench LongCodeQA (2025; 281 items, real GitHub repos, 32K/64K/128K groups).
   - Dropped short code benchmarks (LongBench v1 LCC / RepoBench-P, HumanEval). Their prompts are under 16K tokens
     and Bigger Bird v3's exact window is 16K, so there it computes exactly what dense does.
   - Checked the token lengths with the R1 tokenizer:
     - LongBench v2 code items run 24K–4.1M tokens.
     - Code.Debug items are ~180K+.
     - LongCodeQA groups have medians of 22K / 49K / 79K tokens.
2. **Built the harness** (`tasks.py`, `run.py`, `report.py`).
   - Each benchmark uses its official prompt template inside R1's chat template.
   - Over-long contexts are middle-truncated, LongBench style.
   - The answer is read from the A–D next-token probabilities after an empty `<think></think>` and
     `The correct answer is (`. This is LongBench v2's no-CoT setting, read from logits instead of a regex.
   - Each example is tokenised once and run through all three arms in the same process (paired).
   - Per example and arm it records:
     - accuracy and all 4 letter probabilities
     - prefill seconds (CUDA events)
     - decode ms/token over 16 greedy tokens
     - peak allocated GB
   - The run is resumable from the JSONL.
3. **Arms.**
   - `dense`: unmodified HF SDPA, equivalent to exp 0.
   - `bb_v3`: the best all-sparse config from the modal_rd NIAH work,
     `{"sink":64,"window":16384,"vertical":512,"exact_tail":256}`, with exact decode on the full KV cache.
   - `bb_v3_evict`: the same, plus PR #19's recommended KV eviction (`evict_pool=8, evict_keep=8192,
     evict_window=16384`).
   - Every arm uses the exact chunked MLP (`--mlp-chunk 8192`), so 120K fits in 40 GB.
4. **Measured length scaling on random tokens** to set the length cap. Results are in the length-probe table below.
   - 131K fits on the A100 at 40.8 GB peak; 120K was chosen to leave headroom.
5. **Found and fixed two Triton recompilation bugs** in `kernels/bigger_bird_flash.py`. They were inflating
   `bb_v3`'s timings on real prompts. Details in the kernel fix section below.
   - The first was found in the smoke test: `bb_v3` came out slower than `bb_v3_evict`, which runs the same prefill.
   - The second was found 30 items into the first full run. That run was discarded, and the run restarted from
     scratch with the fixed kernel.
6. **Ran the pilot** (`run_a100.sh`, tag `a100_120k`): all 281 LongCodeQA items, then the first 100 items (seeded
   shuffle) of Code.Debug and of LongBench v2.
   - No OOMs and no crashes.
   - 11 more Code.Debug items finished before the run was stopped at the user's request on 2026-10-12. They are
     included here.

## Findings

- **Accuracy is preserved where the model can do the task.** LongCodeQA (54–68% for dense) matches dense exactly
  at every length.
  - At 128K, Bigger Bird picks a different letter on 12% of items, but its 5 extra right answers cancel its 5 extra
    wrong ones.
  - The answer distributions drift slowly with length: median TV is 0.003 at 32K, 0.021 at 64K and 0.056 at 128K.
    The routing does change the computation, but not the answer quality.
- **LongBench v2 and Code.Debug are near chance for this 8B model even with dense attention** (29% and 24%; chance
  is 25%).
  - Their small negative deltas (−3.0 and −0.9 pp) have p ≥ 0.58.
  - Their ~75% agreement and higher TV (~0.11) are what near-tied guesses look like: small changes in attention flip
    them.
  - These two rows are the least informative, and they are also the incomplete ones.
  - LongBench v2 by domain (code-repo QA: n = 10) is too small to read; see the full tables.
- **Speed: the break-even is ~64K on the A100.**
  - Below that the 16K window covers most of the prompt, and routing plus the Triton kernel cost more than
    FlashAttention.
  - On real prompts the speedup is 1.47× at ~117K, consistent with 1.67× at 131K on random tokens.
  - Decode is identical for `dense` and `bb_v3`, since decode is exact on the full cache.
- **Memory: sparse prefill alone saves nothing on peak**, because the full KV cache (~16 GB at 120K) plus weights
  dominate.
  - With eviction, peak falls 30% at 96K+ and decode cost stops growing with length.
  - Under no-CoT scoring, eviction cannot change the answer, since the answer comes from the prefill logits. Its
    decode quality is untested here.

## Kernel fix: `kernels/bigger_bird_flash.py`

- **Bug 1:** sequence lengths `N`, `T`, `OFFSET`, all strides and `ROUTE_CHUNK` were `tl.constexpr`.
  - Every new prompt length compiled a new kernel (~2 s).
  - Real benchmarks have a new length on almost every item; `modal_rd` hid this by warming each fixed-length NIAH
    prompt.
- **Bug 2:** the exact tail calls the kernel with `window=n`, and `WINDOW` was constexpr, so it also recompiled per
  length.
- **Fix:** all of these are now runtime arguments.
  - Triton still specialises integers on `==1` / `%16`, so only the first even and the first odd length compile.
    The harness warmup covers both.
- **Checks:**
  - Output is bit-identical to the old kernel (`torch.equal` at 32K / 64K / 120K, main and tail calls).
  - The 40 CUDA tests pass (`test_bigger_bird_flash`, `test_rd_attention_gqa`, `test_kv_evict`,
    `test_bigger_bird_cache`).
  - `modal_rd.kernelcheck` passes to 128K (max err ≤ 0.008).

| n | main call: old → new | exact tail (256 q): old → new | first call at a new length |
|---:|---:|---:|---|
| 32K | 85.0 → 82.0 ms | 2.50 → 3.13 ms | ~2 s compile → 0.10 s |
| 64K | 158.0 → 163.9 ms | 4.98 → 5.00 ms | ~2 s compile → 0.16 s |
| 120K | 288.9 → 299.8 ms | 9.16 → 9.19 ms | ~2 s compile → 0.29 s |

These timings are per layer (32 query / 8 KV heads, d = 128, A100). Steady state is within ±5%.

## Length probe (random tokens, A100 40GB, `--mlp-chunk 8192`)

| tokens | dense prefill | bb_v3 prefill | speedup | decode ms/tok (both) | peak GB (both) |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 4.07 s | 4.88 s | 0.83× | 28 | 22.4 |
| 65,536 | 10.23 s | 9.76 s | 1.05× | 37.5 | 28.5 |
| 98,304 | 19.95 s | 14.68 s | 1.36× | 46.9 | 34.6 |
| 114,688 | 26.01 s | 17.11 s | 1.52× | 51.5 | 37.7 |
| 131,072 | 33.20 s | 19.85 s | 1.67× | 56.1 | 40.8 |

Taken with the old kernel after a warm call at each length, so there are no compile effects.

## What is left

| task | items | done (all 3 arms) | left |
|---|---:|---:|---:|
| LongCodeQA 32K / 64K / 128K | 113 / 76 / 92 | 281 | 0 |
| InfiniteBench Code.Debug | 394 | 111 | **294** |
| LongBench v2 | 503 | 100 | **403** |

**Next: a full rerun on an H100 80GB (Modal), tag `h100_130k`.**
- Command: `bash experiments/exp_20_sota_long_context/run_h100.sh h100_130k`.
  - It runs all 1,178 items in 3 parallel containers at `--max-tokens 130000`.
  - LongCodeQA 128K then runs uncut (max 128,463 tokens).
  - Rerun the whole suite rather than only the missing items, so speed and memory are not mixed across GPUs.
  - Each container has a 6 h timeout; rerun the same command to resume.
- Then pull `results/exp20/h100_130k` from the `bb-vol` volume.
  - Run `report.py --tag h100_130k`.
  - Replace the tables in this file, and compare the H100 break-even length with the A100's ~64K.
- Optional follow-ups:
  - A CoT / long-generation eval, to test eviction's decode quality.
  - A `window=4096` arm, to probe accuracy vs speed below 64K.

## Caveats

- **No-CoT scoring.**
  - R1 reasoning would raise absolute accuracy and exercise decode.
  - It costs minutes per item at these lengths.
- **Truncation:** LongBench v2 and Code.Debug contexts beyond 120K are middle-truncated: 41 of the 100 LongBench v2
  items and 45 of the 111 Code.Debug items here. 11 LongCodeQA 128K items are truncated too.
- **One GPU type, BF16.** The speed and memory figures are A100 only.

## Reproduce

```bash
source ~/bb-venv/bin/activate && export SCRATCH=~/scratch
bash experiments/exp_20_sota_long_context/run_a100.sh a100_120k          # what produced these numbers
python -m experiments.exp_20_sota_long_context.report --tag a100_120k     # full tables (written to $SCRATCH/results/exp20/<tag>/RESULTS.md)
```
Raw JSONL (one line per example × arm): `~/scratch/results/exp20/a100_120k/*.jsonl` on the A100 box.

## Full tables (generated by `report.py --tag a100_120k`, 492 examples)

`bb_v3_evict` columns are identical to `bb_v3` for accuracy, by construction.

### Accuracy by task

|  | n | dense acc % | bb_v3 acc % | bb_v3_evict acc % | Δ bb_v3 pp | Δ bb_v3_evict pp | agree bb_v3 % | agree bb_v3_evict % | McNemar p (bb_v3) | McNemar p (bb_v3_evict) |
|---|---|---|---|---|---|---|---|---|---|---|
| codedebug | 111 | 24.3 | 23.4 | 23.4 | -0.9 | -0.9 | 74.8 | 74.8 | 1.00 (7↑/8↓) | 1.00 (7↑/8↓) |
| lbv2 | 100 | 29.0 | 26.0 | 26.0 | -3.0 | -3.0 | 75.0 | 75.0 | 0.58 (5↑/8↓) | 0.58 (5↑/8↓) |
| lcqa128k | 92 | 54.3 | 54.3 | 54.3 | +0.0 | +0.0 | 88.0 | 88.0 | 1.00 (5↑/5↓) | 1.00 (5↑/5↓) |
| lcqa32k | 113 | 68.1 | 68.1 | 68.1 | +0.0 | +0.0 | 100.0 | 100.0 | 1.00 (0↑/0↓) | 1.00 (0↑/0↓) |
| lcqa64k | 76 | 63.2 | 63.2 | 63.2 | +0.0 | +0.0 | 94.7 | 94.7 | 1.00 (2↑/2↓) | 1.00 (2↑/2↓) |
| **All** | 492 | 47.0 | 46.1 | 46.1 | -0.8 | -0.8 | 86.2 | 86.2 | 0.64 (19↑/23↓) | 0.64 (19↑/23↓) |

### LongBench v2 by domain

|  | n | dense acc % | bb_v3 acc % | bb_v3_evict acc % | Δ bb_v3 pp | Δ bb_v3_evict pp | agree bb_v3 % | agree bb_v3_evict % | McNemar p (bb_v3) | McNemar p (bb_v3_evict) |
|---|---|---|---|---|---|---|---|---|---|---|
| Code Repository Understanding | 10 | 30.0 | 20.0 | 20.0 | -10.0 | -10.0 | 60.0 | 60.0 | 1.00 (1↑/2↓) | 1.00 (1↑/2↓) |
| Long In-context Learning | 19 | 31.6 | 36.8 | 36.8 | +5.3 | +5.3 | 84.2 | 84.2 | 1.00 (1↑/0↓) | 1.00 (1↑/0↓) |
| Long Structured Data Understanding | 9 | 33.3 | 11.1 | 11.1 | -22.2 | -22.2 | 44.4 | 44.4 | 0.62 (1↑/3↓) | 0.62 (1↑/3↓) |
| Long-dialogue History Understanding | 6 | 33.3 | 16.7 | 16.7 | -16.7 | -16.7 | 66.7 | 66.7 | 1.00 (0↑/1↓) | 1.00 (0↑/1↓) |
| Multi-Document QA | 23 | 17.4 | 26.1 | 26.1 | +8.7 | +8.7 | 91.3 | 91.3 | 0.50 (2↑/0↓) | 0.50 (2↑/0↓) |
| Single-Document QA | 33 | 33.3 | 27.3 | 27.3 | -6.1 | -6.1 | 72.7 | 72.7 | 0.50 (0↑/2↓) | 0.50 (0↑/2↓) |
| **All** | 100 | 29.0 | 26.0 | 26.0 | -3.0 | -3.0 | 75.0 | 75.0 | 0.58 (5↑/8↓) | 0.58 (5↑/8↓) |

### LongBench v2 by difficulty

|  | n | dense acc % | bb_v3 acc % | bb_v3_evict acc % | Δ bb_v3 pp | Δ bb_v3_evict pp | agree bb_v3 % | agree bb_v3_evict % | McNemar p (bb_v3) | McNemar p (bb_v3_evict) |
|---|---|---|---|---|---|---|---|---|---|---|
| easy | 36 | 36.1 | 25.0 | 25.0 | -11.1 | -11.1 | 69.4 | 69.4 | 0.29 (2↑/6↓) | 0.29 (2↑/6↓) |
| hard | 64 | 25.0 | 26.6 | 26.6 | +1.6 | +1.6 | 78.1 | 78.1 | 1.00 (3↑/2↓) | 1.00 (3↑/2↓) |
| **All** | 100 | 29.0 | 26.0 | 26.0 | -3.0 | -3.0 | 75.0 | 75.0 | 0.58 (5↑/8↓) | 0.58 (5↑/8↓) |

### Accuracy by prompt length (after truncation)

|  | n | dense acc % | bb_v3 acc % | bb_v3_evict acc % | Δ bb_v3 pp | Δ bb_v3_evict pp | agree bb_v3 % | agree bb_v3_evict % | McNemar p (bb_v3) | McNemar p (bb_v3_evict) |
|---|---|---|---|---|---|---|---|---|---|---|
| ≤16K (bb = dense) | 41 | 73.2 | 73.2 | 73.2 | +0.0 | +0.0 | 100.0 | 100.0 | 1.00 (0↑/0↓) | 1.00 (0↑/0↓) |
| 16-32K | 99 | 59.6 | 59.6 | 59.6 | +0.0 | +0.0 | 100.0 | 100.0 | 1.00 (0↑/0↓) | 1.00 (0↑/0↓) |
| 32-64K | 88 | 55.7 | 55.7 | 55.7 | +0.0 | +0.0 | 95.5 | 95.5 | 1.00 (2↑/2↓) | 1.00 (2↑/2↓) |
| 64-96K | 131 | 42.7 | 42.0 | 42.0 | -0.8 | -0.8 | 92.4 | 92.4 | 1.00 (2↑/3↓) | 1.00 (2↑/3↓) |
| 96K+ | 133 | 27.8 | 25.6 | 25.6 | -2.3 | -2.3 | 59.4 | 59.4 | 0.73 (15↑/18↓) | 0.73 (15↑/18↓) |
| **All** | 492 | 47.0 | 46.1 | 46.1 | -0.8 | -0.8 | 86.2 | 86.2 | 0.64 (19↑/23↓) | 0.64 (19↑/23↓) |

### Answer-distribution fidelity vs dense

TV = total-variation distance between the arms' A-D distributions; Δp(gold) = arm minus dense probability on the gold letter.

| arm | mean TV | median TV | mean Δp(gold) | mean dense p(gold) |
|---|---|---|---|---|
| bb_v3 | 0.0907 | 0.0427 | -0.0057 | 0.412 |
| bb_v3_evict | 0.0907 | 0.0427 | -0.0057 | 0.412 |

### Speed and memory by prompt length

Prefill = one CUDA-timed forward over the prompt; peak = max allocated during prefill + decode; speedup = dense prefill / arm prefill (ratio of means).

| prompt length | n | mean tokens | dense prefill s | bb_v3 prefill s | bb_v3_evict prefill s | bb_v3 speedup | bb_v3_evict speedup | dense decode ms/tok | bb_v3 decode ms/tok | bb_v3_evict decode ms/tok | dense peak GB | bb_v3 peak GB | bb_v3_evict peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ≤16K (bb = dense) | 41 | 12110 | 1.08 | 1.08 | 1.08 | 1.00x | 1.00x | 26.8 | 25.7 | 25.8 | 18.8 (max 19.4) | 18.8 (max 19.4) | 18.8 (max 19.4) |
| 16-32K | 99 | 23918 | 2.52 | 3.98 | 3.93 | 0.63x | 0.64x | 27.1 | 25.9 | 25.7 | 20.9 (max 22.3) | 20.9 (max 22.3) | 20.8 (max 21.4) |
| 32-64K | 88 | 50827 | 7.25 | 8.26 | 8.28 | 0.88x | 0.88x | 33.2 | 33.1 | 25.7 | 25.7 (max 28.4) | 25.7 (max 28.4) | 22.4 (max 23.3) |
| 64-96K | 131 | 82702 | 15.27 | 13.37 | 13.39 | 1.14x | 1.14x | 42.3 | 42.3 | 25.8 | 31.7 (max 34.6) | 31.7 (max 34.6) | 24.4 (max 25.3) |
| 96K+ | 133 | 116903 | 27.16 | 18.42 | 18.43 | 1.47x | 1.47x | 51.9 | 51.9 | 25.7 | 38.2 (max 38.7) | 38.2 (max 38.7) | 26.5 (max 26.7) |
| **All** | 492 | 68535 | 13.30 | 10.91 | 10.91 | 1.22x | 1.22x | 38.9 | 38.6 | 25.7 | 29.1 (max 38.7) | 29.1 (max 38.7) | 23.4 (max 26.7) |

