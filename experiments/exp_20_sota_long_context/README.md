# Exp 20 — Bigger Bird v3 vs dense on long-context benchmarks

RULER/NIAH is synthetic. This experiment checks whether Bigger Bird v3's sparse prefill holds up on published
long-context benchmarks: long-document QA, code-repository QA and bug finding in real code.

> **Status (2026-10-12): pilot done, full run NOT done.** The A100 run was stopped after the pilot, and the
> remaining work moves to an H100. See [What is left to run](#what-is-left-to-run).

## Benchmarks

| task id | benchmark | items | what it tests | prompt tokens (R1 tokenizer) |
|---|---|---:|---|---|
| `lbv2` | LongBench v2 (Bai et al., 2024) | 503 | 4-choice QA over long documents in 6 domains: single/multi-doc QA, long in-context learning, dialogue history, structured data, **code repository understanding** (50) | 8K – 4M, middle-truncated |
| `codedebug` | InfiniteBench Code.Debug (Zhang et al., 2024) | 394 | find the function with a planted bug in a real Python repo | ~180K+, middle-truncated |
| `lcqa32k` / `lcqa64k` / `lcqa128k` | LongCodeBench LongCodeQA (Rando et al., 2025) | 113 / 76 / 92 | 4-choice questions about real GitHub repos (issues, behaviour, API changes) | 3–32K / 31–65K / 67–128K |

Short code benchmarks (LongBench v1 LCC and RepoBench-P, HumanEval and so on) are left out on purpose. Their prompts
are under 16K tokens, and Bigger Bird v3's exact window is 16K, so on them it computes exactly the same attention
as dense.

## Protocol

- Model: DeepSeek-R1-Distill-Llama-8B, BF16, batch 1, greedy.
- Prompt: R1 chat template, then the benchmark's official template. The context is middle-truncated to
  `--max-tokens` (keeping its start and end, LongBench style). After that comes an empty `<think></think>` block
  and the cue `The correct answer is (`.
- Answer (no-CoT): the argmax over the next-token probabilities of `A`–`D`. This is LongBench v2's no-CoT
  setting, read from the logits instead of a regex. All four letter probabilities are stored, so the report can
  measure how far a sparse arm's answer distribution moves from dense (TV distance). That is more sensitive than
  accuracy when accuracy is near chance.
- Paired: every example is tokenised once and run through every arm in the same process.
- Arms:
  - `dense`: unmodified HF SDPA (exp 0 equivalent).
  - `bb_v3`: `modal_rd.attention.RDAttention` with `{"sink":64,"window":16384,"vertical":512,"exact_tail":256}`,
    the best all-sparse config from the modal_rd NIAH work. Sparse prefill, exact decode on the full KV cache.
  - `bb_v3_evict`: the same prefill (so the same answer letter), then KV eviction (PR #19 T1:
    `evict_pool=8, evict_keep=8192, evict_window=16384`). This arm measures the memory and decode savings that
    sparsity can actually deliver. Without eviction the full KV cache dominates memory and both arms peak
    at the same level.
- Metrics, per example and arm:
  - accuracy, plus the letter probabilities;
  - prefill seconds (CUDA events around one forward);
  - decode ms/token (16 greedy tokens from the cache);
  - peak allocated GB.
- Every arm uses the exact sequence-chunked MLP (`--mlp-chunk 8192`), so that 120K prompts fit in 40 GB.
- `kernels/bigger_bird_flash.py`: sequence lengths, strides and `WINDOW` are now runtime arguments, no longer `tl.constexpr`
  (the exact tail passes `window=n`).
  - Before: every new prompt length triggered a ~2 s JIT recompile. Real benchmarks have a new length on almost
    every item, so the sparse arm's timings would have been charged for compiles.
  - Output is bit-identical to the old kernel.
  - Steady-state kernel time is within ±5% of the old kernel at 32K–120K (A100).

## What is left to run

Done on the A100 (tag `a100_120k`, prompts middle-truncated to 120K; raw JSONL in `~/scratch/results/exp20/a100_120k/`
on the A100 box):

| task | items | done (all 3 arms) | left |
|---|---:|---:|---:|
| LongCodeQA 32K / 64K / 128K | 113 / 76 / 92 | all 281 | 0 |
| InfiniteBench Code.Debug | 394 | 100 pilot (+11 after the pilot, not in RESULTS.md) | **294** |
| LongBench v2 | 503 | 100 pilot | **403** |

**To do on an H100 80GB (Modal), tag `h100_130k`:**

1. **Rerun the whole suite rather than only the missing items.**
   - Use `bash experiments/exp_20_sota_long_context/run_h100.sh h100_130k`: all 1,178 items, 3 arms,
     `--max-tokens 130000`.
   - Mixing A100 and H100 rows would make the speed and memory columns meaningless.
   - The 80 GB card also removes the 120K truncation, so LongCodeQA 128K runs at full length (max 128,463 tokens).
   - LongBench v2 and Code.Debug are still middle-truncated, to 130K.
2. **Budget about 10–12 H100-hours**, about 5 h wall with the 3 parallel containers. This is an estimate: about
   2× the A100's ~70 s per item for all 3 arms at 120K.
   - Each container has a 6 h Modal timeout.
   - If one is killed, rerun the same command; runs resume from the JSONL.
3. **Pull the results, run `report.py --tag h100_130k` and paste its tables into `RESULTS.md`.** `RESULTS.md` is
   hand-curated, so don't `--out` over it. The commands are in the header of `run_h100.sh`.
   - Then replace the pilot tables below.
   - Check the new speedups against the A100 ones: the H100's FlashAttention is faster, so the crossover length
     may move.
4. **Optional follow-ups, not yet scripted:**
   - A CoT / long-generation eval. It is the only way to test whether KV eviction hurts decode quality, because
     under no-CoT `bb_v3_evict` gives the same answer as `bb_v3` by construction.
   - A `window=4096` arm, to probe whether a smaller window keeps accuracy while winning on speed below 64K.

## Running

```bash
source ~/bb-venv/bin/activate && export SCRATCH=~/scratch   # data -> $SCRATCH/sota_data, results -> $SCRATCH/results/exp20
# A100 40GB (what the pilot used): pilot, then the full sets (resumable), prompts capped at 120K tokens
bash experiments/exp_20_sota_long_context/run_a100.sh a100_120k
# Any subset / other arms
python -m experiments.exp_20_sota_long_context.run --tasks lcqa64k --limit 20 --max-tokens 65536 --tag quick \
    --arms '{"dense":null,"bb_w4k":{"sink":64,"window":4096,"vertical":512,"exact_tail":256}}'
python -m experiments.exp_20_sota_long_context.report --tag a100_120k   # tables -> $SCRATCH/results/exp20/<tag>/RESULTS.md
# Modal H100 80GB: the full run that is still to do (see above); results land in /vol/results/exp20/<tag>
bash experiments/exp_20_sota_long_context/run_h100.sh h100_130k
```

## Results — pilot (A100 40GB, prompts ≤120K, 481 paired examples)

[RESULTS.md](RESULTS.md) has the full record: what was done, the kernel fix, the length probe, what is left, and the
generated tables over all 492 finished examples (the pilot + 11 more Code.Debug items). The table below is the pilot
cut (`report.py --cap codedebug=100,lbv2=100`, 481 examples). The pilot
covers all of LongCodeQA plus the first 100 items (seeded shuffle) of Code.Debug and LongBench v2. **The full
Code.Debug and LongBench v2 sets have not been run.** Those two rows rest on 100 items each and are near chance,
so treat them as provisional until the H100 run.

| | n | dense | bb_v3 | Δ | McNemar p (bb right only / dense right only) |
|---|---:|---:|---:|---:|---|
| LongCodeQA 32K | 113 | 68.1 | 68.1 | 0.0 | 1.00 (0/0) |
| LongCodeQA 64K | 76 | 63.2 | 63.2 | 0.0 | 1.00 (2/2) |
| LongCodeQA 128K | 92 | 54.3 | 54.3 | 0.0 | 1.00 (5/5) |
| LongBench v2 (100 of 503) | 100 | 29.0 | 26.0 | −3.0 | 0.58 (5/8) |
| InfiniteBench Code.Debug (100 of 394) | 100 | 24.0 | 22.0 | −2.0 | 0.79 (6/8) |
| **All** | 481 | 47.4 | 46.4 | −1.0 | 0.53 (18/23) |

| prompt length | dense prefill | bb_v3 prefill | speedup | peak GB dense / bb_v3 / bb_v3_evict | decode ms/tok dense / evict |
|---|---:|---:|---:|---|---|
| 16–32K | 2.5 s | 4.0 s | 0.63× | 20.9 / 20.9 / 20.8 | 27 / 26 |
| 32–64K | 7.3 s | 8.3 s | 0.88× | 25.7 / 25.7 / 22.4 | 33 / 26 |
| 64–96K | 15.2 s | 13.4 s | 1.14× | 31.7 / 31.7 / 24.4 | 42 / 26 |
| 96K+ (mean 117K) | 27.1 s | 18.4 s | **1.47×** | 38.1 / 38.1 / **26.5** | 52 / **26** |

Takeaways:
- **Accuracy: no significant difference from dense anywhere.** On LongCodeQA, the code benchmark the model can
  actually do, accuracy is identical at every length. The disagreements at 128K (12% of items) cancel exactly.
  LongBench v2 and Code.Debug are at or near chance for this 8B model even with dense attention (25% is chance).
  Their small negative deltas come with p > 0.5, and the low agreement there (74–75%) is what near-tied guesses
  look like, not systematic retrieval loss. The full sets will tighten these intervals.
- **Speed: prefill wins only above ~64K.** It reaches 1.47× at ~117K (consistent with the earlier 1.67× at 128K on
  random tokens). It is slower than FlashAttention below ~50K, where the 16K window already covers most of the
  prompt. Decode is identical, since it is exact on the full cache.
- **Memory: the win needs eviction.** Sparse prefill alone peaks exactly like dense, because the full KV cache
  dominates. With PR #19 eviction, peak drops 30% at 96K+ (38.1 → 26.5 GB), and decode is a flat ~26 ms/token at
  every length (vs 52 for dense at 117K). Its answers are identical to bb_v3's, because eviction happens after
  prefill. Under the no-CoT protocol, eviction cannot change the answer; a CoT/long-generation eval would be needed
  to test eviction's decode quality.
- Caveat: no-CoT letter scoring. R1's reasoning mode (CoT) would raise absolute accuracy and stress decode, but it
  costs minutes per item at these lengths.
