# Nemotron bring-up on Modal

## Latest results (October 30, 2026)

The later 64-bit indexing fix corrected the near-1M failure described in the
historical probe below. On 30 warmed synthetic single-needle prompts at
995,636–998,563 tokens, sparse inference scored 30/30. A single matched-prompt
dense no-cache prefill took 843.28 s versus 64.25 s sparse (13.12x); this is
not a 30-prompt dense comparison or a generation speedup. See
`results/nemotron_int64_20261028/results-report.md`.

A later sparse-only harder-task pilot scored 3/3 on essay single-needle,
3/3 on essay UUID, 1/3 on multi-key, 1/3 on UUID distractors, 0/3 on
multi-value, and 1/3 on multi-query by answer-string presence. Its multi-query
scorer does not verify that each answer is paired with the right key. Each
cell has only three examples and has no dense baseline; see
`results/nemotron_hard_1m_20261029/results-report.md`.

Checkpoint: `nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct`, pinned to
`fa3a10b9c6ad02f42c81e1d68dcf6967afb230ef`. Research use: model card lists CC-BY-NC-4.0.
The downloaded weights are FP32 on disk; runners explicitly load BF16.

Set `BB_MODEL=nemotron` for all runners. `BB_MODEL=r1` is the unchanged default.
The model path remains under `$SCRATCH/models/<checkpoint-name>` (Modal: `/vol/models`).
Nemotron defaults to its native user/assistant chat template, encoded without adding a second BOS.
R1 defaults to historical raw prompts and explicit BOS. Exp20 uses the selected model's native chat template outside the historical R1 path, with its existing answer-letter cue. Override with `BB_PROMPT_FORMAT=raw|chat`.
These protocols are separate comparisons; do not pool their results.

CPU-only download and preflight:

```bash
BB_MODEL=nemotron BB_GPU=none modal run modal_rd/app.py::download
BB_MODEL=nemotron BB_GPU=none modal run modal_rd/app.py::run_cpu --module modal_rd.model_preflight
BB_MODEL=r1 BB_GPU=none modal run modal_rd/app.py::run_cpu --module modal_rd.model_preflight
```

Bounded H100 pilot (stops on preflight/parity failure):

```bash
BB_MODEL=nemotron BB_GPU='H100!' modal run --detach modal_rd/app.py \
  --module modal_rd.nemotron_bringup --args '--tag nemotron_chat_pilot_20261028 --n 6'
```

The gate tests full/sparse-union kernel correctness, native-HF versus dense-wrapper last logits
and greedy token IDs on a roughly 16K prompt, then fresh dense-wrapper versus Bigger Bird v3
(sink64/window16384/vertical512/exact-tail256) on six identical prompts at 512000 source units.
Both arms use BF16, greedy max64, KV cache, chunk8192 MLP and CPU embedding, without eviction or INT8.
Every prompt is warmed then timed. Results retain warm predictions, actual token counts, prompt and
source hashes, resolved checkpoint revision, software versions, forward counts, runtime and allocated
peak GPU memory. A summary checks matched hashes and warm/timed equality. This six-prompt pilot is
not a final benchmark and does not establish 1M feasibility.

Download artifacts with `modal volume get bb-vol results/<tag>.json <local-path>`.

Next gates: inspect pilot quality/parity; run a matched 30-prompt confirmation at ~128K; then bounded
256K/512K/1M memory and timing probes. Source length is never a substitute for tokenizer counts.
At 1M, a full BF16 dense KV cache exceeds the tested single H100 80GB; the later
single-GPU sparse feasibility results are linked above.
Budget scaling and eviction should be separate arms after the unmodified-v3 control is established.
Retain the six tuning examples and report the other 24 separately if subsequent tuning uses them.

128K confirmation, including native-HF/dense-wrapper parity on a full ~128K prompt:

```bash
BB_MODEL=nemotron BB_GPU='H100!' modal run --detach modal_rd/app.py \
  --module modal_rd.nemotron_bringup \
  --args '--tag nemotron_chat64_confirm30_20261028 --n 30 --check-bytes 512000'
```

The first six-prompt pilot passed: dense and v3 both 6/6 exact, mean prefill
16.724s versus 7.746s (2.159×), peak allocated memory 38.146GB versus 38.148GB.
This pilot shows no memory saving because eviction is off. Artifacts and independent verification:
`modal_rd/results/nemotron_bringup_20261028/results-report.md`.
The initial max10 attempt was truncated mid-answer; it was stopped and excluded. All compared
max64 outputs finished within 15–18 generated tokens. Confirmation uses the same max64 rule.

## Eviction and scaling probe

`modal_rd.nemotron_scaling` first reproduces the six pilot prompt manifests with a
shorter dataset prefix (validation has an independent seeded RNG). It checks the
checkpoint, attention source and kernel source against the completed control.
It then evaluates the merged memory default: v3 plus `kv_evict=true`,
`evict_keep=16384`, `evict_pool=8`, retaining BF16 cache, chunk8192 and CPU embeddings.
Six warmed/timed 128K prompts must all recover the full number before longer probes start.

```bash
BB_MODEL=nemotron BB_GPU=none modal run modal_rd/app.py::run_cpu \
  --module modal_rd.nemotron_scaling --args '--tag nemotron_prefixcheck_20261028 --cpu-only'
BB_MODEL=nemotron BB_GPU='H100!' modal run --detach modal_rd/app.py \
  --module modal_rd.nemotron_scaling --args '--tag nemotron_evict_scale_20261028'
```

Longer phases use one warmed then timed prompt each at targets 262144, 524288 and
1000000 actual tokens (calibration tolerance 0.5%; always report measured count).
They record full-number accuracy, prefill/task time, allocated/reserved memory,
cache retention counts and prompt hashes. These single-example phases measure
feasibility and expose quality failures. Dense comparisons beyond 128K are pending;
they cannot establish speedup over dense at those lengths. OOM or another exception
is saved with partial results and stops the job. There is no automatic H200 retry.

Measured probe outcomes (2026-10-28; one H100, pinned Nemotron BF16):

| Actual tokens | Exact | Timed prefill s | Peak allocated GB |
|---|---|---|---|
| ~127.6K | 6/6 | 7.760 | 26.295 |
| 261528 | 1/1 | 15.836 | 33.491 |
| 522505 | 1/1 | 31.686 | 47.529 |
| 995993 | 0/1 | 55.731 | 72.999 |

The near-1M prompt fit but returned 64 exclamation marks, identically in warm and timed passes.
The later 64-bit indexing correction identified and fixed an address-overflow bug; see the latest results above.
Artifacts and independent verification: `modal_rd/results/nemotron_next_20261028/`.

`modal_rd.nemotron_prefill_diagnosis` now compares the same prompts at ~128K, ~512K and ~1M
with cache disabled, using v3 and the exact dense wrapper. It also checks in-place RoPE against
HF's formula at large absolute positions. It saves the full last-token logits, top candidates,
source/prompt hashes and failures. These are single cold forwards for diagnosis, with no
full-answer accuracy or latency-benchmark claim. Dense ~1M prefill can take many minutes.
The subsequent 30-prompt sparse confirmation completed; see the latest results above.
