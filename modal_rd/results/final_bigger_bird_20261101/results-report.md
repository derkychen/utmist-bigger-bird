# Bigger Bird final benchmark results — updated 2 Nov 2026

## Slide table: matched R1 benchmarks

Dense, `bb_v3`, and `bb_evict64k` were run on the same official examples per suite, using the same R1-Distill-Llama-8B checkpoint, BF16, H100 80GB, 130,000-token prompt cap, seed 0, 16-token greedy CUDA-graph decode, MLP chunking (8,192), and in-place residuals. All three suites completed with every listed item in all three arms and no OOM. Accuracy is the first answer-token distribution renormalized over A–D; it is not code generated or a patch pass rate.

| Test | n | actual tokens, mean (range) | middle-truncated | Dense accuracy | BB v3 accuracy | Δ vs dense | BB answer agrees with dense | Prefill dense → BB (s) | dense/BB mean ratio | paired median ratio | 16-step decode dense → BB (ms/token) | peak allocation dense → BB (GB; max) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LongCodeQA 32K bucket | 113 | 19768 (2,933–32,067) | 0/113 | 76/113 (67.3%) | 76/113 (67.3%) | +0.0 pp | 98.2% | 0.79 → 1.17 | 0.67× | 0.64× | 10.42 → 10.41 | 21.2 → 20.9 (max 23.1/22.9) |
| LongCodeQA 64K bucket | 76 | 49506 (30,998–65,401) | 0/76 | 48/76 (63.2%) | 48/76 (63.2%) | +0.0 pp | 94.7% | 2.55 → 3.17 | 0.80× | 0.79× | 12.08 → 12.08 | 26.7 → 25.7 (max 29.2/28.4) |
| LongCodeQA 128K bucket | 92 | 88050 (67,488–128,463) | 0/92 | 48/92 (52.2%) | 47/92 (51.1%) | -1.1 pp | 85.9% | 6.07 → 5.59 | 1.09× | 0.99× | 14.06 → 14.08 | 33.4 → 32.3 (max 40.9/39.3) |
| InfiniteBench Code.Debug | 394 | 107570 (74,534–130,000) | 136/394 | 109/394 (27.7%) | 105/394 (26.6%) | -1.0 pp | 75.9% | 8.75 → 6.89 | 1.27× | 1.24× | 15.33 → 15.34 | 37.0 → 35.7 (max 41.2/39.6) |
| LongBench v2 | 503 | 84932 (10,087–130,000) | 186/503 | 135/503 (26.8%) | 131/503 (26.0%) | -0.8 pp | 81.9% | 6.70 → 5.31 | 1.26× | 1.14× | 13.76 → 13.77 | 34.3 → 31.9 (max 41.2/39.6) |

`dense/BB mean ratio` is ratio of mean paired-prefill times; above 1 means BB is faster. The paired median ratio shows the typical per-prompt change. Decode is a separate 16-step graph-decoding measurement. Peak allocation is CUDA max-allocated memory over prefill plus the short decode, not reserved memory or total device utilization. Mean actual tokens are counted after tokenization and middle truncation; the bucket labels are dataset categories, not a claim that every prompt has that exact length.

### Paired accuracy changes versus dense

| Test | BB correct when dense wrong | BB wrong when dense correct | exact McNemar p |
|---|---:|---:|---:|
| LongCodeQA 32K bucket | 1 | 1 | 1.000 |
| LongCodeQA 64K bucket | 2 | 2 | 1.000 |
| LongCodeQA 128K bucket | 3 | 4 | 1.000 |
| InfiniteBench Code.Debug | 25 | 29 | 0.683 |
| LongBench v2 | 25 | 29 | 0.683 |

The paired tests show no clear accuracy separation in these samples (all p ≥ 0.68); this is not an equivalence test, and it does not erase the observed per-item differences.

## Cache eviction comparison

The `bb_evict64k` arm used `evict_keep=65,536` extra selected middle positions per KV head, plus a 64-token sink and 16,384-token recent window (up to about 81,984 cached positions/head before clipping). It is a 65K extra-token budget, not a 65K total-cache cap. Eviction occurs after the prefill logits used for QA are produced, so identical dense-sparse answers here do not prove that long generated answers retain quality.

| Test | n | BB v3 accuracy | BB + eviction accuracy | eviction − v3 (pp) | BB v3 decode ms/token | BB + eviction decode ms/token | v3 peak GB mean/max | eviction peak GB mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LongCodeQA 32K bucket | 113 | 76/113 (67.3%) | 76/113 (67.3%) | +0.0 | 10.41 | 10.53 | 20.9/22.9 | 22.9/26.2 |
| LongCodeQA 64K bucket | 76 | 48/76 (63.2%) | 48/76 (63.2%) | +0.0 | 12.08 | 12.20 | 25.7/28.4 | 30.8/35.0 |
| LongCodeQA 128K bucket | 92 | 47/92 (51.1%) | 47/92 (51.1%) | +0.0 | 14.08 | 13.71 | 32.3/39.3 | 38.4/39.4 |
| InfiniteBench Code.Debug | 394 | 105/394 (26.6%) | 105/394 (26.6%) | +0.0 | 15.34 | 14.11 | 35.7/39.6 | 39.3/39.4 |
| LongBench v2 | 503 | 131/503 (26.0%) | 131/503 (26.0%) | +0.0 | 13.77 | 12.74 | 31.9/39.6 | 34.1/39.4 |

## Readout

- `bb_v3` mean prefill is 1.27× faster on Code.Debug and 1.26× on LongBench v2; it is slower on LongCodeQA 32K/64K (0.67×/0.80×). The 128K bucket mean ratio is 1.09×, but the paired median ratio is 0.99×, so a typical item is near parity.
- Exact multiple-choice QA is close to dense but slightly lower on the larger LongCodeQA bucket, Code.Debug, and LongBench v2. Report the paired counts and denominators, not ‘same accuracy’.
- `bb_v3` and the 65K-extra eviction arm should have identical first-token probabilities because eviction is downstream of prefill. Any decode/memory benefit is a separate outcome; the measured peak is dominated by the full prefill/cache path and should not be described as 64K total memory.
- These are R1 128K-class evaluations. The 1M-bucket work uses Nemotron-UltraLong-8B because R1's configured maximum is 131,072 tokens; do not combine the models in one accuracy column.

## Protocol and limitations

- The app warmed each arm at 20,000 and 20,001 tokens, then ran each test prompt once in each arm. It did not warm every exact prompt length and then retime that same prompt; lengths that trigger first-use specialization may retain a cold-start effect.
- Code.Debug and LongBench v2 prompts beyond the 130,000-token cap were middle-truncated. Their model-visible lengths are recorded in raw JSONL. LongCodeQA rows were not truncated in these 32K–128K runs.
- `resolved_revision` for the R1 weights was absent from the Modal run metadata (`requested_revision=main`). The paired arms used the same local checkpoint within each run, but the exact upstream weight commit is not recoverable from these result files; pin it for a strict rerun.
- The next-token A–D scorer is LongCodeQA/LongBench comprehension QA. The 16 decoded tokens are only a speed probe; they are not reviewed code, a valid patch, or test-suite success.

## LongCodeQA 1M bucket: tokenizer audit and follow-up

CPU token audit: 50/50 prompts; actual tokens 542,434–878,039 (mean 762,948); no truncation. Tokenizer is pinned to `fa3a10b9c6ad02f42c81e1d68dcf6967afb230ef`.

Same-H100 BF16 no-KV-cache result (9/50 official examples covered):
| n | actual tokens (mean, range) | Dense accuracy | BB v3 accuracy | Δ | BB gain/loss vs dense | McNemar p | Dense/BB prefill (s) | mean ratio | paired median ratio | mean peak allocated GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 754890 (542,434–878,039) | 1/9 (11.1%) | 3/9 (33.3%) | +22.2 pp | 3/1 | 0.625 | 509.90 → 47.94 | 10.64× | 11.83× | 55.6 → 49.5 |

Measured prompt range 542,434–878,039; truncation 0/9; audit IDs, gold answers, token counts, and hashes match: `True`. Warm/timed predictions equal: `True`. This is a full-prompt single forward with the final query available to routing; it is not a prefix-invariant causal-cache or decode result.
Additional matched H100 batch(es) are still active: [{'tag': 'lcqa1m_chunk_06_14_20261101', 'status': 'running', 'range': {'start': 6, 'end_exclusive': 15, 'count': 9}, 'warm_rows_saved': 18, 'timed_rows_saved': 0}]. Saved warm/timed arm rows across all batches: 54.

LongCodeBench separates LongCodeQA comprehension from LongSWE-Bench patch repair. Its published 1M bucket has 50 LongCodeQA prompts and a separate 100-item LongSWE-Bench repair set. A repair claim requires generated patches followed by the official Docker-based test harness, not a multiple-choice score. A matched LongSWE run is not included here: this workstation has no Docker CLI/daemon, and the official harness starts Docker containers. Do not treat the R1 Code.Debug QA run as patch repair. LongSWE remains the next benchmark workstream after the current matched LongCodeQA run. See the [paper](https://arxiv.org/abs/2505.07897) and [official benchmark code](https://github.com/Zteefano/long-code-bench).

## Earlier 1M synthetic RULER results (separate model/task)

The six-task synthetic NIAH comparison used Nemotron-UltraLong-8B, three prompts/task (~971K–980K actual tokens), depth 0.5, seed 42, BF16, and cached greedy output up to 64 tokens. Sparse used H100 with routed attention; dense used B200 with a CPU-offloaded BF16 KV cache. Prompt hashes match, but timing is not a hardware-matched comparison. The multi-query scorer checks answer-string presence, not answer-to-key pairing. Verification: `pass`.

| Task | n | Dense correct | BB correct |
|---|---:|---:|---:|
| niah_single_2 | 3 | 3/3 | 3/3 |
| niah_single_3 | 3 | 3/3 | 3/3 |
| niah_multikey_1 | 3 | 2/3 | 1/3 |
| niah_multikey_3 | 3 | 2/3 | 1/3 |
| niah_multivalue | 3 | 0/3 | 0/3 |
| niah_multiquery | 3 | 0/3 | 1/3 |

See the [historical report](../nemotron_hard_1m_20261029/results-report.md) for prompt hashes, prefill, and memory. These synthetic retrieval tasks are not code QA or LongSWE-Bench.

## Exp21 1M follow-up facts and efficiency

Separate 25-prompt held-out fact follow-up, four facts/prompt. The original report has 16K eviction at 62/100 facts and 20/25 needles; 64K at 85/100 and 19/25; 128K at 86/100 and 21/25; optimal 4-bit full-cache at 92/100 and 21/25; uniform 4-bit at 91/100 and 19/25. The 1M prefill medians were measured (63 s eviction; 72–78 s compressed); the 1M decode values were estimated from per-layer kernel benchmarks, not measured end-to-end. Do not merge these with LongCodeQA accuracy.

## Raw artifacts and verification

- Paired JSONL, metadata, 1M token audit, and Modal logs are under `raw/` and `logs/`. The matched 1M forward JSONs and separate slide CSV are under `longcodeqa_1m_prefill/` and `longcodeqa_1m_slide.csv`.
- Machine checks, warnings, and source hashes are in `verification.json`; aggregate metrics are in `summary.json` and `slide_summary.csv`.
- Report generated from repository commit `c3914ced3cca8b3fc3b3c6974264bd73912c1c8f`. The benchmark runs retain their own source commits and SHA-256 hashes; those hashes are checked below.
- No code was pushed. The Modal jobs used profile `maker041006`, environment `main`.

### Source SHA-256

| File | SHA-256 |
|---|---|
| `modal_rd/model_config.py` | `ffd81b0183a6471b8ce33e6ff49a3d7798ac2bb4e01863fce6e8881371e26684` |
| `modal_rd/niah.py` | `42ca15e9fbf360254be3245a239e7482682e09c8f62c6090d8a63d2539839917` |
| `modal_rd/attention.py` | `f8840746ba44418b082a9c1ff4bf0affc8cb8ef67f9e2b61c6e4bb4d03fceea7` |
| `modal_rd/graph_decode.py` | `031979471b7c5f7c282898d34b4b0ff0d9d11e653ddceb86ba57d8ad4cf415ae` |
| `modal_rd/inplace_residual.py` | `5c642002a36699d08bafbe97ecdbe5f9b25ba79ea31de43c422bae1ef5139718` |
| `modal_rd/mlp_chunk.py` | `0b8166731cecb9fbc43da702bcad26af9b05012e773845ab1ac31d92d1240728` |
| `modal_rd/cpu_embed.py` | `708e6b59c3f2a28d99de2acede5deea9a85fc62d1df7b3c941ada7fab2cb681d` |
| `kernels/bigger_bird_flash.py` | `713c725888d42c1608b919cb66c7e376bd3fb52b7a79cc314e593344e522226d` |
| `experiments/exp_20_sota_long_context/run.py` | `8d63696a7d81a25787b30e6d1576fdf80500bdc4dd5ac2b7bd372306da636e4d` |
| `experiments/exp_20_sota_long_context/tasks.py` | `4460e25a219e79d9167597026cdb07dec429d7f4beacc79ee1580b27c0b39ef6` |
| `modal_rd/app.py` | `ec52e946b51d53930c5d1599d3c6674b50920117fa5e14cf336f9dee01c0dd68` |
| `modal_rd/final_code_manifest.py` | `2a6b7099c1299957057778092a93af0e6a4fcb8fe0307002825569e69d2716bf` |
| `modal_rd/longcodeqa_1m_matched_prefill.py` | `93eb0d76acf7b771c13dcf50d2c78346583dd254cd7f0bf790ce5221b8f474a3` |
| `modal_rd/final_benchmark_report_20261101.py` | `075785cd1449d47b839f5780b12a65c8bfa99a59d2a2484661a44f96ae35e7b6` |
