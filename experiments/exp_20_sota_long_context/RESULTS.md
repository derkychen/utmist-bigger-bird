# Exp 20 results: `a100_120k`

> **Pilot only — the full run is not done.** LongCodeQA is complete. Code.Debug and LongBench v2 are 100 of 394 and 100 of 503 items. The remaining items, and a full rerun at 130K tokens, are planned on an H100: see README.md, "What is left to run".

GPU NVIDIA A100-SXM4-40GB, torch 2.13.0+cu130, max prompt 120000 tokens (middle-truncated), MLP chunk 8192, 16 decode tokens, capped at codedebug=100,lbv2=100. Arms: `dense` = HF SDPA dense; `bb_v3` = {"sink": 64, "window": 16384, "vertical": 512, "exact_tail": 256}; `bb_v3_evict` = {"sink": 64, "window": 16384, "vertical": 512, "exact_tail": 256, "kv_evict": true, "evict_pool": 8, "evict_keep": 8192, "evict_window": 16384}

481 examples complete in every arm; OOM per arm: {'dense': 0, 'bb_v3': 0, 'bb_v3_evict': 0}.

### Accuracy by task

|  | n | dense acc % | bb_v3 acc % | bb_v3_evict acc % | Δ bb_v3 pp | Δ bb_v3_evict pp | agree bb_v3 % | agree bb_v3_evict % | McNemar p (bb_v3) | McNemar p (bb_v3_evict) |
|---|---|---|---|---|---|---|---|---|---|---|
| codedebug | 100 | 24.0 | 22.0 | 22.0 | -2.0 | -2.0 | 74.0 | 74.0 | 0.79 (6↑/8↓) | 0.79 (6↑/8↓) |
| lbv2 | 100 | 29.0 | 26.0 | 26.0 | -3.0 | -3.0 | 75.0 | 75.0 | 0.58 (5↑/8↓) | 0.58 (5↑/8↓) |
| lcqa128k | 92 | 54.3 | 54.3 | 54.3 | +0.0 | +0.0 | 88.0 | 88.0 | 1.00 (5↑/5↓) | 1.00 (5↑/5↓) |
| lcqa32k | 113 | 68.1 | 68.1 | 68.1 | +0.0 | +0.0 | 100.0 | 100.0 | 1.00 (0↑/0↓) | 1.00 (0↑/0↓) |
| lcqa64k | 76 | 63.2 | 63.2 | 63.2 | +0.0 | +0.0 | 94.7 | 94.7 | 1.00 (2↑/2↓) | 1.00 (2↑/2↓) |
| **All** | 481 | 47.4 | 46.4 | 46.4 | -1.0 | -1.0 | 86.3 | 86.3 | 0.53 (18↑/23↓) | 0.53 (18↑/23↓) |

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
| 64-96K | 125 | 42.4 | 41.6 | 41.6 | -0.8 | -0.8 | 92.0 | 92.0 | 1.00 (2↑/3↓) | 1.00 (2↑/3↓) |
| 96K+ | 128 | 28.9 | 25.8 | 25.8 | -3.1 | -3.1 | 59.4 | 59.4 | 0.60 (14↑/18↓) | 0.60 (14↑/18↓) |
| **All** | 481 | 47.4 | 46.4 | 46.4 | -1.0 | -1.0 | 86.3 | 86.3 | 0.53 (18↑/23↓) | 0.53 (18↑/23↓) |

### Answer-distribution fidelity vs dense

TV = total-variation distance between the arms' A-D distributions; Δp(gold) = arm minus dense probability on the gold letter.

| arm | mean TV | median TV | mean Δp(gold) | mean dense p(gold) |
|---|---|---|---|---|
| bb_v3 | 0.0892 | 0.0419 | -0.0059 | 0.416 |
| bb_v3_evict | 0.0892 | 0.0419 | -0.0059 | 0.416 |

### Speed and memory by prompt length

Prefill = one CUDA-timed forward over the prompt; peak = max allocated during prefill + decode; speedup = dense prefill / arm prefill (ratio of means).

| prompt length | n | mean tokens | dense prefill s | bb_v3 prefill s | bb_v3_evict prefill s | bb_v3 speedup | bb_v3_evict speedup | dense decode ms/tok | bb_v3 decode ms/tok | bb_v3_evict decode ms/tok | dense peak GB | bb_v3 peak GB | bb_v3_evict peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ≤16K (bb = dense) | 41 | 12110 | 1.08 | 1.08 | 1.08 | 1.00x | 1.00x | 26.8 | 25.7 | 25.8 | 18.8 (max 19.4) | 18.8 (max 19.4) | 18.8 (max 19.4) |
| 16-32K | 99 | 23918 | 2.52 | 3.98 | 3.93 | 0.63x | 0.64x | 27.1 | 25.9 | 25.7 | 20.9 (max 22.3) | 20.9 (max 22.3) | 20.8 (max 21.4) |
| 32-64K | 88 | 50827 | 7.25 | 8.26 | 8.28 | 0.88x | 0.88x | 33.2 | 33.1 | 25.7 | 25.7 (max 28.4) | 25.7 (max 28.4) | 22.4 (max 23.3) |
| 64-96K | 125 | 82549 | 15.22 | 13.35 | 13.37 | 1.14x | 1.14x | 42.2 | 42.2 | 25.8 | 31.7 (max 34.6) | 31.7 (max 34.6) | 24.4 (max 25.3) |
| 96K+ | 128 | 116782 | 27.11 | 18.40 | 18.42 | 1.47x | 1.47x | 51.9 | 51.8 | 25.7 | 38.1 (max 38.7) | 38.1 (max 38.7) | 26.5 (max 26.7) |
| **All** | 481 | 67784 | 13.11 | 10.79 | 10.79 | 1.22x | 1.21x | 38.7 | 38.4 | 25.7 | 29.0 (max 38.7) | 29.0 (max 38.7) | 23.4 (max 26.7) |

