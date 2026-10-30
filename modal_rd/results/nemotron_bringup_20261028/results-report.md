# Nemotron bring-up — 2026-10-28

Six-prompt pilot, synthetic NIAH at depth 0.5 and seed 42. Native Nemotron chat, BF16, greedy max64, full KV cache; all six prompts warmed then timed. Same H100, prompt hashes and source hashes across arms. Exact full-number scoring independently checked.

| Arm | Exact | Prefill s | Task s | Peak allocated GB | Actual token range |
|---|---:|---:|---:|---:|---|
| dense_rd | 6/6 | 16.724 | 17.128 | 38.15 | 127253–127768 |
| vs | 6/6 | 7.746 | 8.161 | 38.15 | 127253–127768 |

Prefill speedup: 2.159×. Native-HF versus dense-wrapper last-logit max error: 0.0. Greedy output IDs matched.

The original max10 attempt was stopped after its first dense warm prediction was truncated at `760` for gold `7606921`. It is preserved and excluded from the comparison. Both arms in the replacement use max64.

This pilot establishes bring-up only. It does not establish 30-prompt accuracy, harder-task quality, 1M feasibility, memory savings, or asymptotic scaling. No eviction or budget retuning was used.
