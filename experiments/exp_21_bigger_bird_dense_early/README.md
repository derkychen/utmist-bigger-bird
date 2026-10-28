# Exp 21: exp 19 with dense early layers (diagnostic)

Exp 19 (Bigger Bird Flash) unchanged in layers 3–31; layers 0–2 run full causal SDPA (the same operation as
exp 0). Defaults: exp 19's registry config plus `dense_layers: [0, 1, 2]`.

**This is a diagnostic, not a sparse-only method.** The dense layers cost O(n²) at any length, so this does not
meet the project's sparse-only goal. It shows where exp 19 loses the needle.

## Why

Exp 19 failed multikey NIAH: 30% exact, against 90% dense. Most wrong answers were invented numbers
(`1234567`, `9876543`).

The following were ruled out:
- the essay haystack: `niah_single_2` gets 100%;
- MMR: diversity 0 gives 27%;
- needle position: the needle is always at depth 0.5.

`scripts/oracle_exp19_needle.py` forces the 128 tokens around the gold needle into every head of chosen layers,
keeping the route budget the same. It is answer-aware and only a diagnostic.

| layers with the forced needle | exact |
|---|---:|
| none | 30% |
| all 32 | 93% |
| 24–31 | 30% |
| 12–23 | 27% |
| 6–11 | 33% |
| 0–11 | 90% |
| 0–5 | 80% |
| **0–2** | **83%** |

So exp 19 loses the needle in the early layers. Late layers do not need it.

## Results so far

All runs below share these conditions:
- RTX 5070 Ti laptop with 8-bit weight quantization (bitsandbytes LLM.int8);
- 30 examples per run, depth 0.5, exact full-number scoring;
- no answer information used.

Absolute numbers are not comparable with BF16 runs. Within a row, all arms use the same quantized weights.

| task | `--seq` (≈ tokens) | exp 19 | dense 0 | dense 1–2 | **dense 0–2 (exp 21)** | dense 0–11 | exp 0 (all dense) |
|---|---|---:|---:|---:|---:|---:|---:|
| niah_multikey_1 | 32768 (≈6.6K) | 30% | 30% | 93% | **100%** | 97% | 90% |
| niah_multikey_1 | 65536 (≈13.3K) | 20% | | | **90%** | | OOM |
| niah_single_1 | 32768 (≈8.2K) | | | | **100%** | | |

The dense-layer runs used `scripts/oracle_exp19_needle.py --oracle none --dense ...`. That script uses the same
model code as this class, except that its decode steps stay sparse. Without a KV cache, every generated token is a
full prefill, so this does not change any result.

## Open question: does it hold at ~128K tokens in BF16?

```bash
# Trillium (positional args: EXPS TASKS MAX_EXAMPLES)
sbatch experiments/exp_21_bigger_bird_dense_early/run_128k_trillium.sbatch
# one command, anywhere with an 80 GB GPU
python -m eval.ruler_llama.run_generative --task niah_multikey_1 --exp 21 --seq 614400 --max-examples 30
```

Results go to `benchmarks/exp_21_BiggerBirdDenseEarly_generative_ruler_<task>_seq<seq>/`. On Modal, add
`--results-root /vol/results/exp21`, because the container's repo copy is discarded after the run.

RULER `--seq` counts characters, and characters per token differ by task:
- multikey_1: 614400 ≈ 124K tokens (extrapolated from 4.94 chars/token; not yet tokenized at this length);
- single_1: 491520 ≈ 122.5K tokens (measured, 4.01 chars/token).

Prompts over 131072 tokens raise an error. Before this change they were silently right-truncated, which removed
the question.

## Caveats

- 30 examples per cell, and only lengths up to ~13K tokens. Nothing has been checked at 128K.
- Speed and memory were not measured. Three dense layers add 3/32 of dense attention cost.
- Team round 3 (`scripts/run_round3_probe.py`) ran dense/oracle layer hybrids at 127K. It used 8-layer groups,
  teacher-forced scoring and 3 prompts, and it treated the hybrids as diagnostics. Those results are not in the
  repo.
