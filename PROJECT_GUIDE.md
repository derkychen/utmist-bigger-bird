# Bigger Bird: Project Guide

A walkthrough of the repo layout, how to run things, and how to read `dashboard.html`.
Updated for `main` at `653fcb6` (PR #23, Oct 2026).

## What's new since `828a1a4`

- **Exp 19 is now a real experiment:** `experiments/exp_19_bigger_bird_flash/` (Bigger Bird v2 + a Triton sparse FlashAttention kernel). The old Kaggle IMDb run that used to show as "exp 19" is now **exp 24** on the dashboard.
- **Exp 20** (`experiments/exp_20_sota_long_context/`): Bigger Bird v3 vs dense on real benchmarks (LongBench v2, InfiniteBench Code.Debug, LongCodeQA). Only a pilot has been run so far.
- **`modal_rd/`**: a new research harness that runs on Modal H100s. This is where **Bigger Bird v3**, the current best method, was tuned. It also contains the memory work (KV eviction, INT8 KV cache, chunked MLP) and the presentation deck.
- **Exp 13 v2**: an input-adaptive budget plus a Triton kernel. It reaches **97% at RULER "128K"** with `tokens_per_key=16`.
- **Tokenizer fix:** on transformers 5.x, `AutoTokenizer` loaded the wrong tokenizer for R1-Llama (it dropped spaces, and dense fell to 20% at 4K). All R1 entry points now use `patches/llama/tokenizer.py`.
- **RULER runner changes:** `run_generative.py` now stops generating after the first complete number (`--no-early-stop` turns this off). Latency from older runs is **not comparable** with new runs.
- **New Slurm scripts** for Trillium (full H100) and Narval (A100 MIG), plus many new tests in `tests/`.

## What the project is

It tests **sparse attention** methods: each query looks at only a small chosen set of keys instead of all of them, which is cheaper than dense O(n²) attention. There are three generations of work:

1. **Older work: BART-base / BigBird on IMDb, LRA and RULER.** The model is fine-tuned with a classification head, using `run_experiment.py`, `eval/ruler` and `eval/lra`. [README.md](README.md) describes this part.
2. **R1-Distill-Llama-8B, zero-shot (exps 0–19).** No training. Each sparse method is patched into Llama's attention and tested on RULER needle-in-a-haystack (NIAH), where a "needle" fact is hidden in a long text and the model has to find it. [AGENTS.md](AGENTS.md) describes this part, and it's what `dashboard.html` is built around.
3. **Bigger Bird v3 (`modal_rd/`, exp 20).** The tuned final method: sink tokens + a large exact local window + routed "vertical" keys + an exact tail. It's validated on NIAH variants and on real long-context benchmarks, with its own results files and figures (**not** in the dashboard).

## Layout

| Path | What's in it |
|---|---|
| [experiments/](experiments/) `exp_N_*/` | One folder per method. `model.py` / `run.py` are the BART version; `model_llama.py` / `run_llama.py` are the Llama version. Exp 0 is dense, the only dense one allowed. |
| [experiments/exp_19_bigger_bird_flash/](experiments/exp_19_bigger_bird_flash/) | Bigger Bird v2. Read its [README](experiments/exp_19_bigger_bird_flash/README.md) for the design and caveats. |
| [experiments/exp_20_sota_long_context/](experiments/exp_20_sota_long_context/) | Real-benchmark eval of v3. See its [README](experiments/exp_20_sota_long_context/README.md) and [RESULTS.md](experiments/exp_20_sota_long_context/RESULTS.md). |
| [modal_rd/](modal_rd/) | The v3 R&D harness. `attention.py` (`RDAttention`), `eval_niah.py`, memory savers (`kv_evict`, `kv_int8.py`, `mlp_chunk.py`, `cpu_embed.py`), `results/*.json`, [RESULTS.md](modal_rd/RESULTS.md), `figures/`, [MEMORY_DEFAULTS.md](modal_rd/MEMORY_DEFAULTS.md), `deck/index.html` (the slides). |
| [sparse_attn_utils.py](sparse_attn_utils.py) | Shared sparse building blocks such as `causal_sparse_attention` (now ~22x less memory, with an opt-in `use_triton=True`). |
| [kernels/](kernels/) | Triton/PyTorch kernels. New: `routed_window_flash.py` (exp 13), `bigger_bird_flash.py` / `bigger_bird_routing.py` (exp 19 and v3), `kv_int8_triton.py`. |
| [patches/llama/](patches/llama/) | Code that swaps the custom attention into Llama, and `tokenizer.py` with the tokenizer fix. |
| [eval/ruler_llama/](eval/ruler_llama/) | **The main RULER eval.** `run_generative.py` asks the model to answer and checks whether it recalled the needle. |
| [eval/lra_llama/](eval/lra_llama/) | LRA text/listops evals for Llama. |
| `eval/{ruler,lra,imdb,nolima,longbench-v2}` | Older BART-side evals and other benchmarks. |
| [configs/](configs/), [config_schema/](config_schema/) | YAML configs and their schemas. |
| [benchmarks/](benchmarks/) | Every result file that feeds the dashboard. `exp_<N>_<Name>_generative_ruler_niah_seq*/results_<timestamp>.json` holds the R1 runs; `bigger_bird_flash/` holds the exp 19 smoke tests and microbenchmarks. |
| [benchmark_artifacts/](benchmark_artifacts/) | Matched dense-vs-exp-19 RULER runs (same prompts, cold and warm). |
| [scripts/](scripts/) | `build_dashboard.py`, Slurm launchers (`ruler_generative_trillium.sbatch`, `run_ruler_exp19.sbatch`, `run_bigger_bird_smoke.sbatch`, …) and round-2/3 probes. |
| [research-plan.md](research-plan.md), [research-plan-round2.md](research-plan-round2.md) | Research plans for the newer rounds. |
| [viz/](viz/), [report.html](report.html) | Plotting scripts and a separate written report. |
| [tests/](tests/), [kernels/tests/](kernels/tests/) | Unit tests (kernels, cache equivalence, KV eviction/INT8, GQA, causal sparse attention). |

### Experiment numbers

| # | Method | One-line idea |
|---|---|---|
| 0 | Dense | Full attention baseline (SDPA, causal) |
| 1 | DeepSeek top-k | Low-rank scores pick top-k keys (tuned: k=512, low_rank_dim=128) |
| 2 | Lightning Hybrid | Local window + linear attention |
| 3 | Dynamic Globals | Local window + learned-gate global tokens |
| 4 | PBS | Block-level top-k |
| 5 | BiggerBird v1 | Local MMR + global gate + random teleports |
| 6 | DeepSeek + PBS | Block routing, then top-k within blocks |
| 7 | Layer Adaptive | Different k per layer depth |
| 8 | Token Drop | Drop 30% of tokens after layer 3 |
| 9 | Attention Speculation | Local window + anchor tokens |
| 10 | GQA Sparse | Top-k with low-rank proxy over KV groups |
| 11 | NSA | Compressed + selected + sliding-window branches |
| 12 | S2-HHST | Strided block-sparse |
| 13 | Dynamic Context (v2) | Routed budget scales with length: k = clamp(N / tokens_per_key, 512, 4096), plus sinks and a 256-token local window; optional Triton kernel |
| 14 | Token Drop + DeepSeek | Token drop, then top-k |
| 15 | BiggerBird (proper) | Block-mean routing + globals |
| 16 | Free NSA | Parameter-free NSA variant |
| 17 | Coarse-to-fine | Block selection, then token top-k inside blocks |
| 18 | Confidence-gated | Local window + linear branch + routed global correction |
| 19 | **Bigger Bird Flash (v2)** | Content globals + selective wide local window (MMR) + content-biased long links + 128-token exact recent window, in one Triton sparse FlashAttention kernel |
| 20 | **Real-benchmark eval of v3** | Not a new method: it compares `dense`, `bb_v3` and `bb_v3_evict` on LongBench v2, Code.Debug and LongCodeQA |
| (v3) | **Bigger Bird v3** (`modal_rd`) | `{"sink":64,"window":16384,"vertical":512,"exact_tail":256}`: 64 sinks + 16K exact window + 512 routed keys + exact attention for the last 256 queries |

> **Dashboard-only numbers.** `CUSTOM_EXP_MAP` in [scripts/build_dashboard.py](scripts/build_dashboard.py) also gives numbers to old BigBird/Kaggle IMDb folders: 15–18 (`bigbird`, `bigbird_8000`, `biggerbird_globals_ver1`, `biggerbird_topk_mmr`), 20–23 (`biggerbird_topkmmr_globals_*`) and 24 (`biggerbird_topk_mmr_ver2`). **15–18 and 20 clash** with the real experiments of the same number. Those rows are `bart-base` / `imdb`, so filter by Model and Track to tell them apart.

## How to run

The R1 evals need a GPU with about 40 GB. There are three places the team runs things:

### Narval / original cluster (H100 or A100 MIG)

```bash
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# One RULER needle run: experiment 1, 4K, needle halfway in, 30 examples
python -m eval.ruler_llama.run_generative --task niah --exp 1 --seq 4096 --depth 0.5 --eval-samples 128 --max-examples 30

# Override method settings
python -m eval.ruler_llama.run_generative --exp 13 --seq 131072 --attn-kwargs '{"tokens_per_key": 16, "use_triton": true}'

# Exp 19 RULER at 4K and 16K, then rebuild the dashboard
sbatch scripts/run_ruler_exp19.sbatch
# Exp 19 smoke tests: numerical checks, cache equivalence, timings, paired retrieval
sbatch scripts/run_bigger_bird_smoke.sbatch
```

Slurm: account `def-guerzhoy`; partitions `gpubase_bygpu_b1` (3h), `b5` (7h), `b2` (12h).

### Trillium (full H100)

```bash
cd $SCRATCH/bigger-bird-r1
sbatch scripts/ruler_generative_trillium.sbatch EXP SEQS [MAX_EXAMPLES] [ATTN_KWARGS_JSON]
sbatch --partition=debug --time=1:00:00 scripts/ruler_generative_trillium.sbatch 13 4096 5
```

Trillium quirks: `sbatch` forces `--export=NONE` (pass settings as script arguments, not env vars), `$HOME` is read-only on compute nodes (set `TRITON_CACHE_DIR`), and `debug` allows one job per user.

### Modal (H100, for `modal_rd` and exp 20)

```bash
modal profile activate <workspace>
modal run modal_rd/app.py::download                       # weights to volume bb-vol (once)
modal run modal_rd/app.py --module modal_rd.selfcheck     # check kernels before long runs
modal run [--detach] modal_rd/app.py --module modal_rd.eval_niah \
    --args "--arm vs --attn '{\"sink\":64,\"window\":16384,\"vertical\":512,\"exact_tail\":256}' --bytes 512000 --n 6 --tag x"
python modal_rd/report.py                                 # pull results -> modal_rd/RESULTS.md + figures/

# Exp 20
bash experiments/exp_20_sota_long_context/run_h100.sh h100_130k
python -m experiments.exp_20_sota_long_context.report --tag <tag> --out experiments/exp_20_sota_long_context/RESULTS.md
```

GPU functions need a payment method on the Modal workspace. Don't write logs inside the repo tree while `modal run` is building; it aborts the build.

### Notes

- **Start small:** run `--exp 0 --seq 4096 --max-examples 5` first to check the setup.
- **Where results go:** RULER runs write `benchmarks/exp_<N>_<Name>_generative_ruler_<task>_seq<seq>/results_<timestamp>.json`. The dashboard only shows a new run after `python scripts/build_dashboard.py`.
- **Older BART pipeline:** `python run_experiment.py --exp 1 --size small` (see [COMMANDS.md](COMMANDS.md)).

## Reading dashboard.html

> **The committed dashboard is stale.** It was built on 2026-09-27. The exp 13 v2 runs (October) are in `benchmarks/` but not in the dashboard. Run `python scripts/build_dashboard.py` before reading it. Bigger Bird v3 and exp 20 never appear in the dashboard; read `modal_rd/RESULTS.md` and `experiments/exp_20_sota_long_context/RESULTS.md` for those.

Open it in a browser; it's self-contained. Use the filters at the top (**Track, Model, Task, Needle depth**) first. For the main results, pick Track `ruler`, Model `r1-llama-8b`, Task `niah`, depth `0.5`.

**Tracks:** `ruler`, `lra`, `imdb`, `nolima`, and the new **`smoke`**. `smoke` holds the exp 19 paired tests: task `exact_number` (dense vs exp 19 `last_query`), `exact_number_causal_chunk` (the causal ablation) and `attention_microbench` (attention-only timings).

### Tabs

| Tab | What it shows |
|---|---|
| **Overview** | Three charts against sequence length: accuracy (the headline), latency per example, and peak memory. The summary table gives mean and best accuracy per experiment. A good sparse method keeps accuracy near exp 0's 100% while its memory line stays flat as length grows. |
| **Coverage** | A grid of experiments by sequence length. Each cell is a real result: accuracy, examples completed out of expected, number of source files, and status. An empty cell means that run was never done; it's not a 0. |
| **Efficiency** | The same data by metric (latency, memory, eval time, softmax comparisons). The "Preferred observations" table lists the one representative row per cell. |
| **All source runs** | Every parsed file, searchable, with a link to its JSON. Use it to check where a number came from. |
| **Complexity** | A separate microbenchmark that times only the attention layer, not full generation. OOM means the GPU ran out of memory, and those points are kept. |
| **Data quality & sparse audit** | Parse errors, files the script didn't recognize, and a check that only exp 0 calls dense attention. The `benchmarks/bigger_bird_flash/invalid_tokenizer/` folder is skipped on purpose. |

### Status values

| Status | Meaning |
|---|---|
| `ok` / `pass` | The run finished. |
| `partial` / `incomplete` | Fewer examples finished than expected. |
| `oom` | Ran out of GPU memory. |
| `fail` / `collapse` / `skipped` | Did not produce a usable result. |

### Validity values

| Validity | Meaning |
|---|---|
| `dense_baseline` | Exp 0. |
| `sparse_only` | A trustworthy sparse run. |
| `pre_sparse_fix` | R1 runs from before 2026-08-13. They may have quietly fallen back to dense attention, so the charts leave them out. |
| `not_r1` | BART or BigBird results. |

### Things to watch for

1. **RULER "seq" is characters, not tokens.** `--seq` counts characters (~3.6 chars per token), so RULER "128K" is only about **36K tokens**. The `modal_rd` NIAH results use real token counts (~127K tokens), so they are not the same test as the dashboard's "128K" column.
2. **Each chart cell shows the best run.** When several runs share an experiment, length and depth, `preferred_rows()` in [scripts/build_dashboard.py](scripts/build_dashboard.py) picks the one with the highest accuracy (after preferring complete runs). For exp 13 at 128K that means 97% (`tokens_per_key=16`) wins over 10% (the default settings). Check All source runs for the spread and the settings.
3. **Old vs new latency.** Runs since early stop was added generate fewer tokens, so they look faster than older runs even for the same method.
4. **Tokenizer.** R1 runs made on transformers 5.x before the tokenizer fix are wrong. The exp 19 README says other experiments still need a tokenizer audit before their fresh results are compared with older ones.
5. **Small sample sizes.** Most cells are 30 examples, so one example is 3.3 points. Some 128K runs used only 5–15 examples, and the smoke track uses 1 example per row.
6. **Mixed numbering and tasks.** See the dashboard-only numbers note above. IMDb and LRA accuracies are classification on a different model, so don't compare them with RULER retrieval.
7. **Build log is missing.** AGENTS.md says the build writes `benchmarks/dashboard_build.log`, but that file isn't in the repo. Rerunning `build_dashboard.py` creates it.

## Main results

### RULER NIAH (dashboard), depth 0.5, 30 examples unless noted

Remember: "128K" here is about 36K tokens.

| Method | 4K | 8K | 16K | 32K | 64K | 128K | Peak memory |
|---|---|---|---|---|---|---|---|
| Dense (exp 0) | 100% | 100% | 100% | 100% | 100% | 100% | 16–20 GB |
| TopK k=512 (exp 1) | 100% | 100% | 100% | 86.7% | 90.0% | 26.7% | 21–23 GB |
| TopK k=2048 (exp 1) | — | — | — | — | — | 80%* | 21–23 GB |
| CoarseToFine (exp 17) | 100% | 100% | 100% | 87% | 93% | 0% (60%* with fine_k=2048) | 21–33 GB |
| **DynamicContext v2 (exp 13)** | 100% | 100% | 100% | 93% | 90% | 10% default / **97%** with `tokens_per_key=16` | 16–20 GB |
| **Bigger Bird Flash (exp 19)** | 100% | 100% | 100% | 73% | 53% | 0% | 16–20 GB |
| Most others (3, 4, 7, 8, 11, 12, …) | ~0% | ~0% | ~0% | 0% | 0% | 0% | — |

\* 5–15 examples only. The exp 13 v2 and exp 19 numbers come straight from the result files in `benchmarks/` (exp 13 v2 isn't in the committed dashboard yet).

### Bigger Bird v3 on NIAH (`modal_rd/RESULTS.md`, real token counts)

| Context | Dense | v3 (window 16K) | Prefill speedup |
|---|---|---|---|
| ~65K tokens | 30/30 | 30/30 | 1.15x |
| ~128K tokens, depth 0.5 | 28/30 | 30/30 | 1.9x |
| ~128K tokens, depth 0.1 | 28/30 | 30/30 | 1.9x |

- Prefill time grows roughly linearly for v3 (exponent ~1.0) vs ~1.65 for dense.
- Harder NIAH variants (multi-key, multi-value, UUIDs) at ~128K are hard for **dense too**; v3 is worse on some (e.g. `niah_single_2`: 11/24 vs dense 17/24).
- Ablation: 64 routed keys gets 0/6; 256–512 gets 6/6. Larger budgets hurt.

### Exp 20 pilot (A100, 481 paired examples, no-CoT letter scoring)

- **Accuracy:** dense 47.4% vs v3 46.4% overall, not a significant difference (p = 0.53). LongCodeQA is identical at every length. LongBench v2 and Code.Debug are near chance even for dense.
- **Speed:** prefill is slower than dense below ~64K tokens, and 1.47x faster at ~117K.
- **Memory:** sparse prefill alone peaks the same as dense (the full KV cache dominates). With KV eviction, peak drops 38.1 → 26.5 GB at 96K+ and decode stays ~26 ms/token.
- **Not done:** 294 Code.Debug and 403 LongBench v2 items, planned as a full H100 rerun.

### Takeaways

- Dense gets 100% on RULER at every length.
- Content-based top-k routing (exp 1, 13, 17) holds up through 64K; at the longest length it needs a larger budget (exp 1 k=2048, exp 13 `tokens_per_key=16`).
- Exp 19 keeps 100% through 16K but drops off from 32K.
- Bigger Bird v3 matches or beats dense on single-needle NIAH at ~128K real tokens with ~1.9x faster prefill, and matches dense on the exp 20 pilot so far.
- Methods that pick keys with fixed patterns or untrained learned gates get about 0%.
