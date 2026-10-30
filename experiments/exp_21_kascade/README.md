# Exp 21: Kascade-style cross-layer route reuse on Bigger Bird v3

Based on Kascade (arXiv 2512.16391): high-attention keys are stable across
neighbouring layers, so only a few "anchor" layers compute top-k routing and the
layers in between reuse it. Here the base method is Bigger Bird v3
(`modal_rd/attention.py`, arm `vs`), whose `routes()` picks per-head routed key
columns at every layer. `model.py` makes non-anchor layers reuse the nearest
earlier anchor's columns (same head index); everything else is unchanged v3.

## Local setup (RTX 3080, 10 GB, WSL)

R1-Distill-Llama-8B in 4-bit NF4 (bitsandbytes) with chunked MLP, last-token
logits and a CPU-offloaded KV cache (`common.py`). This fits ~16K-token prompts.
Numbers are therefore 4-bit and not directly comparable to the bf16 H100 tables.
Under WSL an over-full GPU spills into system RAM instead of raising OOM, so
timings near the 10 GB limit are unreliable.

```bash
python3 -m venv ~/venvs/bb
~/venvs/bb/bin/pip install "torch==2.13.0" "transformers==5.15.1" accelerate datasets omegaconf \
    huggingface_hub safetensors tokenizers numpy bitsandbytes matplotlib
export SCRATCH=$HOME/scratch   # weights at $SCRATCH/models/DeepSeek-R1-Distill-Llama-8B
```

## Phases

| Phase | Script | What |
|---|---|---|
| 0 | `probe_overlap.py` | Run v3 unchanged; save every layer's routed columns + needle span |
| 0/1 | `analyze_overlap.py` | Cross-layer overlap matrices, needle coverage, anchor choice by DP |
| 2a | `eval_reuse.py` | NIAH accuracy of anchor reuse vs v3 on held-out prompts; time in `routes()` |

```bash
python -m experiments.exp_21_kascade.probe_overlap --bytes 64000 --n 6 --depths 0.5,0.25 --tag p0_16k
python -m experiments.exp_21_kascade.analyze_overlap --tag p0_16k
python -m experiments.exp_21_kascade.eval_reuse --sets "all;0;0,11;0,7,13,23" --start 6 --n 6 --tag p2a_16k
```

Outputs go to `benchmarks/exp_21_kascade/` (locally) or `/vol/results/exp_21_kascade/` (Modal).

## Modal H100 (Nemotron-UltraLong-1M, ~128K tokens)

`common.py` switches to BF16, an on-GPU KV cache and volume output paths when `SCRATCH=/vol`
(set by `modal_rd/app.py`); `BB_QUANT=nf4|bf16` overrides. `BB_MODEL=nemotron` selects
`nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct` and its chat format via `modal_rd.model_config`.
That checkpoint is byte-identical (revision `fa3a10b`) to the `Llama-3.1-8B-UltraLong-1M-Instruct`
used for the local `ul_*` runs; symlink it under the Nemotron name to run locally with `BB_MODEL=nemotron`.

Exp 21 as tested on the H100 = v3 prefill unchanged + sparse decode with anchor-layer reuse, head
remap and KV-head pooling (`dpK:remap:pool:decodeonly`). Anchors and head maps come from the local
16K Phase 0 run `ul_p0_16k_tail0_pool` (uploaded to the volume) instead of a 128K Phase 0 on the
H100, to save credits; `dpK` = the K layers `analyze_overlap.py` picks (3: 0,9,19; 5: 0,4,9,14,19;
8: 0,1,2,4,8,11,14,19). `h100_plan.py` runs everything with one model load; `run_h100.sh`:

| Stage | What | Est. H100 time |
|---|---|---|
| `download` | Nemotron weights to `bb-vol` (CPU) | - |
| `upload` | local Phase 0 run to `bb-vol` | - |
| `smoke` | every arm on one ~16K prompt with a 2K window (plumbing only) | ~5 min |
| `plan` accuracy | 30 single-needle prompts, exp 21 dp5 (dense / v3 are in `modal_rd/results/nemotron_confirm30_20261028`); multikey_1 n=6: v3 vs dp3 / dp5 / dp8 | ~15 min |
| `plan` long | `eval_decode.py --mode long`, ~384-token answers, n=6: v3, v3+evict, exp 21 with decode window 16384 / 4096 / 1024 (+ routed 1024) | ~15 min |
| `plan` followup | `eval_decode.py --mode followup`, second question on the same cache, same arms | ~10 min |

v3 decodes exactly, so it is the reference for the decode-only arms. `eval_decode.py` reports decode
ms/token for the whole model and for the attention modules alone (CUDA events), since per-token
Python overhead can hide attention savings. `dw<N>` / `dv<N>` set the decode window / routed budget.

## Phase 0 results (~16K tokens, v3 `sink64+win4096+vert512+tail256`, prompts 0-5 x depths 0.5/0.25)

v3 answered 12/12. Overlap of routed columns between adjacent layers (chance 0.043):

| Matching | Adjacent-layer overlap |
|---|---:|
| Same head index | 0.25 |
| Best-matching head (oracle) | 0.43 |
| Pooled over heads | 0.63 |

Needle coverage (fraction of heads whose 512 columns contain the whole answer
number): layers 0-2 ~0.00-0.04, layer 3+ mostly 0.7-0.99, mean 0.70.

| Anchors (DP on same-head overlap) | Mean overlap | Needle coverage under reuse |
|---|---:|---:|
| 1: [0] | 0.08 | 0.00 |
| 2: [0, 11] | 0.24 | 0.50 |
| 4: [0, 7, 13, 23] | 0.33 | 0.66 |
| 8: [0, 1, 3, 5, 7, 11, 16, 23] | 0.45 | 0.64 |
| 32 (plain v3) | 1.00 | 0.70 |

Reading: most routed columns change between layers, but the needle columns are
stable, so coverage under reuse stays close to all-layer routing from 4 anchors.
Layers 0-2 never route the needle, which makes layer 0 a poor anchor.

## Phase 2a results (~16K tokens, held-out prompts 6-11, depth 0.5)

With v3's default `exact_tail=256` every arm scored 6/6, including the layer-0-only
control: the dense tail delivers the needle at 16K without routed columns, so that
setting cannot discriminate (partial log: `phase2a/p2a_16k_tail256_partial.json`).
With `exact_tail=0`, routed columns are the only long-range path:

| Anchors | Exact | Time in routes() / total per prompt |
|---|---:|---:|
| all 32 (plain v3, tail 0) | 6/6 | 0.13 s / 7.9 s |
| [0] (negative control) | 0/6 | 0.01 s / 7.7 s |
| [0, 11] | 6/6 | 0.01 s / 7.7 s |
| [0, 7, 13, 23] | 6/6 | 0.02 s / 7.6 s |
| [0, 1, 3, 5, 7, 11, 16, 23] | 6/6 | 0.03 s / 7.8 s |

Reuse from 2 anchors keeps retrieval at 16K even though predicted needle coverage
drops from 0.70 to 0.50: enough heads from layer 11 on carry the needle. With the
default tail, `routes()` took 0.38 s of 17.6 s (~2%). Small screen (6 prompts,
4-bit, 16K); needs longer contexts (bf16 on H100/Modal) to find where it breaks.

### Harder RULER tasks (~16K tokens, 80,000 bytes, 24 prompts, exact_tail 0, max 32 new tokens)

| Task | Dense | All 32 route | 2 anchors [0,11] | 4 anchors [0,7,13,23] |
|---|---:|---:|---:|---:|
| niah_multikey_1 (4 keys, retrieve 1) | 24/24 | 24/24 | 17/24 | 21/24 |
| niah_multivalue (4 values, one key) | 15/24 (0.82) | 10/24 (0.71) | 4/24 (0.47) | 1/24 (0.25) |
| niah_multiquery (4 keys queried) | 22/24 (0.98) | 22/24 (0.98) | 23/24 (0.98) | 20/24 (0.96) |

(fraction of gold values found in parentheses). Reuse holds when the needles are
*distinct* (multi-query names four different keys) but breaks when they are
*near-identical*:
- multi-key: all 7 wrong answers at 2 anchors (2 of 3 at 4 anchors) are a
  distractor needle's value. Needles are still routed; the right one is not
  distinguished from look-alikes.
- multi-value: reuse arms usually return only the first value and stop (0.25 found
  at 4 anchors = one of four), while dense/all-layer list all four.
The 4-vs-2-anchor ordering on multi-value is within noise for 24 prompts.
Possible fix: non-anchor layers re-rank the anchor's candidate set with their
own query (cheap) instead of copying it.

## Phase 2b: sparse decode with anchor reuse (`sparse_decode.py`, `eval_gsm8k.py`)

v3 decodes dense (every generated token reads the whole KV cache). Phase 2b keeps
v3 prefill and makes decode sparse: each layer attends to sink + recent window +
`vertical` routed keys per head. Anchor layers score the current query against
all cached keys and take the per-head top-k; other layers reuse the nearest
earlier anchor's indices. Routes are recomputed every step at the anchors, so
the routed set never overlaps the sliding window (no double-counted keys).

Checked on a tiny random Llama (CPU): when routed + sink + window cover the
whole cache, sparse decode matches dense to 1e-7; with a small budget it differs.

KV tokens read per layer per generated token: dense 2n, anchor n + 2S, reuser 2S
(S = sink + window + vertical). E.g. n = 2000, 32 layers, 2 anchors, S = 832:
57K vs 128K (2.2x); the ratio improves as n grows. Evaluated on GSM8K with R1
chain-of-thought (decode-dominated), window 512 so sparsity applies to short prompts:

```bash
python -m experiments.exp_21_kascade.eval_gsm8k --sets "dense;all;0,11;0,7,13,23" --n 20 --tag p2b_gsm8k
```

### GSM8K, first pass (20 problems, max 2048 new tokens, window 512, vertical 256)

| Arm | Correct | Hit length cap | Mean new tokens |
|---|---:|---:|---:|
| dense | 15/20 | 5 | 1408 |
| sparse decode, all 32 route | 15/20 | 5 | 1371 |
| sparse decode, 2 anchors [0,11], same head, unpooled | 14/20 | 7 | 1479 |

The [0,7,13,23] arm was stopped after 5 problems (superseded by the combined run).
The 2-anchor arm lost two problems only by hitting the length cap (longer reasoning,
as Kascade reports for R1-Distill) and gained two others. Read ratios logged by this
run assumed pooled picks and understate unpooled reads.

## Combined run: head remapping + KV-head pooling + Kascade anchors (`p2e_*`)

Kascade's three missing pieces together: non-anchor heads reuse the most similar anchor
head (map from a pooled Phase 0 run, `phase0/p0_16k_tail0_pool`, prompts 0-5); each KV
head's four query heads share one pick set (averaged softmax); anchors [0, 2, 8, 13, 14]
(Kascade's set for Llama-3.1-8B). Reference arm `all:pool` = every layer routes, pooled.
Arm syntax: `--sets "0,2,8,13,14:remap:pool"`.

Pooled Phase 0: plain pooled v3 12/12; remapping raises copied-pick overlap 0.25 -> 0.37
(in-sample); needle coverage 0.76 (all layers) vs 0.67 under reuse.

| Test (~16K, exact_tail 0, 24 prompts) | all:pool | combined | earlier same-head reuse |
|---|---:|---:|---:|
| niah_multikey_1 | 24/24 | **24/24** | 17/24 [0,11], 21/24 [0,7,13,23] |
| niah_multivalue | 10/24 (0.71) | **10/24 (0.75)** | 4/24, 1/24 |

| GSM8K (20 problems) | Correct | Hit cap | Mean new tokens | KV read ratio |
|---|---:|---:|---:|---:|
| dense | 15/20 | 5 | 1408 | 1.00 |
| all:pool | 14/20 | 6 | 1389 | 1.11 |
| combined | **15/20** | 4 | 1374 | 0.87 |

The combined version recovers both look-alike failures and keeps GSM8K accuracy at
dense level without the longer outputs. Read savings are small at GSM8K lengths
(contexts ~1-2K tokens; ratio improves with length). Since all three changes went in
together, the gain is not attributed to any single one. Small samples, 4-bit, 16K.

## NoLiMa, generative (`eval_nolima.py`, `phase3/`)

NoLiMa (Modarressi et al., ICML 2025): needle and question share no keywords
("{CHAR} lives next to the Kiasma museum" / "Which character has been to Helsinki?").
Data from `scripts/get_nolima_data.sh` into `$NOLIMA_DATA_DIR` (outside the repo).

Without thinking, R1-Distill-8B cannot make the latent hop even with dense attention at
~2K tokens (3/10), while the literal control set (`needle_set_ONLYDirect.json`) scores
10/10, so the run lets R1 reason (`--think`, max 1024 new tokens; scored on the text after
`</think>`, unfinished = wrong). ~8K-token prompts (31,000 chars), window 2048, vertical 512,
exact tail 0, KV cache on GPU; head map from `phase0/p0_8k_w2k_tail0_pool`. 30 questions
(16 one-hop, 14 two-hop), depths 0.25/0.5/0.75.

| Arm | Correct | Unfinished (hit 1024) | one-hop | two-hop |
|---|---:|---:|---:|---:|
| dense (prefill + decode) | 6/30 | 12 | 4/16 | 2/14 |
| all:pool:decode (every layer routes, sparse prefill + decode) | 6/30 | 12 | 4/16 | 2/14 |
| 0,2,8,13,14:remap:pool:decode (combined) | 10/30 | 8 | 5/16 | 5/14 |

NoLiMa is near this model's limit even with dense attention (most finished-but-wrong
answers say no character qualifies). Dense and all-layer sparse agree on 24/30 questions
but give identical final text on only 2/30. The combined arm's +4 is within noise and
comes mostly from fewer runs hitting the length cap; read it as "no worse than dense".
A larger thinking budget (~2.5K tokens) would reduce truncation.

## Other 1M-context models, local 16K (`check_adapter.py`, `ul_*` / `qw_*` / `glmfix_*`)

`BB_MODEL_PATH` switches checkpoints; `common.use_model_rope` points v3 and sparse decode at the
model's own RoPE (GLM-4 rotates half of each head, interleaved), and `common.fixed_config` fixes the
GLM-4-1M `-hf` checkpoint's `rope_theta` (10000 -> 1e8; with 10000 even dense attention degenerates by
~16K). `check_adapter.py` (native HF vs dense wrapper / full-coverage sparse kernel / full-coverage
sparse decode on ~1.5K tokens) passes on all three. NF4, exact_tail 0, window 4096, vertical 512,
pooled Phase 0 per model, 5 DP anchors per model (UltraLong ran with R1's 0,2,8,13,14):

| Model | Anchors | NIAH single (dense / all-layer / exp 21) | multikey_1 (dense / all-layer / exp 21) |
|---|---|---|---|
| Llama-3.1-8B-UltraLong-1M | 0,2,8,13,14 | 6 / 6 / 6 of 6 | 12 / 12 / 12 of 12 |
| Qwen2.5-7B-Instruct-1M | 0,5,9,15,21 | 6 / 6 / 6 of 6 | 12 / 12 / 11 of 12 |
| GLM-4-9B-chat-1M (rope fixed) | 0,1,7,15,31 | 6 / 6 / 6 of 6 | 12 / 12 / 12 of 12 |

Sparse decode scores in fp32: in bf16, Qwen2's large logits flipped a near-tie digit.

## H100 results at ~128K (Nemotron-UltraLong-1M, `h100_plan.py`, Oct 30 2026)

BF16, v3 final config (`sink64 window16384 vertical512 exact_tail256`), exp 21 = v3 prefill + sparse decode
with reuse/remap/pooling (`decodeonly`), anchors/head maps from the local 16K UltraLong Phase 0. Cost $2.6
(`benchmarks/exp_21_kascade/phase2a/nemotron_*`, `phase4/nemotron_*`, log `phase4/nemotron_h100_plan.log`).

**Accuracy.** Single-needle NIAH on Thomas's 30 prompts (same token counts as
`modal_rd/results/nemotron_confirm30_20261028`, where dense and v3 are 30/30): exp 21 (5 anchors) **30/30**,
24/24 on the held-out prompts 6-29. multikey_1, n=6: v3 3/6, exp 21 with 3 / 5 / 8 anchors 2/6, 3/6, 3/6.

**Long answers** (summary of ~120-230 tokens, then the answer; n=6): every arm, including v3+evict and
exp 21 with a 1K decode window, ends with the same numbers as v3 except one prompt where v3+evict is right
and v3 is wrong (v3 2/6, v3+evict 3/6, exp 21 2/6 in all three windows). The misses pick another needle's
value, a prefill limit shared by all arms. Token-level agreement is not informative: every arm, eviction
included, diverges from v3 within 3-35 tokens (bf16 near-ties in free text).

**Decode speed** (ms per generated token, followup stage; attention modules alone in brackets; KV read =
estimated fraction of dense KV reads):

| Arm | ms/token | KV read |
|---|---|---|
| v3 (dense decode) | 25.5 (18.3) | 100% |
| v3 + evict (MEMORY_DEFAULTS) | **17.5 (10.1)** | small cache |
| exp 21, decode window 16384 | 53.0 (45.8) | 24% |
| exp 21, decode window 4096 | 29.4 (22.1) | 12% |
| exp 21, window 1024 + routed 1024 | 26.2 (18.8) | 10% |

The 16K decode window made the gather-based PyTorch decode 2x slower than dense; shrinking it brings exp 21
level with v3 but not faster: at 128K on an H100, dense decode over the full cache costs about as much as
exp 21's scoring, top-k and scattered gathers. A speedup needs a fused decode kernel. Peak memory: exp 21
keeps the full cache (35.7 GB vs 27.6 GB with eviction).

**Second question on the same cache:** every arm 6/6, eviction included. The test is too easy: the four
multikey needles share one template, so question 1's routes keep all of them and eviction drops nothing
question 2 needs. Needs a second question about different content.

## Caveat: speed

v3 routes with only the last 64 queries (O(64 * n) per head) while its sparse
attention costs O(n * (window + vertical)), so routing is a small share of v3
prefill and reuse cannot speed v3 prefill much. Reuse pays off where scoring is
expensive: per-token routing in sparse decode (v3 currently decodes dense) or
per-query routing. See Phase 2a timing.
