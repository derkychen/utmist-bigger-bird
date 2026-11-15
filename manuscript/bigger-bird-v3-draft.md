# Bigger Bird: Sparse Attention for Long-Context Inference

*Draft for supervisor review · Authors and affiliations to be confirmed*

## Problem and approach

Long prompts are useful for repository QA and multi-document reasoning, but dense attention compares each prompt position with every earlier position. As the context grows, this pairwise work and its memory demands make prompt processing expensive. We ask whether the model can keep useful links across a long prompt while computing attention over fewer positions.

A recent frontier example is DeepSeek's DSA: a lightweight indexer scores prior tokens and selects top-k KV entries per query. This reduces the attention step, but the indexer still scores the preceding context and the model was further trained with sparse attention. [BigBird](https://arxiv.org/abs/2007.14062) uses local, random, and global links; [MInference](https://arxiv.org/abs/2407.02490) assigns heads sparse pattern families. [SnapKV](https://arxiv.org/abs/2404.14469) instead evicts KV entries based on attention near the prompt end, saving cache memory but making discarded entries unavailable later. Recent [LSA](https://arxiv.org/abs/2610.42147) and [ZHA](https://arxiv.org/abs/2610.42112) preprints use hierarchical key summaries to reduce routing work; they are promising future comparisons, not methods evaluated in this study.

Bigger Bird's current Triton kernel combines 64 initial sink tokens, a 16,384-token local window, and up to 512 routed positions per head. It scores candidate distant positions using their mean causal attention probability across the final 64 prompt queries, then selects each head's top positions. Unlike a fixed link graph or a chosen pattern family, this shortlist can change with the prompt and can include any distant key. The final 256 prompt queries attend exactly to their visible prefix. Like other selection methods, it can still omit useful information; here, the main risk is that the final 64 queries may not surface what earlier queries need.

The intuition is coverage: keep early anchors, nearby context, and a small set of distant tokens the model itself attends to near the answer. That may preserve useful information while avoiding many other query-key comparisons. It depends on the final 64 queries surfacing distant tokens needed elsewhere. Routing can miss information earlier queries need, and it changes when the prompt is extended. The six-prompt tuning experiment below tests one part of the design; it does not establish a general accuracy guarantee.

## R1 benchmark: real long-context questions

We evaluated 1,178 paired examples on one H100 80 GB using BF16 R1-Distill-Llama-8B. LongCodeQA tests repository comprehension, Code.Debug tests bug localization, and LongBench v2 tests long-context reasoning. Accuracy is the first-token choice among A–D. Prefill is a full-prompt forward; decode is a separate 16-step CUDA-graph timing probe. In the table, BB denotes Bigger Bird.

| Benchmark | n | Accuracy dense / BB | Prefill dense / BB (s) | Prefill ratio |
|---|---:|---:|---:|---:|
| LongCodeQA 32K | 113 | 67.3% / 67.3% | 0.79 / 1.17 | 0.67× |
| LongCodeQA 64K | 76 | 63.2% / 63.2% | 2.55 / 3.17 | 0.80× |
| LongCodeQA 128K | 92 | 52.2% / 51.1% | 6.07 / 5.59 | 1.09× |
| Code.Debug | 394 | 27.7% / 26.6% | 8.75 / 6.89 | 1.27× |
| LongBench v2 | 503 | 26.8% / 26.0% | 6.70 / 5.31 | 1.26× |

The prefill ratio is mean dense time divided by mean Bigger Bird time; above 1× favors Bigger Bird. The 128K bucket has a 0.99× median paired ratio and averages about 88K actual tokens. Mean decode, dense / Bigger Bird, was 10.42 / 10.41, 12.08 / 12.08, 14.06 / 14.08, 15.33 / 15.34, and 13.76 / 13.77 ms/token, respectively. Pooled accuracy was 35.3% / 34.6%; paired McNemar p >= 0.683 does not establish equivalence. Code.Debug and LongBench v2 prompts beyond 130K were middle-truncated. These scores do not measure patch success.

<!-- pagebreak -->

## One-million-token evaluations

| Evaluation | Quality | Full-prompt prefill |
|---|---|---|
| Official LongCodeQA 1M · 18/50 prompts · 542K–878K tokens | Dense 7/18 (38.9%) · BB 9/18 (50.0%) · paired p = 0.625 | 470.38 → 45.97 s · **10.23×** mean ratio · 9.48× median paired ratio |
| Synthetic RULER · 18 prompts · 971K–980K tokens | Dense 10/18 (55.6%) · BB 9/18 (50.0%) | No fair ratio |
| Exp 21 held-out facts · 25 prompts · 4 facts each | Eviction 16K / 64K / 128K: 62% / 85% / 86% · 4-bit full cache: 91% | Median: 63 s eviction · 72–78 s compressed full cache |

LongCodeQA used the same H100 and BF16 model for both arms; prompts were identical and untruncated. This no-KV-cache test measures first-answer logits, not decoding or cache retention; 32 prompts remain unmeasured. RULER used dense on a B200 with CPU-offloaded BF16 KV and BB on an H100; its multi-query scorer checks answer-string presence, not answer-to-key pairing. Exp 21 has no dense BF16 full-cache quality baseline at 1M; decode values are estimates.

## Earlier project experiments

These were screening runs for repository prototypes on one synthetic needle task: byte budget 65,536, 30 examples, depth 0.5, R1 on an H100 3g.40GB MIG, and no KV cache. The scorer extracts the first number in each output and checks its last digit; a response with no number is counted wrong. So a 0/30 means none of the 30 outputs passed that test. Several saved zero-score examples are non-numeric or incomplete. This shows those configurations failed this task; it does not evaluate the corresponding published methods.

| Screening result | Experiments | Score | Relative to dense |
|---|---|---:|---:|
| Dense reference | Exp 0 | 30/30 | 100% |
| Content selection / routing | Exp 1 Top-k, 10 GQA, 13 DynamicContext, 15 Earlier Bigger Bird; **each** | 27/30 | 90% |
| Coarse-to-fine | Exp 17 | 22/30 | 73% |
| Bigger Bird Flash | Exp 19 | 16/30 | 53% |
| Zero-score prototype screens | Exp 2, 3, 4, 7, 8, 11, 12, 18; **each** | 0/30 | 0% |
| No row selected for this setting | Exp 5, 6, 9, 14, 16 | — | — |

Relative score is each run's last-digit accuracy divided by dense's 30/30. The table is a development record, not a fair ranking: it selects one saved run per approach, uses incomplete configuration and checkpoint provenance, and does not record separate prefill/decode times. Exp 11 also added untrained components to frozen weights. The other runs for these experiments remain available in the dashboard.

<!-- pagebreak -->

## Matched synthetic design experiments

On six matched ~127.5K-token development prompts, dense, exp 19 and Bigger Bird scored 100%, 0% and 100% exact answers. Their mean prefills were 16.33, 4.33 and 8.50 s: dense/Bigger Bird **1.92×**, dense/exp 19 **3.77×**. With a 4K local window, exact tails of 0 / 64 / 256 queries scored 33.3% / 66.7% / 100%, taking 4.95 / 5.45 / 6.09 s. These six prompts were used in tuning.

In a 30-prompt confirmation at ~128K, dense and Bigger Bird scored 93.3% / 100%; the first six were used in tuning. Mean prefill was 16.25 / 8.61 s (**1.89×**). On harder untuned tasks, dense / Bigger Bird scored 17/24 / 11/24 on single-needle retrieval and 4/12 / 1/12 on multi-key retrieval; both scored 0/12 on UUID retrieval. Performance therefore depends on the task.

## Limitations

The related methods above were not run under our matched benchmark protocol, so these are design distinctions, not a head-to-head ranking. LSA reports experiments up to 10M tokens, with 20 prompts per task above 1M; ZHA evaluates quality to 256K and extrapolates latency beyond 1M. Their model and hardware setups differ from ours.

R1 timings used short warmups, and the resolved upstream R1 weight revision was not recorded. The 1M pilot uses a different model and no-cache protocol. Full-cache Bigger Bird retains its KV cache; first-token scores do not establish longer-generation quality after eviction. Next steps are the remaining 1M prompts, longer-generation and eviction tests, and matched comparisons with DSA, LSA, and ZHA.

## Sources

[Final benchmark report](../modal_rd/results/final_bigger_bird_20261101/results-report.md) · [Synthetic results](../modal_rd/RESULTS.md) · [All experiments dashboard](../dashboard.html) · [BigBird](https://arxiv.org/abs/2007.14062) · [MInference](https://arxiv.org/abs/2407.02490) · [SnapKV](https://arxiv.org/abs/2404.14469) · [DeepSeek-V3.2 / DSA](https://arxiv.org/abs/2512.02556) · [LSA](https://arxiv.org/abs/2610.42147) · [ZHA](https://arxiv.org/abs/2610.42112)
