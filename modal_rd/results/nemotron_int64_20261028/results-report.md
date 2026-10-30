# 64-bit indexing correction — model rerun

Same checkpoint, prompt hashes, routing/window/eviction budgets as the failed run. GPU kernel checks cover the overflow boundary and995993 tokens; sampled nontrivial attention references also passed. Model forwards were guarded against NaN/Inf logits.

| Actual tokens | Exact | Mean prefill s | Peak allocated GB |
|---|---:|---:|---:|
| 127667 | 6/6 | 8.235 | 26.295 |
| 522505 | 1/1 | 33.557 | 47.529 |
| 995993 | 1/1 | 63.594 | 72.999 |

The previously failing995993-token prompt now returns the full correct number5845899 in both warm and timed passes. All warm/timed output IDs match. This supports kernel indexing overflow as the cause of that collapse. It is one prompt at the long lengths, not a final quality benchmark or a matched dense speed comparison. The corrected kernel is slightly slower on the sampled sub-threshold workloads; optimize only after correctness confirmation.

## Six-prompt near-1M confirmation

The corrected kernel was tested on the first six deterministic synthetic NIAH prompts at the 1,000,000-token target. Actual checkpoint input lengths were **995,880–997,726 tokens** (mean 996,625). The six prompts used the same checkpoint revision, source, chat template, BF16 precision, fixed routing and eviction configuration, depth 0.5, seed 42, and H100 80GB as the corrected single-prompt rerun. Every prompt was warmed once before the same six were timed.

| Measure | Result |
|---|---:|
| Full-number exact, warm / timed | **6/6 / 6/6** |
| Mean timed first-forward prefill | **64.302 s** |
| Mean timed full generation wall time | **64.571 s** |
| Peak allocated GPU memory | **73.092 GB** |

Independent checks passed for manifest/result alignment, all six distinct prompt and source hashes, actual token counts, full-number and last-digit scoring, model/source/config identity, first-prompt identity with the prior run, output-ID equality between warm and timed passes, forward counts, and timing/memory sanity. The finite-logit guard stayed enabled. Raw results: `nemotron_int64_1m_six_v2_20261028.json`; checks: `six-prompt-verification.json`; log: `six-prompt-v2-complete.log`. The initial six-prompt launch failed at argument parsing before evaluation and is preserved in `six-prompt-v1-failed.log`.

These six examples are a small synthetic NIAH check at one depth and seed. They do not establish performance on held-out prompts or harder retrieval/long-form tasks. The later single-prompt dense prefill probe is described below.

## Thirty-prompt near-1M result

The 30-prompt evaluation is now complete. All 30 prompts were correct in the warm pass and all 30 were correct in the timed pass. Actual prompt lengths ranged from **995,636 to 998,563 tokens** (mean **996,689**). Combined timed mean prefill was **64.239 s**, mean full generation wall time was **64.504 s**, and the highest measured GPU allocation was **73.137 GB**.

The original worker completed warm-up and timed prompts 0–27, then stopped without a model exception recorded. A separate recovery worker warmed and timed prompts 28–29. Its prompt hashes and answers match the original manifest. The combined check verifies all 30 exact answers and per-prompt warm/timed agreement; timing aggregates span those two workers. Details are in `thirty-prompt-verification.json`, raw data in `nemotron_int64_1m_thirty_20261028.json` and `nemotron_int64_1m_recover28_20261029.json`, and logs in `thirty-prompt-partial.log` and `recovery28-29.log`.

This is a 30-example synthetic single-needle NIAH result for one seed and depth, with no 30-prompt matched dense comparison at 1M. The 128K 30-prompt dense comparison is the basis for the measured 2.14x prefill result.

## Matched dense 1M prefill probe

On the same H100 80GB, checkpoint revision, BF16 weights, MLP chunking, and exact 995,993-token prompt (matching token and source hashes), a dense causal SDPA forward with KV caching disabled took **843.28 s** after warm-up. The matched Bigger Bird sparse prefill took **64.25 s**, a **13.12x prefill speed ratio** for this one prompt. The dense and sparse runs selected the same first next-token ID. Dense peak allocation was **76.77 GB** for its no-cache forward; sparse peak was **73.00 GB** over generation with KV eviction. Those memory peaks come from different cache protocols, so treat them as observed run peaks rather than a fully matched memory comparison. This probe measured no dense answer generation or full-number accuracy. The standard full dense BF16 KV cache would require roughly 130.7 GB for this 995,993-token prompt, before model weights, exceeding the 80GB GPU. The 13.12x number is a measured single-prompt prefill comparison, not a 30-prompt result or a full-generation speedup.

Raw output: `nemotron_dense_1m_prefill_20261029.json`; verification: `dense_1m_prefill_verification.json`; log: `dense-1m-prefill.log`.
