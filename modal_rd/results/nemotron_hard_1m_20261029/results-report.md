# Nemotron harder retrieval pilot at approximately 1M tokens

Sparse-only screen: 3 prompts per task, depth 0.5, seed 42, BF16, H100 80GB, no dense baseline. All prompts warmed before timed runs. The scorer counts an example correct when all gold answer strings occur in the output, ignoring whitespace. Multiquery score does not verify answer-to-key pairing. Three examples do not establish general accuracy.

| Task | Actual tokens | All answers present | Mean answer recall | Mean prefill (s) | Peak allocated (GB) |
|---|---:|---:|---:|---:|---:|
| niah_single_2 | 979,155–979,886 | 3/3 | 1.00 | 61.84 | 72.13 |
| niah_single_3 | 979,137–979,870 | 3/3 | 1.00 | 61.85 | 72.13 |
| niah_multikey_1 | 979,140–979,867 | 1/3 | 0.33 | 61.55 | 72.13 |
| niah_multikey_3 | 970,838–971,286 | 1/3 | 0.33 | 61.31 | 71.67 |
| niah_multivalue | 979,149–979,877 | 0/3 | 0.25 | 61.83 | 72.13 |
| niah_multiquery | 979,132–979,858 | 1/3 | 0.83 | 61.77 | 72.13 |

No accuracy or speed comparison with dense was run at this length. Max output 64 tokens; KV eviction enabled. These are repository RULER-style synthetic tasks, not the full official RULER suite.

Sources: `nemotron_hard_1m_pilot_20261029.json`, `nemotron_hard_1m_pilot_20261029_plan.json`.
