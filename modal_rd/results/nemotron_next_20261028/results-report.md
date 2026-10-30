# Nemotron eviction and scaling probe

Status: complete. Checkpoint: `fa3a10b9c6ad02f42c81e1d68dcf6967afb230ef`.

Single H100, BF16, native chat, greedy max64, cache eviction keep16384/pool8, sink64/window16384/routes512/exact-tail256, MLP chunk8192 and CPU embeddings.

| Target | Actual tokens | Completed | Exact | Prefill s | Task s | Peak allocated GB |
|---|---|---:|---:|---:|---:|---:|
| 128000 | 127253–127768 | 6 | 6/6 | 7.760 | 7.982 | 26.295 |
| 262144 | 261528–261528 | 1 | 1/1 | 15.836 | 16.065 | 33.491 |
| 524288 | 522505–522505 | 1 | 1/1 | 31.686 | 31.901 | 47.529 |
| 1000000 | 995993–995993 | 1 | 0/1 | 55.731 | 56.673 | 72.999 |

The six-example 128K phase checks eviction quality against a prior matched control. Longer phases have one prompt each; their scores are feasibility evidence, not a benchmark. There is no matched dense baseline beyond 128K, no tuned budget selection, and no general long-generation or multi-question cache claim.

Matched six-prompt full-cache v3 control: 6/6 exact, 7.746s prefill and 38.148GB allocated peak. Eviction allocated peak is 31.1% lower. Kernel, attention, checkpoint, GPU type and prompt hashes match; these are separate runs.
