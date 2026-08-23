# Bigger Bird Project — Agent Notes

## Project Overview
Evaluating sparse attention mechanisms on R1-Distill-Llama-8B using RULER (retrieval) and LRA (classification) benchmarks.

## Key Findings

### RULER Evaluation
- **Generative eval is the correct approach** for RULER niah (needle-in-a-haystack retrieval).
  A classification head cannot learn retrieval — it stays at random accuracy (10%) even with
  2000+ training samples and 10 epochs. Zero-shot generative eval works perfectly for dense
  attention (100% at all tested lengths).
- **Dense attention works perfectly through 128K**: 100% at 4K, 8K, 16K, 32K, 64K, 128K.
  The earlier claim that "dense degrades beyond 8K" was WRONG — it was caused by a tokenizer
  truncation bug (max_length=4096) that removed the needle from long prompts. After fixing
  to max_length=131072, dense R1-Llama-8B retrieves needles at 100% accuracy up to 128K.
  Peak memory: 16.22 GB (4K) → 20.03 GB (128K).

### TopK (exp 1) — PROMISING THROUGH 64K (tuned: top_k=512, low_rank_dim=128)
- **Accuracy**: 100% at 4K-16K, 86.7% at 32K, 90.0% at 64K.
  With k=512: 26.7% at 128K. With k=2048: **80% at 128K** (15 examples).
  Retrieval preserved through 64K with k=512, and through 128K with k=2048!
- **Memory**: Nearly flat — 21.13 GB (4K) → 22.15 GB (64K) → 23.23 GB (128K).
  Only ~2 GB increase across 32x length. This is the key advantage of O(n·k) over O(n²).
- **Latency**: Linear scaling — 25.7s/ex (4K) → 397.4s/ex (64K) → 797.1s/ex (128K).
  Doubles per 2x sequence length, confirming O(n·k) scaling.
- **128K recovery**: k=2048 at 128K achieves 80% (vs 26.7% with k=512).
  The 128K barrier is a budget issue, NOT a RoPE generalization limit.
  k=4096 OOMs on 40GB H100 MIG — k=2048 is the practical max at 128K.
- **Key insight**: Content-based top-k routing with k=512 preserves needle retrieval
  through 64K while using O(n·k) attention instead of O(n²). Memory stays nearly constant
  because only k keys/values are gathered per query regardless of sequence length.
  Increasing k from 512 to 2048 extends retrieval to 128K.

### Complete Results Table (RULER niah, depth=0.5, 30 examples unless noted)

| Method | 4K | 8K | 16K | 32K | 64K | 128K | Memory range |
|--------|-----|-----|------|------|------|-------|-------------|
| Dense (exp 0) | 100% | 100% | 100% | 100% | 100% | 100% | 16-20 GB |
| **TopK k=512 (exp 1)** | **100%** | **100%** | **100%** | **86.7%** | **90.0%** | **26.7%** | 21-23 GB |
| **TopK k=2048 (exp 1)** | **100%** | **100%** | **100%** | **86.7%** | **90.0%** | **80%** | 21-23 GB |
| Lightning (exp 2) | 80% | 37% | 30% | 0% | 0% | 0% | 19-21 GB |
| DynGlobals (exp 3) | 0% | 0% | 0% | 0% | 0% | N/A | 19-23 GB |
| PBS (exp 4) | 0% | 7% | 0% | 0% | 0% | 0% | 23-28 GB |
| BiggerBird v1 (exp 5) | 0% | N/A | N/A | N/A | N/A | N/A | 19 GB |
| LayerAdaptive (exp 7) | 0% | 0% | 0% | 0% | 0% | N/A | 19-20 GB |
| TokenDrop (exp 8) | 0% | 0% | 0% | 0% | 0% | 0% | 21-22 GB |
| GQA Sparse k=512 (exp 10) | 100% | 97% | 100% | 87% | 90% | 10% | 18-23 GB |
| **GQA Sparse k=2048 (exp 10)** | — | — | — | — | — | **60%** | 33 GB |
| NSA (exp 11) | 0% | 0% | 0% | 0% | 0% | N/A | 19-24 GB |
| S2-HHST (exp 12) | 0% | 0% | 0% | 0% | 0% | N/A | 17-22 GB |
| DynamicContext k=512 (exp 13) | 100% | 100% | 100% | 87% | 90% | 10% | 21-23 GB |
| BiggerBird proper (exp 15) | 100% | 100% | 100% | 87% | 90% | 13% | 21-32 GB |
| CoarseToFine blocks=256, fine_k=512 (exp 17) | 100% | 100% | 100% | 87% | **93%** | 0% | 21-23 GB |
| **CoarseToFine fine_k=2048 (exp 17)** | — | — | — | — | — | **60%** | 33 GB |

Note: 128K results with k=2048 used 5-15 examples (not 30) due to 12h job time limit.
exp_17 at 32K used topk_blocks=128 (50% coverage), at 64K used topk_blocks=256 (50% coverage).
exp_17 coverage-accuracy relationship at 64K: 6%→3%, 25%→73%, 50%→93%.
exp_17 at 128K with fine_k=512: 0% regardless of block coverage (25% or 50%).
exp_17 at 128K with fine_k=2048: 60% — fine budget is the bottleneck, not block coverage.
exp_10 k=2048 at 128K: 60% (5 examples), 33 GB memory (vs 23 GB for k=512).
exp_1 k=1024 at 128K: 0% (10 examples) — sharp phase transition between k=1024 and k=2048.
Budget-accuracy phase transition at 128K: k=512→10-27%, k=1024→0%, k=2048→60-80%.

### Other Sparse Methods — ALL FAIL RETRIEVAL
- **Exp 2 (Lightning Hybrid, fixed)**: 80% at 4K → 0% at 32K+. top_k=256 too small.
  Memory: 18.65-20.75 GB. Linear time scaling but accuracy collapses.
- **Exp 3 (Dynamic Globals, fixed)**: 0% at ALL lengths (4K-64K). Learned gate is randomly
  initialized (untrained) — selects irrelevant tokens as globals. Only 16 globals out of
  thousands of tokens means needle is almost never selected.
- **Exp 4 (PBS)**: 0% at most lengths, 6.7% at 8K. Block selection with 8 blocks insufficient.
  Memory: 22.78-28.20 GB (highest of all methods). Also slowest at 128K (531s/ex).
- **Exp 5 (BiggerBird v1)**: 0% at 4K. Gate-based selection fails like exp 3.
- **Exp 7 (LayerAdaptive)**: 0% at ALL lengths (4K-64K). Layer-adaptive top-k with small k
  values (k_early=192, k_mid=64, k_late=32) fails. Memory: 18.68-20.08 GB.
- **Exp 8 (TokenDrop)**: 0% at ALL lengths (4K-128K). Dropping 30% of tokens destroys retrieval.
  Memory: 21.13-22.15 GB. Same as TopK but without content-based selection.
- **Exp 10 (GQA Sparse)**: 0% at ALL lengths (4K-64K). Top-k with top_k=64 too small.
  Memory: 18.23-19.25 GB. Most memory-efficient but completely fails retrieval.
- **Exp 11 (NSA)**: Fixed GQA tensor size mismatch and BFloat16 projection bugs. The sparse-only run
  completed 0% at 4K, 8K, 16K, 32K, and 64K. Memory rose from 18.98 GB at 4K to 24.19 GB at 64K.
- **Exp 12 (S2-HHST)**: 0% at ALL lengths (4K-64K). Strided block-sparse pattern misses needle.
  Memory: 17.41-21.54 GB. Most memory-efficient of all methods.

### Key Insight: Content-Based Routing is Essential
- Methods that use content-based routing (TopK with QK similarity) can preserve retrieval
  because the needle's distinctive content makes it likely to be selected.
- Methods that use fixed/learned selection (TokenDrop, Dynamic Globals, PBS, LayerAdaptive,
  S2-HHST) fail because they don't prioritize the needle's content.
- Methods with too-small budgets (Lightning with k=256) fail at long context because the
  probability of selecting the needle decreases with sequence length.
- Only TopK with k=512 achieves useful retrieval (100% through 16K, ~90% at 32K-64K).

### R&D: New Parameter-Free Experiments (exp 16, 17)
- **Root cause analysis**: Experiments 3, 5, 11, 15 failed at 0% because they use learned
  routing components (nn.Linear gates, compress_k/v, gate_mlp) that are randomly initialized
  and never trained in zero-shot RULER eval. Random routing = random key selection = needle
  almost never selected.
- **Exp 10 (GQA sparse) with k=512**: 100% at 4K, 13.68s/ex, 18.68 GB. Previous 0% was due
  to using k=64 (default). With k=512 and low_rank_dim=128, it matches exp 1's performance.
  This confirms token-level top-k routing is the winning approach.
- **Exp 16 (Free NSA)**: 0% at 4K. Block-mean routing dilutes the needle signal — the needle
  is only a few tokens in a 64-token block, so the block mean is dominated by filler.
  Three-branch combination (window + compressed + selected) with fixed gates doesn't help
  because none of the branches can find the needle.
- **Exp 17 (Coarse-to-fine)**: 43.3% at 4K. Two-stage routing (block selection + token top-k
  within blocks) shows partial promise. At 4K with block_size=128, all 32 blocks are selected,
  so the coarse stage doesn't filter — the fine stage does token-level top-k=512, which should
  match exp 1 but gets 43.3% instead of 100%. Possible numerical precision issue in gather.
- **Exp 15 (BiggerBird proper, tuned)**: 46.7% at 4K. With globals_per_head=32, the global
  tokens cover the entire 4K sequence, but duplicate indices in the routed set distort the
  softmax. Need to deduplicate indices before passing to causal_sparse_attention.
- **Exp 4 (PBS, tuned)**: 0% at 4K with block_size=128, num_blocks=8. Block routing still
  fails because block-mean QK doesn't distinguish the needle's block.
- **Key insight confirmed**: Token-level content-based routing (exp 1, exp 10) is the ONLY
  approach that works for NIAH. Block-level routing fails because the needle is a tiny
  fraction of any block. The solution is to increase k, not to use block-level routing.
- **Slurm jobs**: 54774271 (full sweep, 12h — exp 1 at 128K with k=1024), 54808873 (follow-up
  sweep — exp 10 at 8K-128K, exp 17 with smaller blocks).

### Sparse-Only Attention Audit
- `exp_0` is the only permitted dense baseline.
- Experiments `exp_1`–`exp_15` now use sparse implementations in both causal/generative and
  bidirectional/classification paths. Short sequences use sparse gather/top-k paths too; they do
  not call `dense_self_attention`.
- Exp 9's training-only full-QK verifier was removed because it violated the sparse-only contract.
- Exp 12's optional dense layer was removed; passing non-empty `dense_layers` now raises an error.
- The audit is enforced by `scripts/build_dashboard.py` and recorded in
  `benchmarks/dashboard_build.log` and `benchmarks/dashboard_build_report.json`.
- Current audit result: PASS (only exp 0 contains `dense_self_attention`).
- Dashboard analysis excludes R1 observations timestamped before `20260813_000000`, because those
  historical runs may have used the now-removed dense causal fallback. They remain visible in the
  source-run table with `pre_sparse_fix` validity.

### Causal Sparse Attention Patterns
- Pattern A (top-k): exp_1, 7, 10, 14 — last-query_topk_indices + causal_sparse_attention
- Pattern B (block): exp_4, 6 — last_query_block_topk_indices + causal_sparse_attention
- Pattern C (gate/anchor): exp_3, 5, 9 — gate/anchor selection + causal_sparse_attention
- Pattern D (linear/local): exp_2 — causal local window + content-routed global keys
- Pattern E (multi-branch): exp_11 (NSA) — causal 3-branch, exp_12 (S2-HHST) — causal strided
- Pattern F (token budget): exp_8, 13, 15 — bounded token selection + sparse attention
- Key utilities: causal_sparse_attention, last_query_topk_indices, last_query_block_topk_indices,
  causal_linear_attention, causal_sparse_attention_with_indices (all in sparse_attn_utils.py)

### LRA Evaluation
- **LRA text (IMDb)**: Works with classification head (55-89% accuracy).
- **LRA listops**: Cannot be learned by classification head (stays at random ~10% even with
  2000 train samples / 10 epochs). Few-shot generative eval also fails (16% with 3-10 shots).
  Listops requires recursive computation that a frozen Llama model can't learn from limited data.

### Model Config
- Model: DeepSeek-R1-Distill-Llama-8B
- `max_position_embeddings`: 131072 (128K)
- `original_max_position_embeddings`: 8192
- `rope_scaling`: factor 8.0, llama3 type
- GPU: H100 40GB MIG partition
- Dense attention works through 128K (no OOM, 100% accuracy)

## Build & Run Commands
```bash
# Activate environment
source /scratch/$USER/r1-venv/bin/activate
export HF_HOME=/scratch/$USER/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# RULER generative eval (zero-shot)
python -m eval.ruler_llama.run_generative --task niah --exp 0 --seq 4096 --depth 0.5 --eval-samples 128 --max-examples 30

# LRA classification eval
python -m eval.lra_llama.run --task listops --exp 0 --seq 1024 --size lra-report

# LRA listops generative eval (few-shot)
python -m eval.lra_llama.run_generative_listops --exp 0 --seq 1024 --shots 5 --eval-samples 128 --max-examples 30

# Rebuild dashboard and durable diagnostics
python scripts/build_dashboard.py
cat benchmarks/dashboard_build.log
```

## Slurm
- Partitions: `gpubase_bygpu_b1` (3h), `gpubase_bygpu_b5` (7h), `gpubase_bygpu_b2` (12h)
- GPU: `nvidia_h100_80gb_hbm3_3g.40gb:1`
- Account: `def-guerzhoy`

## Sparse Attention Experiments
- exp 0: Dense baseline (SDPA, causal)
- exp 1: DeepSeek top-k (head-shared top-k selection, TUNED: k=512, low_rank_dim=128) — PROMISING
- exp 2: Lightning Hybrid (local window + ELU linear attention, block_size=128)
- exp 3: Dynamic Globals (local window + learned gate global tokens, window=64, globals=16)
- exp 4: PBS (block-level top-k, block_size=64, num_blocks=2)
- exp 5: BiggerBird v1 (local MMR + global gate + teleports, window=64, local_k=32, globals=16)
- exp 6: DeepSeek PBS (block routing + top-k refinement, block_size=32, num_blocks=4, top_k=64)
- exp 7: Layer Adaptive (layer-adaptive top-k: k_early=192, k_mid=64, k_late=32)
- exp 8: Token drop (drop 30% of tokens after layer 3)
- exp 9: Attention Speculation (local window + anchors, window=64, anchors=4)
- exp 10: GQA Sparse (top-k with low-rank proxy, top_k=64)
- exp 11: NSA (3-branch: compressed + selected + sliding window, block_size=32, topk_blocks=4)
- exp 12: S2-HHST (strided block-sparse, shard_size=32, local_blocks=2, stride=16)
- exp 13: Dynamic context (target budget 4096)
- exp 14: Token drop + DeepSeek top-k
- exp 15: BiggerBird (block-mean routing, fragment_size=64, max_k=512, globals=8)
- exp 18: Confidence-gated multi-resolution attention (local window + low-rank causal linear branch + head-shared routed global correction; optimized Triton inference kernels, validation in progress)
