# Bigger Bird: Content-Aware Sparse Transformer Attention

## Start here (status as of 2026-10-25)

**Bigger Bird v3** is a training-free sparse attention for long-context **prefill** on
DeepSeek-R1-Distill-Llama-8B (BF16). For each layer, every token reads three things: the first 64 tokens (sink), the
previous 16,384 tokens (window), and 512 positions per head that the last 64 prompt tokens pick by softmax attention
probability (routed). The last 256 prompt tokens read everything exactly (exact tail). Decode is exact on the KV cache.

Final config: `{"sink":64,"window":16384,"vertical":512,"exact_tail":256}`.

Everything below this section, plus `COMMANDS.md`, `research-plan*.md` and exp 1–19 under `experiments/`, is
**earlier work** (BART/BigBird/LRA era and the Slurm R1 sweep). Keep it for history; the current code is listed here.

### Where we are

| Area | Result | Source |
|---|---|---|
| RULER NIAH, ~128K, 30 docs | 30/30 vs dense 28/30 (24/24 each on held-out prompts 6–29), 1.9× faster prefill | `modal_rd/RESULTS.md` |
| Real benchmarks (exp 20, **A100 pilot**) | LongCodeQA 32K/64K/128K: identical to dense. LongBench v2, Code.Debug: near chance even for dense, no significant change. Prefill 1.47× at ~117K, slower than FlashAttention below ~50K | `experiments/exp_20_sota_long_context/` |
| Memory (PRs #17–#23) | ~127K on an A100 40GB: peak 38.1 → 26.3 GB with KV eviction, 20/20 NIAH, decode ~2× faster | `modal_rd/MEMORY_DEFAULTS.md` |
| Harder NIAH at ~128K | multi-key / multi-query / UUID tasks: both dense and sparse struggle; sparse trails on multi-key | `modal_rd/RESULTS.md` |

### What is next (open work)

1. **Exp 20 full run on a Modal H100** (`run_h100.sh h100_130k`, ~10–12 H100-hours). Still missing: 294 Code.Debug and
   403 LongBench v2 items. Rerun all 1,178 items on one GPU type so speed numbers are comparable.
2. **Dense baselines for multi-needle at ~125K.** Multiquery is 0/10 with and without eviction, so the sparse prefill
   config is the suspect; the dense rows are unfinished (command in `MEMORY_DEFAULTS.md`).
3. **Long-generation (CoT) eval** to test whether KV eviction hurts decode quality. Under the current no-CoT scoring,
   eviction cannot change the answer.
4. **Move to the new base model, NVIDIA Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct** (not in the repo yet). Start with
   a matched dense vs sparse comparison at 128K, then extend. A full BF16 KV cache is ~128 GiB at 1M tokens and ~512 GiB
   at 4M, so long runs need eviction / INT8 and multiple GPUs.
5. **Kernel and memory work:** strided kernel output (−0.5 to −1 GB), drop HF's `inputs_embeds` reference
   (−0.5 to −1 GB), a `window=4096` arm for speed below 64K. List in `MEMORY_DEFAULTS.md` → "Further memory improvements".
6. **Unmerged branches to review:** `origin/exp20-sota-long-context` (full exp 20 write-up, one commit ahead of main)
   and `origin/david-test` (exp 21: anchor-layer route reuse and a NoLiMa eval).

### Files that matter

| Path | What it is |
|---|---|
| `modal_rd/attention.py` | **The method.** `RDAttention`: `routes()` (last-64 scoring, top-512 per head), sparse prefill, exact tail, KV eviction, in-place RoPE |
| `kernels/bigger_bird_flash.py` | Triton kernel for sink + window + routed attention (GQA K/V read in-kernel) |
| `modal_rd/eval_niah.py` | RULER NIAH evaluation. Arms: `dense` (HF SDPA), `dense_rd` (exact, same memory savers), `vs` (Bigger Bird v3) |
| `modal_rd/niah.py` | Prompt builder; matches the CCDB Round 4 prompts token for token (`modal_rd/manifests/`) |
| `modal_rd/mlp_chunk.py`, `cpu_embed.py`, `kv_int8.py` | Memory savers: chunked MLP + RMSNorm (exact), CPU embedding table (exact), INT8 KV cache (lossy, opt-in) |
| `modal_rd/app.py` | Modal harness: image, `bb-vol` volume, H100 runner |
| `modal_rd/selfcheck.py`, `kernelcheck.py` | Correctness checks to run before long jobs |
| `modal_rd/report.py` | Pulls results, rebuilds `modal_rd/RESULTS.md` and `modal_rd/figures/` |
| `modal_rd/RESULTS.md`, `MEMORY_DEFAULTS.md` | NIAH results tables; memory decision and evidence |
| `experiments/exp_20_sota_long_context/` | Real-benchmark eval: `run.py`, `tasks.py`, `report.py`, `run_a100.sh`, `run_h100.sh`, `README.md`, `RESULTS.md` |
| `tests/test_rd_attention_gqa.py`, `test_kv_evict.py`, `test_kv_int8.py`, `test_mlp_chunk.py`, `test_exact_memory_savers.py` | Tests for the current stack |
| `modal_rd/deck/index.html` | EigenAI presentation (open in a browser) |
| `AGENTS.md` | Detailed running notes and the full history of results |

### How to run

**Modal (H100 80GB), one-time setup:**

```bash
pip install modal
modal setup                                    # log in; ask to be invited to the team Modal workspace
modal run modal_rd/app.py::download            # model weights -> volume bb-vol:/vol/models
```

**Before long runs**, check kernels against exact attention:

```bash
modal run modal_rd/app.py --module modal_rd.selfcheck
modal run modal_rd/app.py --module modal_rd.kernelcheck
```

**NIAH at ~127K with the default (memory-saving) config:**

```bash
modal run modal_rd/app.py --module modal_rd.eval_niah --args "--arm vs --attn '{\"sink\":64,\"window\":16384,\"vertical\":512,\"exact_tail\":256,\"kv_evict\":true,\"evict_pool\":8,\"evict_keep\":16384}' --mlp-chunk 8192 --cpu-embed --bytes 512000 --n 10 --tag default_127k"
```

Dense baseline: same command with `--arm dense_rd` and no `--attn`. Other knobs: `--task niah_multikey_1` (and other RULER
variants), `--depths 0.1,0.5`, `--max-new 64`, `--kv-int8` (only if memory is tight). Add `--detach` after
`modal run` to keep a job running after you close the terminal. Results land in `bb-vol:/vol/results/<tag>.json`.

**Exp 20 full real-benchmark run (next job):**

```bash
bash experiments/exp_20_sota_long_context/run_h100.sh h100_130k     # 3 detached containers; rerun to resume
modal volume get bb-vol results/exp20/h100_130k ~/scratch/results/exp20/
SCRATCH=~/scratch python -m experiments.exp_20_sota_long_context.report --tag h100_130k \
    --out experiments/exp_20_sota_long_context/RESULTS.md
```

**Local GPU (A100/H100, model under `$SCRATCH/models`):** drop the `modal run ... --args` wrapper and call the module
directly, e.g. `python -m modal_rd.eval_niah --arm vs --attn '{...}' --mlp-chunk 8192 --cpu-embed --bytes 512000 --n 10 --tag x`.

**Tests:** `pytest tests/test_rd_attention_gqa.py tests/test_kv_evict.py tests/test_kv_int8.py tests/test_mlp_chunk.py tests/test_exact_memory_savers.py`
(run them on a CUDA GPU; only some pass on CPU, e.g. `test_exact_memory_savers.py`).

**Rebuild NIAH tables and charts:** `python modal_rd/report.py`.

### Gotchas

- Don't write logs inside the repo while `modal run` is building the image (it aborts the build). Use `modal_rd/logs/`,
  which is excluded.
- `modal_rd/app.py` defaults to `H100!` (no silent H200 upgrade) so timings stay comparable. `BB_GPU=none` registers only
  CPU functions.
- Compare speed only between runs on the same GPU type, with the same prompts, tokenizer and stopping rule.
- Below ~16K tokens Bigger Bird v3 is identical to dense (the window covers the whole prompt), so short benchmarks tell
  you nothing about it.

---

## Overview (historical: original proposal and exp 1–15)

Bigger Bird is a research project conducted by the University of Toronto Machine Intelligence Student Team (UTMIST). The primary objective is to reduce the computational complexity of standard Transformer self-attention from $O(n^2)$ to $O(n)$ using content-aware sparse routing.

Standard sparse attention models, such as Big Bird and Longformer, utilize fixed or random attention patterns. Bigger Bird improves upon these by implementing discrete routing mechanisms that adaptively select relevant tokens based on input content, aiming for higher accuracy with significantly fewer softmax comparisons.

## Current Experimental Implementations

### Original Single-Idea Methods (exp 1-4)

> **Implementation note (May 2026):** Exp 1–4 kernels were refactored for efficient PyTorch (head-shared routing, no full `T×T` score tensors, BART first-token pooling). See [context/exp-1-4-fixes-may2026.md](context/exp-1-4-fixes-may2026.md) for why older benchmarks looked bad and what changed. Use `--fixed-length` for long-context tests (dynamic padding often yields ~150–200 tokens on IMDb).

1.  **DeepSeek-Inspired Top-K Routing (exp_1_deepseek_topk)**: Low-rank Lightning Indexer–style routing (`d_low=16`), then full-dim softmax on **K=64 keys per head** (shared across query positions for speed). Defaults: `top_k=64`, `low_rank_dim=16`.
2.  **Lightning Hybrid Attention (exp_2_lightning_hybrid)**: Sliding-window softmax via `unfold` (no full `QK^T` matrix) plus optional linear attention when `T > 4 × block_size`. Default `block_size=128`.
3.  **Content-Aware Dynamic Globals (exp_3_dynamic_globals)**: Learned gate picks **G=16 global tokens per batch**; local context via sliding window (`W=64`); single softmax over **G + W** keys. Defaults: `window_size=64`, `num_globals=16`.
4.  **Permuted Block-Sparse Attention (exp_4_pbs_attn)**: Head-shared block routing, **sorted** token indices for coalesced GPU reads, sparse softmax on **M×block_size** tokens. Defaults: `block_size=64`, `num_blocks=2` (128 keys).

### Advanced Hybrid / Diverging Ideas (exp 5-13)

5.  **Bigger Bird — Unified (exp_5_bigger_bird)**: Reimplements the **original proposal**: three components combined in a single attention module — (i) diversity-aware local top-k via MMR-lite, (ii) submodular-style global token selection via learned gate, (iii) biased random "teleports" mixing high-gate candidates with uniform random. Selects only `local_k + globals + teleports` keys per query.
6.  **DeepSeek + PBS Hybrid (exp_6_deepseek_pbs)**: Combines content-aware routing with GPU-friendly block coalescing. Low-rank Q/K projections identify the top-M most-relevant blocks (PBS-style), then top-K tokens are selected *within* those blocks. Selected indices are sorted to guarantee contiguous memory reads.
7.  **Layer-Adaptive Sparsity (exp_7_layer_adaptive)**: Different layers do different work — early layers extract local syntax (need more context), middle layers compose semantics, late layers do high-level reasoning. This experiment uses a per-layer top-k schedule: dense at the bottom (`k_early=192`), moderate in the middle (`k_mid=64`), aggressive at the top (`k_late=32`).
8.  **Token Dropping (exp_8_token_drop)**: Diverges from sparse attention entirely. After 3 dense layers extract local features, computes token importance (L2 norm of hidden state) and **drops the bottom 30%**. Subsequent layers process a shorter sequence — attention cost drops quadratically with `keep_ratio²`.
9.  **Attention Speculation (exp_9_attn_specul)**: Inspired by speculative decoding. Every layer runs a *fast path* (window + first/middle/last anchors). Every 4th layer also computes the *full* attention as a verifier and adds a KL distillation loss, teaching the fast path to mimic full attention during training. At inference, only the fast path runs.
10. **GQA + Sparse Routing (exp_10_gqa_sparse)**: Stacks two memory wins from the DeepSeek-V3 playbook. (a) Grouped-Query Attention compresses 12 K/V heads into 4 KV groups via mean pooling — 3× memory reduction. (b) Top-K sparse routing on top — softmax cost drops from O(N) to O(K) per query.
11. **Native Sparse Attention — NSA (exp_11_nsa)**: TODO
12. **S2-HHST (exp_12_s2_hhst)**: TODO
13. **Dynamic Context Window (exp_13_dynamic_context)**: Builds on Token Drop to accept **arbitrarily long inputs** by capping the post-early-layer attention to a fixed token budget. After `drop_after_layer` dense layers, tokens are scored by hidden-state L2 norm and exactly `target_budget` tokens are kept — regardless of input length (4k, 100k, 1M). For very long inputs (> chunk_size), early layers run in independent chunks so the full O(n²) attention matrix never materialises. Late layers always run on the fixed budget, giving O(budget²) attention cost. Defaults: `drop_after_layer=3`, `target_budget=4096`, `chunk_size=8192`.
14. **Token Drop + DeepSeek Top-K (exp_14_token_drop_deepseek)**: Combines Token Dropping with DeepSeek-style sparse attention for **two levels of sparsity**. Early layers run dense on the full sequence to extract local syntax. After `drop_after_layer`, low-importance tokens are dropped. Remaining layers use **DeepSeek low-rank top-K routing** on the shorter sequence — giving both reduced sequence length AND fewer keys per query. Defaults: `drop_after_layer=3`, `drop_ratio=0.3`, `top_k=64`, `low_rank_dim=16`.
15. **Proper Bigger Bird (exp_15_bigger_bird)**: The original UTMIST Bigger Bird implementation built on top of BigBird-RoBERTa. Extends `BigBirdBlockSparseAttention` with three content-aware modifications to the middle-block attention path: (i) diversity-aware local top-k via MMR-lite, (ii) submodular-style global token selection via facility-location, (iii) biased random "teleports" with block-wise routing. Uses `google/bigbird-roberta-base` as the backbone (not BART). Defaults: `fragment_size=128`, `max_k=64`, `globals_per_head=6`, `teleports_per_head=4`.

## Architecture Diagrams

### Base Model: BART-base (Standard Transformer)

> **Analogy:** Imagine you're in a room of 768 people and every single person has to shake hands with every other person before anyone can speak. That's standard attention — thorough, but exhausting and slow. As the room gets bigger, the number of handshakes explodes quadratically.

```
Input Tokens (N=768)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Token Embeddings + Positional Encoding             │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  BART Encoder (12 layers)                           │
│  ┌───────────────────────────────────────────────┐  │
│  │  Each Layer:                                  │  │
│  │                                               │  │
│  │  Input ──► Self-Attention ──► FFN ──► Output │  │
│  │            (O(n²) complexity)                  │  │
│  │                                               │  │
│  │  Attention(Q,K,V) = softmax(QK^T/√d)V         │  │
│  │  Every token attends to ALL other tokens    │  │
│  │  Memory: O(n²), Compute: O(n²)              │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Classification Head                                │
│  └─► Pool position 0 ──► Linear ──► Softmax       │
│      (HF BART style; exp 1–4 via shared/patched)   │
└─────────────────────────────────────────────────────┘
       │
       ▼
   [Logits]
```

---

### Experiment 1: DeepSeek-Inspired Top-K Routing

**Key Innovation**: Low-rank routing to pick a small key set, then full-dim attention on those keys only.

> **Analogy:** Skim chapter titles (cheap low-rank scores), pick the top chapters once per reading head, then read those chapters in full detail for every sentence in the book.

```
Input ──► Q, K, V projections
              │
              ▼
    ┌─────────────────────────────┐
    │ Head-shared low-rank routing  │  ◄── O(n × d_low), not O(n²)
    │ Q̄_low = mean_t Q_low per head │
    │ scores = Q̄_low × K_low^T     │
    │ topk → [BH, K] (same K keys   │
    │         for all query pos.)   │
    └─────────────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Gather K, V         │  ◄── K_eff = min(top_k, max(32, n/8))
    │ (no T×T expand)       │      default top_k=64
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Precise Attention   │  ──► softmax(Q × K_sel^T) V
    │ O(n × K × d)        │
    └─────────────────────┘
```

**Complexity**: O(n × d_low + n × K × d) ≈ O(n) when K ≪ n. Short sequences fall back to dense attention.

---

### Experiment 2: Lightning Hybrid Attention

**Key Innovation**: Band-local softmax (window only) + optional linear global path on long sequences.

> **Analogy:** Read the newspaper closely in a sliding window around each paragraph; on long pages, also keep a running “front page summary” via linear attention and add it in.

```
Input ──► Q, K, V
              │
              ▼
    ┌─────────────────────────────┐
    │ Local: unfold window W      │  ◄── scores [BH, T, W] only
    │ |i-j| ≤ block_size/2       │      no full T×T matrix
    │ softmax → local_out         │      O(n × W × d)
    └─────────────────────────────┘
              │
              ▼ (if T > 4 × block_size)
    ┌─────────────────────────────┐
    │ Global: linear attention    │
    │ Q' = elu(Q)+1, K' = elu(K)+1│
    │ out = Q'(K'^T V) / norm     │      O(n × d²)
    └─────────────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ local_out + 0.5 ×   │  (linear path skipped on short T)
    │ global_out          │
    └─────────────────────┘
```

**Complexity**: O(n×W + n×d²) when linear path is on; O(n×W) only on short sequences. Default `block_size=128`.

---

### Experiment 3: Content-Aware Dynamic Globals

**Key Innovation**: Learned gating network selects dynamic global tokens + sliding window.

> **Analogy:** In a meeting, some people are topic experts who everyone needs to hear from — their words are "globally" important regardless of where they sit. This experiment trains a small neural network to identify which tokens in a sentence are those "VIP speakers", then makes everyone pay attention to those VIPs plus their immediate neighbours. The VIPs change dynamically per sentence — unlike Big Bird which hardcodes fixed positions as global.

```
Input ──► hidden_states
              │
              ▼
    ┌─────────────────────┐
    │ Global Gate         │  ◄── Learned linear layer
    │ Linear(embed_dim→1) │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Globalness Scores   │  ──► Per-token importance scores
    │ [B, T]              │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Top-G Selection     │  ──► Select G=16 most "global" tokens
    │ torch.topk(scores,G)│
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────────────────┐
    │ Top-G globals [B, G]          │  ◄── once per batch (shared by
    │ + sliding window via unfold   │      all query positions)
    │ K_g, K_win → concat scores    │
    │ softmax over G+W keys         │
    └─────────────────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Output              │  ──► einsum over V_g and V_win
    │ O(n × (G+W) × d)    │
    └─────────────────────┘
```

**Complexity**: O(n × (G + W) × d) = O(n) when G, W are constant (defaults G=16, W=64).

---

### Experiment 4: Permuted Block-Sparse Attention (PBS)

**Key Innovation**: Block-level selection for GPU memory coalescing.

> **Analogy:** Instead of searching a library book-by-book, you first scan the shelf labels to find the most relevant shelves, then pull every book off those shelves. This is both faster to search (fewer comparisons at the shelf level) and physically efficient — grabbing a whole shelf at once is much faster than random scattered picks. On a GPU, reading memory in contiguous chunks like this dramatically speeds up computation compared to jumping around randomly.

```
Input ──► Q, K, V [shape: BH, T, d]
              │
              ▼
    ┌─────────────────────┐
    │ Block Pooling       │  ◄── mean-pool Q/K into blocks
    │ block_size = 64     │      (default in run_experiment.py)
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Head-shared routing │  ──► q_pool = mean(Q_blocks)
    │ top-M blocks [BH,M] │      M = num_blocks (default 2)
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Expand + sort idx   │  ──► M×64 token indices, sorted
    │ for coalesced read  │      for GPU locality
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Sparse Attention    │  ──► head-shared softmax on 128 keys
    │ O(n × M × B × d)    │
    └─────────────────────┘
```

**Complexity**: O(n_blocks × B + n × M × B × d) ≈ O(n) for fixed M, B. `M` scales with sequence length via `_effective_num_blocks`.

**Hardware Benefit**: Sorted indices + block-aligned gathers improve memory coalescing vs random per-token sparsity.

---

### Experiment 13: Dynamic Context Window

**Key Innovation**: Fixed-token-budget dropping with chunked early layers for unbounded input length.

> **Analogy:** A journalist covering a city-wide protest can't interview every single person. Instead, they send reporters to every neighborhood (chunks), each reporter takes notes on the most vocal 10 people in their area. Then the editor-in-chief picks the top 50 most important voices city-wide. The final article only quotes those 50 people — regardless of whether 500 or 50,000 people attended.

```
Input (any length N)
       │
       ▼
┌─────────────────────────────┐
│ Embedding + Positions       │  ◄── O(N) — cheap
│ [B, N, D]                   │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│ PATH A: N ≤ chunk_size      │
│   Full early layers 0..L-1  │  ◄── O(N²) attention (dense)
│ PATH B: N > chunk_size      │
│   Split into chunks         │
│   Run early layers per chunk│  ◄── O(num_chunks × chunk²)
│   Concatenate outputs       │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│ Score all tokens globally   │
│ Importance = ||h||₂         │
│ Keep top target_budget      │  ◄── budget is FIXED (e.g. 4096)
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│ Late layers L..end          │
│ Attention over budget tokens│  ◄── O(budget²) always
│ Same cost for 4k or 1M input│
└─────────────────────────────┘
       │
       ▼
   [Classification]
```

**Complexity**: 
- Early: O(num_chunks × chunk_size²) when chunked, O(N²) when short
- Late: O(budget²) — **independent of input length**
- Total: O(N + budget²) for long sequences, dominated by O(budget²)

**Key property**: Input of 4,096 → keep 4,096 tokens. Input of 100,000 → still keep 4,096 tokens. Input of 1,000,000 → still keep 4,096 tokens.

---

## Repository Structure

The project is organized to facilitate fair and consistent benchmarking across different architectures:

*   **shared/**: Contains standardized utility modules used by all experiments.
    *   `dataset.py`: IMDb loading (`stanfordnlp/imdb`); supports dynamic padding or `fixed_length` padding.
    *   `runner.py`: Hugging Face `Trainer` wrapper; logs F1, memory, latency, `seq_stats_train`, `fixed_length`.
    *   `patched_model.py`: BART first-token pooling + `classification_forward` for fair exp 1–4 eval.
    *   `sparse_attn_utils.py`: Head-shared top-k, efficient gathers, dense fallback helpers.
*   **exp_*/**: Individual implementation folders for each experimental architecture, containing specific model definitions and execution scripts. `exp_15_bigger_bird/` contains the proper BigBird-based Bigger Bird implementation (extends `BigBirdBlockSparseAttention` with content-aware top-k MMR, dynamic globals, and teleports).
*   **benchmarks/**: Automated output directory for tracking performance metrics (F1 score), training latency, and memory utilization.

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd utmist-bigger-bird

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Execution Instructions

To execute the experiments, ensure that the environment requirements (PyTorch, Transformers, Accelerate) are met.

### Running experiments (recommended)

Use the unified runner and presets in [COMMANDS.md](COMMANDS.md):

```bash
python run_experiment.py --list
python run_experiment.py --exp 1 --size small              # quick smoke
python run_experiment.py --exp 1 --size big --fixed-length --seq 1024   # long-context
```

Legacy per-folder scripts (`exp_*/run.py`) still exist but may use older defaults.

**Padding:** Without `--fixed-length`, IMDb batches are padded to the longest sequence in the batch (~150–200 tokens typical), not necessarily `max_length`. For long-context experiments, always pass `--fixed-length` (and consider `--grad-checkpoint` for baseline at 2048+).

## Efficiency & Complexity Benchmarks

Beyond accuracy, every training run now automatically captures:
- **Peak memory** (MB)
- **Inference latency** (ms/sequence)
- **Softmax comparison count** (vs full dense baseline) — counts softmax *keys* only; use `complexity_verify.py` for wall-clock attention time

### Long-Context Scaling Test
Test how each method handles longer sequences with the **same compute budget**:
```bash
python benchmarks/efficiency_eval.py --exp 0,1,2,3,4 --seq 256,512,1024,2048
```
This fixes samples=500, epochs=2 and varies sequence length. Results are saved to `benchmarks/efficiency_results.json`.

Visualize with:
```bash
python viz/efficiency_viz.py
```

### Empirical Complexity Verification
Micro-benchmark the attention forward pass to verify O(n) vs O(n²):
```bash
python benchmarks/complexity_verify.py --exp 0,1,2,3,4 --seq 128,256,512,1024,2048
```
Output includes log-log slope per experiment (slope ≈ 1 means O(n), ≈ 2 means O(n²)).

### Compare Experiments
After running experiments, compare across all metrics:
```bash
python viz/compare_experiments.py
```

### Interactive Dashboard
Open `dashboard.html` in any browser for a fully interactive, self-contained benchmark explorer (all data embedded inline):

```bash
# Regenerate from latest JSON/CSV benchmarks
python scripts/build_dashboard.py
open dashboard.html
```

**Tabs:**
- **Overview** — F1 vs seq length, attention forward time, peak memory, softmax comparisons
- **Complexity** — log-log slope verification, linear-scale short/long sequence plots
- **Efficiency** — per-sequence-length bar charts for F1, train time, latency, memory
- **All Runs** — sortable/filterable table of every training run
- **Radar** — 6-axis multi-factor comparison (F1, Accuracy, Speed, Memory, Train Efficiency, Sparsity) with overall score
- **Ranking** — **cross-track overall leaderboard**: computes per-track radar scores on IMDb, LRA, and RULER, then ranks experiments by the mean across all three tracks. Pick any sequence length to see which method dominates at that scale.

## Technical Stack

*   **Frameworks**: Python, PyTorch, Hugging Face Transformers, Accelerate.
*   **Evaluation Metrics**: F1 score (primary), training throughput (steps/sec), memory footprint, inference latency, and softmax comparison reduction.
*   **Baseline**: Experiments are currently validated against the facebook/bart-base architecture as a comparative patch.

## Team

This project is a collaborative effort by the University of Toronto Machine Intelligence Student Team (UTMIST).

*   **Leads**: Derek Chen, Han Fang
*   **Advisors**: Isaac
*   **Developers**: Adrian, Brian, Jerry, Karma, Mohamad, Ryan, Thomas