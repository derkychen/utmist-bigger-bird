# Exp 18 Design: Confidence-Gated Multi-Resolution Sparse Attention

**Status:** Design and review only. No Exp 18 implementation has been started.

**Target folder:** `experiments/exp_18_confidence_gated/`

**Model:** `DeepSeek-R1-Distill-Llama-8B`

**Primary objective:** Preserve dense-attention retrieval quality on long contexts while reducing *measured* training and inference cost. The initial target is 128K context on the available H100 40 GB MIG partition.

**Important scope rule:** `exp_0` remains the only dense baseline. Exp 18 must use sparse/linear attention in every runtime path, including short sequences. It must not silently fall back to dense SDPA.

**Implementation status:** The single Exp 18 folder, sparse-only reference path, fused causal linear/local+routed inference kernels, autograd routed-window kernel, registry entries, diagnostics, and CPU correctness tests are now implemented. H100 profiling and RULER smoke validation are queued; no speed or accuracy claim is final until those jobs complete.

---

## 1. Executive summary

The proposed Exp 18 method is **Confidence-Gated Multi-Resolution Sparse Attention**. The implementation combines four ideas:

1. **Exact local attention** over a bounded causal window. This preserves nearby syntax and recency dependencies using a FlashAttention-style window kernel.
2. **Global linear completion** over the remote context. Instead of deleting all remote tokens, a positive feature-map summary gives every remote token an approximate contribution.
3. **Content-based exact global correction** for selected remote tokens. A low-rank router identifies likely high-impact tokens; exact full-dimensional attention is computed only for those tokens.
4. **A confidence gate** that decides whether a query or query block needs the expensive global correction. Confident queries use only local attention plus the global completion. Uncertain queries receive the exact sparse correction.

The important design correction compared with a naive `linear_output + sparse_output` implementation is that the branches are combined in the **unnormalized numerator/denominator domain**. Selected tokens are removed from the approximate completion before their exact contribution is added, so their attention mass is not double-counted.

The intended behavior is:

```text
all queries:
    exact local attention + approximate remote completion

only uncertain queries/query-blocks:
    + exact content-based remote retrieval
```

This is a research hypothesis, not an accuracy theorem. No sparse replacement can guarantee identical dense-model outputs for arbitrary inputs without either computing dense attention or having a proven approximation bound that is not available here. Exp 18 will therefore use explicit empirical accuracy gates, selection-recall measurements, conservative routing, and a strict stop rule rather than claiming an unconditional guarantee.

The proposal is also **not being presented as automatically novel**. Linear-plus-sparse hybrids, adaptive budgets, hierarchical routing, and confidence-based computation all have related published work. The possible contribution is the particular combination of:

- overlap-corrected linear completion;
- confidence-gated skipping of global exact retrieval;
- sparse-only execution on a frozen R1/Llama model;
- training-oriented query-block routing and fused kernels;
- detailed accuracy/latency/selection-recall analysis at 128K.

Prior-art review must be completed before making any novelty claim.

---

## 2. Why we are taking a step back

The current experiments establish an important but uncomfortable result: theoretical sparsity is not enough. The current PyTorch sparse paths are much slower than dense FlashAttention even when their asymptotic interaction count is lower.

### 2.1 Current empirical picture

The following values are representative results already present in the repository. They are not yet a perfectly matched benchmark matrix; 128K sparse runs often used fewer examples because of the Slurm time limit.

| Method | 4K accuracy | 64K accuracy | 128K accuracy | 128K time/memory observation | Interpretation |
|---|---:|---:|---:|---|---|
| Dense `exp_0` | 100% | 100% | 100% | About 34.3 s/example and 20.03 GB in the current scaling result | Strong quality and highly optimized FlashAttention path |
| DeepSeek top-k `exp_1` | 100% | 90% | 80% on 15 examples | The 80% run used about 2,338 s/example and 33.16 GB | Best current retrieval candidate, but implementation overhead dominates |
| GQA sparse `exp_10` | 100% | 90% | 60% on 5 examples | About 2,390 s/example and 33.15 GB in the representative selection | Similar algorithmic behavior, not yet a speed solution |
| Coarse-to-fine `exp_17` | 100% | 93.3% on 15 examples | 60% on 5 examples with `fine_k=2048` | About 2,326 s/example and 33.15 GB in the representative selection | Promising routing idea, but long-context fine budget remains limiting |
| Dynamic context `exp_13` | 100% | 90% | 10% on 30 examples | About 786 s/example and 23.23 GB | Fixed token dropping loses the needle at 128K |
| BiggerBird proper `exp_15` | 100% | 90% | 13.3% on 15 examples | About 783 s/example and 31.60 GB | Block aggregation dilutes rare high-value tokens |

The exact timing values above must be re-measured after Exp 18 exists. The current sparse results use `use_cache=False`, Python loops, advanced-index gathers, and multiple unfused operations. They demonstrate the cost of the current implementation, not a fundamental lower bound for sparse attention.

### 2.2 What the current experiments teach us

The strongest validated findings are:

- Dense R1/Llama attention retrieves the RULER NIAH needle at 100% through 128K.
- Content-based token routing is essential for this retrieval task.
- Fixed patterns, untrained gates, token norms, and block means frequently miss the needle.
- `exp_1` top-k works through 64K with `k=512` and reaches 80% at 128K with `k=2048`, but the current implementation is dramatically slower than dense FlashAttention.
- The 128K top-k behavior has a sharp budget transition: `k=512` and `k=1024` are unreliable, while `k=2048` is materially better.
- `exp_17` shows that coarse routing is useful only when the fine token budget is large enough and the coarse representation does not dilute the needle.
- A method can have lower theoretical comparison counts and still lose badly in wall-clock time if it uses Python loops, random gathers, materialized intermediate tensors, or unoptimized backward kernels.

Exp 18 must therefore improve both dimensions:

```text
quality = content retrieval + background-context preservation
speed  = less work + fused GPU execution + no unnecessary global branch
```

---

## 3. Design goals and non-goals

### 3.1 Goals

1. Remain entirely R1/Llama based.
2. Run on the frozen `DeepSeek-R1-Distill-Llama-8B` model before any learned adaptation is introduced.
3. Preserve dense-like retrieval on RULER NIAH at 4K–128K and across needle depths.
4. Improve on `exp_1` quality at 128K without increasing the global exact-token budget for every query.
5. Reduce actual wall-clock attention time, not merely count fewer theoretical comparisons.
6. Support a training path with gradients through local, linear, and selected sparse attention.
7. Avoid all `[batch * heads, n, n]` score tensors.
8. Keep memory bounded with chunking, streaming summaries, and packed active query blocks.
9. Record enough diagnostics to explain every accuracy loss and every speed gain.
10. Make the method rejectable: if the accuracy or speed gates fail, we stop or redesign instead of calling the result successful.

### 3.2 Non-goals

- Claiming a mathematical guarantee of equal dense-model accuracy.
- Claiming novelty before a current literature and code search.
- Replacing the whole Llama model or pretraining a new model from scratch.
- Adding a silent dense path for short sequences or difficult inputs.
- Optimizing only a synthetic microbenchmark while ignoring end-to-end generation and training.
- Expanding another family of loosely related experiment folders before this one is evaluated.

---

## 4. Proposed attention computation

Let a head receive already projected and RoPE-encoded tensors:

- `Q ∈ R^(Tq × d)`
- `K ∈ R^(Tk × d)`
- `V ∈ R^(Tk × d)`

The existing Llama patching infrastructure already provides these tensors to experiment-specific attention classes. Exp 18 will preserve that interface and support both causal/generative and bidirectional/classification calls.

### 4.1 Partition the context

For a causal query at position `t`, define:

```text
L_t = recent local keys, positions max(0, t - W + 1) ... t
R_t = all valid remote keys before t that are not in L_t
```

`W` is fixed, initially 256 or 512. The local set is exact and causal. It handles recency, syntax, and short-range composition.

For bidirectional classification, `L_t` is a symmetric window and `R_t` contains the remaining valid keys. The first milestone is causal RULER; bidirectional support is required before the experiment is considered complete.

### 4.2 Positive linear feature map for remote completion

Use a low-dimensional positive feature map over a projected query/key representation:

```text
q_phi = phi(Pq(Q))
k_phi = phi(Pk(K))
```

where `d_phi` is small, initially 16, 32, or 64, and `phi` may start with `ELU(x) + 1`. The projection and feature map are configurable so that we can compare:

- the existing first-dimension proxy used by `exp_1`;
- a fixed orthogonal projection;
- a small learned projection trained only after the zero-shot version is understood.

For the remote context, maintain a streaming summary:

```text
S_R = sum over i in R of k_phi[i] outer V[i]
Z_R = sum over i in R of k_phi[i]
```

For a query `q`, the approximate remote numerator and denominator are:

```text
N_lin(q, R) = q_phi outer-product-multiplied by S_R
Z_lin(q, R) = q_phi dot Z_R
```

The exact feature-map computation and its causal prefix state must be implemented as a streaming or tiled kernel. We must not materialize a `[T, d_phi, d]` prefix tensor for 128K unless memory profiling proves it safe.

### 4.3 Exact local attention

Compute exact softmax attention over `L_t`:

```text
s_i = q_t dot k_i
w_i = exp(s_i - max(s))
N_local = sum_i w_i * v_i
Z_local = sum_i w_i
```

This should use the existing window-kernel infrastructure where possible, but the current window helper is inference-oriented. A training-capable implementation and backward parity tests are required.

### 4.4 Exact sparse global correction

For an uncertain query, select a small remote set `S_t ⊂ R_t` using a low-rank content router. Compute the full-dimensional logits and values exactly for only those tokens:

```text
N_exact(q, S_t) = sum over i in S_t exp(q dot k_i) * v_i
Z_exact(q, S_t) = sum over i in S_t exp(q dot k_i)
```

The initial candidate budget should be tested at:

```text
k = 512, 1024, 1536, 2048
```

The 128K `exp_1` results show that `k=2048` may be required for strong retrieval on the current model. The purpose of Exp 18 is to avoid paying that cost for every query, not to assume that a smaller `k` is automatically accurate.

### 4.5 Correct overlap-aware combination

A naive implementation would compute a normalized linear output and a normalized sparse output, then add them. That is not a valid softmax decomposition because the two branches have different normalizers and would double-count selected tokens.

Instead, for a corrected query, subtract the selected tokens' approximate feature contributions from the remote summary:

```text
S_R_minus_S = S_R - sum over i in S_t of k_phi[i] outer V[i]
Z_R_minus_S = Z_R - sum over i in S_t of k_phi[i]
```

Then combine all branches before normalization:

```text
N_hat = N_local + N_exact(q, S_t) + N_lin(q, R_t minus S_t)
Z_hat = Z_local + Z_exact(q, S_t) + Z_lin(q, R_t minus S_t)
output = N_hat / (Z_hat + epsilon)
```

For a confident query that skips global retrieval:

```text
N_hat = N_local + N_lin(q, R_t)
Z_hat = Z_local + Z_lin(q, R_t)
output = N_hat / (Z_hat + epsilon)
```

This construction gives every remote token an approximate contribution while treating selected high-impact tokens with the original full-dimensional softmax. It is the intended final accuracy mechanism of Exp 18.

**MVP implementation note:** The first optimized kernel milestone uses the exact local-plus-routed branch as the output for active heads, while confident heads use the local/linear base branch. This avoids an unstable scale mismatch while the fused unnormalized numerator/denominator correction is implemented and benchmarked. The active-head replacement is a valid sparse-only gate ablation, but it must not be reported as the completed overlap-corrected method until the correction formula is wired into the runtime path.

### 4.6 Confidence gate

The gate must be deterministic and calibrated before it is learned. This is important because the project has already observed that randomly initialized learned routing modules select irrelevant tokens and fail zero-shot retrieval.

Candidate cheap signals include:

1. **Local attention entropy**: whether the exact local distribution is concentrated or diffuse.
2. **Local/linear disagreement**: normalized distance between the local output and the remote-completion output.
3. **Estimated remote mass**: the approximate fraction of attention denominator attributed to remote tokens.
4. **Proxy margin**: a low-rank score margin or concentration statistic, when available without an additional full-dimensional pass.
5. **Remote peakiness**: the remote maximum proxy score minus remote log-sum-exp. This normalizes the extreme-value effect as context length grows and distinguishes a single retrieval peak from diffuse filler.
6. **Layer/depth metadata**: only as a controlled ablation, not as the primary signal.

A first deterministic score can be:

```text
uncertainty =
    a * normalized_local_entropy
  + b * normalized_local_linear_disagreement
  + c * normalized_remote_mass
```

The coefficients and thresholds are calibrated on held-out prompts using dense attention only as an **offline reference**. Dense attention is never called by Exp 18 at runtime. We will also test a parameter-free threshold derived from the distributions of the cheap signals so that the experiment does not depend on dense labels.

The gate has three possible states:

| State | Runtime work | Intended use |
|---|---|---|
| `BASE` | Local exact + remote linear completion | Confident local/diffuse-background queries |
| `COARSE` | Base + cheap candidate-block search | Queries with possible remote dependence |
| `FINE` | Base + exact content-routed sparse correction | Retrieval-sensitive or high-disagreement queries |

The `COARSE` state is optional in the first implementation. It becomes important if a full token scan for every `FINE` query dominates runtime.

### 4.7 Routing granularity and causality

Dynamic per-token execution is attractive conceptually but can be slow on GPUs because it creates irregular shapes and scattered work. The implementation will use two modes:

#### Decode mode

A generated token is one current query. The gate can make a per-head/per-layer decision. If uncertain, the current query performs global sparse retrieval; otherwise it uses the base path.

#### Training/prefill mode

Queries are grouped into causal blocks, initially 128 or 256 tokens. The gate is evaluated per block and active blocks are packed before the exact correction kernel runs. The route for a block must be built from a **causal-safe prefix proxy** and must never depend on future query states.

The first implementation may use a conservative route-sharing strategy for active blocks. It must log whether route sharing reduces token-selection recall. If it does, we will use more than one query prototype per block or a hierarchical index.

No asymptotic claim will be made for per-query routing until the selector cost is separately measured. A sparse value/softmax path is not sufficient if the router still performs a hidden `T × T` scan.

---

## 5. Why accuracy could remain close to dense

### 5.1 It fixes the main failure mode of token dropping

`exp_13` and `exp_15` often fail at 128K because they discard or dilute a small, distinctive needle before later layers can use it. Exp 18 does not force every remote token to zero:

- local tokens are exact;
- remote tokens remain in the linear completion;
- high-impact remote tokens can be restored exactly;
- selected tokens are not double-counted.

This is better aligned with the dense attention equation than simple token eviction.

### 5.2 It preserves the successful content-based retrieval mechanism

The `exp_1` result demonstrates that low-rank content-based routing can find the RULER needle when the budget is large enough. Exp 18 reuses that proven principle only for queries that need exact global retrieval. It does not replace content routing with block means, random globals, hidden-state norms, or an untrained gate.

### 5.3 It retains a global signal even when the router misses

If the exact selector misses the needle, the linear completion may still preserve some information about it. This is not guaranteed to be sufficient for generation, but it is strictly less destructive than assigning the token zero contribution.

### 5.4 It uses a conservative skip policy

The gate should prefer false positives over false negatives during the first accuracy phase:

```text
If uncertain, pay for the sparse correction.
If confidently safe, skip it.
```

Only after measuring false-negative rates can the threshold be made more aggressive.

### 5.5 What is and is not guaranteed

There is no unconditional accuracy guarantee. The realistic guarantees are operational:

- every valid query always receives exact local attention;
- every remote query receives a linear summary contribution;
- uncertain queries receive exact sparse retrieval;
- the model never silently switches to dense attention;
- the experiment is accepted only if its measured accuracy is within a predefined margin of `exp_0`.

For a selected-set approximation, if the exact dense attention assigns omitted mass `epsilon` to tokens outside the selected set and values are bounded, the output error is bounded in proportion to `epsilon` and the value range. In practice, Exp 18 does not know the true omitted mass because routing uses a proxy and the completion is approximate. Therefore we will measure:

- selected needle-token recall;
- dense attention mass covered by selected tokens on calibration examples;
- output cosine error and relative norm error against dense attention;
- per-layer teacher KL where dense teacher traces are available;
- final answer accuracy.

These measurements are more informative than claiming that `k=512` or a confidence threshold guarantees quality.

---

## 6. How Exp 18 differs from related approaches

The field already contains close components. The table below describes the intended distinction, not a guaranteed novelty claim.

| Approach | Main idea | Difference from Exp 18 |
|---|---|---|
| Dense FlashAttention | Exact softmax over all keys with IO-aware tiling | Accuracy reference; quadratic interaction count remains |
| `exp_1` DeepSeek top-k | Low-rank content routing, exact attention over a fixed selected set | Exp 18 adds remote completion and skips exact global work for confident queries |
| `exp_17` coarse-to-fine | Select blocks, then refine tokens inside them | Exp 18 uses coarse routing as an optional accelerator, but retains a global completion and confidence gate; it must explicitly address block dilution |
| BASED | Linear attention plus a sliding local window | Similar base branch, but no exact query-aware global correction for distant needles in the core design |
| LoLA | Linear recurrent state, sliding cache, and sparse cache for difficult-to-memorize items | Primarily an inference memory/cache strategy with self-recall management; Exp 18 targets layer attention during training and prefill as well |
| LISA | Linear attention plus a Lightning Indexer and sparse self-attention, fused with a gate | Very close related work. Exp 18's proposed distinction is compute gating based on confidence and overlap-corrected numerator/denominator completion, with a sparse-only frozen R1/Llama study |
| NSA | Compressed, selected, and sliding-window branches | Exp 18 does not always execute all expensive branches; its exact global branch is conditional and its completion removes selected-token overlap |
| Twilight / Tactic / PSA | Adaptive token budgets, top-p, cumulative mass, or progressive selection | These adapt how many tokens are retained; Exp 18 additionally decides whether the exact global mechanism is needed at all and uses an approximate background completion |
| HISA / PIVOT / other index optimizations | Reduce the cost of finding top-k candidates through hierarchy or query sharing | These are useful optimizations for Exp 18's fine branch. Exp 18's main additional hypothesis is confidence-gated branch skipping |
| SharePrefill / cross-head pattern sharing | Reuse attention patterns across heads | Orthogonal optimization; Exp 18 may use it later, but initially needs to verify head-specific retrieval recall |

### Novelty position

The safe claim is:

> Exp 18 studies a confidence-gated, overlap-corrected local-plus-linear-plus-sparse attention design for sparse-only R1/Llama training and long-context evaluation, with explicit runtime and selection-recall accounting.

The unsafe claim is:

> Nobody has ever combined linear attention, sparse attention, and a gate.

That broader claim is false. Before publication, we must search recent papers, preprints, repositories, and concurrent work for confidence-gated mechanism routing and linear-completion attention.

---

## 7. Speed hypothesis

### 7.1 Work model

For fixed local window `W`, feature dimension `d_phi`, sparse budget `k`, and active correction fraction `rho`, the intended work is approximately:

```text
local exact path:       O(T * W * d)
linear completion:      O(T * d_phi * d)       [streaming/recurrent kernel]
router/index:           depends on implementation; must be measured separately
exact global correction:O(rho * T * k * d)
```

The important term is `rho`, the fraction of query blocks that use the exact global correction. If `rho=1`, Exp 18 may be slower than a good dense FlashAttention kernel. If `rho` is small and the base kernels are fused, it may be substantially faster.

For example, with an aspirational configuration:

```text
W = 256
k = 1024 or 2048
rho = 0.10 to 0.25
```

Exp 18 performs exact sparse work for only 10–25% of query blocks rather than every query. This can reduce the sparse correction work by roughly 4–10x relative to always-on top-k, before accounting for indexing and branch overhead.

These are work estimates, not speed guarantees. GPU performance depends on memory traffic, kernel occupancy, gather locality, launch count, dynamic-shape overhead, and backward efficiency.

### 7.2 Why the current sparse code is slow

The current measurements identify likely bottlenecks:

- Python loops over query chunks;
- advanced-index gathers that create large temporary tensors;
- random memory access for selected K/V rows;
- no fusion of routing, gather, softmax, and value accumulation;
- linear attention kernels that are inference-only or use full head dimension summaries;
- repeated work for queries that do not require long-range retrieval;
- `use_cache=False` in the current generative runner;
- no packed active-query representation for conditional branches.

Exp 18 must not repeat this pattern.

### 7.3 Practical speed targets

Targets are deliberately stated as go/no-go thresholds rather than promises:

| Metric | Minimum first target | Strong result |
|---|---:|---:|
| 4K accuracy | Within 5 percentage points of dense | Matches dense |
| 16K accuracy | Within 5 points of dense | Matches dense |
| 64K accuracy | Within 5 points of dense | ≥95% with adequate sample size |
| 128K accuracy | ≥90% on a standardized evaluation, then tighten | ≥95% and within 5 points of dense |
| 4K end-to-end latency | No more than 1.5× dense | Within 1.1× dense |
| 32K end-to-end latency | Faster than dense | ≥2× dense |
| 64K end-to-end latency | ≥2× dense | ≥4× dense |
| 128K end-to-end latency | ≥3× dense | ≥6× dense |
| Long-context memory | Lower than dense at the same batch/length | Flat or near-flat with length |
| Training throughput | Demonstrable gain at 8K+ | ≥2× dense at long context |
| Active exact-correction fraction | ≤50% | ≤25% |
| Sparse audit | PASS | PASS with no runtime dense path |

If Exp 18 reaches the accuracy target but does not beat dense wall-clock time, it is not an efficiency success. If it is fast but misses needles, it is not a quality success.

---

## 8. Training strategy

### 8.1 Stage 1: frozen-model, deterministic routing

The first version must run with the original R1/Llama weights and no learned gate. This isolates the attention approximation from training artifacts and avoids the failure mode already observed in experiments with randomly initialized routing layers.

Trainable parameters in this stage:

- none for the attention replacement;
- optional classification head only for LRA tests;
- no learned global selector.

The first question is whether the architecture preserves the pretrained model's zero-shot behavior.

### 8.2 Stage 2: lightweight calibration

If deterministic gating is promising, calibrate only:

- confidence thresholds;
- branch temperatures/scales;
- feature-map projection;
- optional gate coefficients.

Calibration data must be held out from the final RULER report. Dense attention may provide offline teacher traces, but those traces must never execute inside Exp 18's runtime path.

### 8.3 Stage 3: distillation and LoRA

Only after the zero-shot sparse path works should we train small adapters. Possible objectives:

```text
language-model loss
+ lambda_output * hidden/output distillation loss
+ lambda_gate * gate-label loss
+ lambda_budget * active-correction penalty
+ lambda_recall * selected-mass/needle-recall penalty
```

The gate target can be generated offline from the dense teacher:

```text
label = 1 if sparse base output differs from dense output beyond tolerance
```

This converts the gate into an error predictor rather than a randomly initialized importance scorer. The gate can still be trained with a straight-through or stop-gradient treatment of top-k indices; gradients flow through selected attention values and all linear/local branches.

### 8.4 Training-specific risks

- Top-k indices are discrete and do not receive ordinary gradients.
- A full prefix linear summary can create large activation memory if implemented naively.
- Dynamic active-row packing can cause compiler recompilations.
- The local and linear kernels must have autograd-capable backward paths.
- The current repository's Triton gather and window helpers have separate inference/training limitations that must be audited before claiming training speed.

---

## 9. Optimization plan

Optimization is part of the experiment design, not a later cleanup task.

### Phase 0 — Freeze the specification

Before coding:

1. Review and approve this document.
2. Confirm the first configuration:
   - `W=256`;
   - `d_phi=32`;
   - `k=512` and `k=2048` ablations;
   - score-margin and remote-peakiness thresholds;
   - block size 128 or 256;
   - deterministic gate;
   - causal RULER first.
3. Define the result schema and diagnostic counters.
4. Complete a current literature/code search and mark overlapping methods.

### Phase 1 — Mathematical reference implementation

Implement small CPU/GPU reference functions for:

- exact local attention;
- linear remote completion;
- exact sparse correction;
- overlap subtraction;
- causal and bidirectional masks;
- confidence feature calculation;
- hard gate and packed active rows.

Reference tests must compare the overlap-aware formula against a direct implementation on tiny tensors. They should test duplicate selected indices, padding, causal boundaries, all-future routed indices, and empty remote sets.

This phase is for correctness, not speed.

### Phase 2 — Sparse-only model integration

Create only one new experiment folder:

```text
experiments/exp_18_confidence_gated/
    __init__.py
    model_llama.py
    attention_core.py
    confidence_gate.py
    linear_completion.py
    global_router.py
    metrics.py
```

The model class should inherit from the shared `LlamaSparseAttention` infrastructure and reuse Q/K/V projection, GQA expansion, and RoPE handling. It must implement:

- causal generative path;
- bidirectional classification path;
- sparse-only behavior for short and long sequences;
- no call to `dense_self_attention`;
- explicit `use_cache` behavior or an explicit documented limitation during the initial milestone.

The dashboard registry and sparse audit should be updated only after the implementation exists and passes unit tests.

### Phase 3 — Accuracy ablations

Run the following controlled variants before optimizing kernels:

| Variant | Local | Linear completion | Exact global correction | Gate |
|---|---|---|---|---|
| A | yes | no | no | no |
| B | no | yes | no | no |
| C | yes | yes | no | no |
| D | yes | yes | yes | always |
| E | yes | yes | yes | deterministic confidence |
| F | yes | yes | yes | learned/calibrated gate |
| G | yes | yes | yes | confidence + hierarchical index |

Use the same prompts and seeds as `exp_0`, `exp_1`, and `exp_17`. Save branch counts and selected-token diagnostics for every run.

### Phase 4 — Kernel optimization

#### 4.1 Local kernel

- Extend the existing window attention path with a training-capable version.
- Fuse online softmax and value accumulation.
- Avoid materializing `[BH, T, W]` if the online kernel can accumulate directly.
- Validate bf16 forward and backward parity against fp32/PyTorch references.

#### 4.2 Linear completion kernel

The existing linear kernel computes a full-head-dimension summary and is inference-oriented. Exp 18 needs:

- low-dimensional `d_phi` support;
- causal prefix/recurrent accumulation;
- remote-minus-selected summary updates;
- chunked state with bounded activation memory;
- backward support for training;
- no Python loop over individual tokens.

A fused kernel should keep the recurrent state in registers/shared memory where practical and store only the tensors required for backward or recomputation.

#### 4.3 Sparse correction kernel

Reuse and extend the existing gather machinery, but do not assume it is sufficient:

- packed active query blocks instead of masked inactive rows;
- sorted or block-local candidate indices when possible;
- fused gather, score, online softmax, and value accumulation;
- int32 indices;
- autograd-capable backward;
- duplicate-index gradient accumulation tests;
- no `[BH, T, k, d]` allocation for inactive queries.

#### 4.4 Router/index optimization

The router is a first-class cost. Measure it separately from the exact sparse branch.

Initial hierarchy:

1. low-rank proxy scores;
2. block or candidate-level pruning;
3. exact token refinement only inside candidates.

If the hierarchical route loses needle selection recall, increase the number of representatives per block or disable hierarchy for the affected head/layer. Do not silently claim O(n) when the selector still scans every query against every key.

#### 4.5 Gate execution

The hard gate must avoid computing both branches and then blending them with `where`. The implementation should:

1. compute cheap base features;
2. compact active query/block indices;
3. launch the exact sparse correction only on active rows;
4. scatter corrected outputs back;
5. use a small set of active-count buckets to reduce dynamic compiler overhead.

If the gate only masks an already-computed sparse branch, it provides no speedup.

#### 4.6 Compilation and memory

Test, in order:

- eager fused kernels;
- `torch.compile` with static shape buckets;
- gradient checkpointing/recomputation for linear states;
- bf16 accumulation with fp32 statistics where needed;
- CUDA graph capture only after shapes stabilize.

Track allocator fragmentation and peak allocated/reserved memory separately.

### Phase 5 — Training benchmark

Compare against dense and `exp_1` at matched:

- model weights;
- batch size or effective batch size;
- sequence length;
- gradient accumulation;
- precision;
- checkpointing;
- optimizer and LoRA configuration.

Measure:

- forward time;
- backward time;
- optimizer time;
- tokens/second;
- examples/second;
- peak allocated and reserved memory;
- active correction fraction;
- router time;
- linear/local/sparse kernel time;
- loss and validation quality.

A faster forward pass that loses all training throughput to backward or packing overhead is not sufficient.

---

## 10. Evaluation plan

### 10.1 Primary quality evaluation

RULER generative NIAH remains the primary benchmark because it directly tests long-context retrieval with the native generative R1/Llama model.

Required lengths:

```text
4096, 8192, 16384, 32768, 65536, 131072
```

Required depths:

```text
0.0, 0.25, 0.5, 0.75, 1.0
```

Required comparison methods:

```text
exp_0 dense
exp_1 top-k k=512
exp_1 top-k k=2048
exp_18 base and gated variants
```

Use identical examples, seeds, tokenizer settings, and prompt construction. Increase the final evaluation to at least 100 examples per major 128K configuration when resources allow. A 3/5 or 12/15 result is useful for triage but not a final accuracy claim.

### 10.2 Secondary quality evaluation

After NIAH:

- RULER `mq_niah`;
- NoLiMa tasks where data and runner are available;
- LRA IMDb text classification;
- language-model loss/perplexity on a held-out text sample;
- short-context behavior at 512–4K;
- generation stability and malformed-answer rate.

LRA listops is not a primary success criterion for a frozen R1 generative model because the existing project evidence shows that the task is not learned reliably by the current classification/generative setup.

### 10.3 Efficiency evaluation

For every run, record:

- wall-clock time and time/example;
- prefill latency;
- decode latency per generated token when cache support exists;
- tokens/second;
- peak allocated/reserved GPU memory;
- attention comparison count;
- local/linear/router/sparse time breakdown;
- gate state histogram;
- `rho`, average and maximum `k`;
- selected-token recall and estimated retained attention mass;
- whether the result is complete or partial;
- source file and configuration provenance.

The dashboard must display incomplete results as incomplete and must not choose a high-accuracy partial result without showing its sample count.

### 10.4 Accuracy and speed go/no-go rules

Exp 18 advances only if all of the following hold on a standardized report:

1. It remains above random baseline by a large margin at every tested length.
2. It is within 5 absolute accuracy points of dense on the primary lengths, or clearly improves on `exp_1` while the remaining gap is explained by sample size.
3. At 128K, it reaches at least 90% on a sufficiently large evaluation before any strong claim is made.
4. It beats the current PyTorch `exp_1` implementation by a large factor and also beats dense FlashAttention at at least one long length after kernel optimization.
5. Its exact global correction fraction is low enough that conditional execution is doing real work.
6. The sparse audit remains PASS.
7. No causal leakage, dense fallback, or invalid representative selection is found.

If the gate has a high false-negative rate, reduce sparsity or redesign the confidence signal. If the gate activates on most queries, stop calling it an efficiency method until the base path or index is improved.

---

## 11. Instrumentation and result logging

Every Exp 18 JSON result should include at least:

```json
{
  "exp": 18,
  "model": "DeepSeek-R1-Distill-Llama-8B",
  "task": "niah",
  "seq_len": 131072,
  "depth": 0.5,
  "window_size": 256,
  "feature_dim": 32,
  "top_k_max": 2048,
  "gate_mode": "deterministic",
  "gate_threshold": 0.0,
  "active_query_fraction": 0.0,
  "base_fraction": 0.0,
  "coarse_fraction": 0.0,
  "fine_fraction": 0.0,
  "selected_token_recall": null,
  "retained_mass": null,
  "time_seconds": 0.0,
  "time_local_seconds": 0.0,
  "time_linear_seconds": 0.0,
  "time_router_seconds": 0.0,
  "time_sparse_seconds": 0.0,
  "peak_memory_gb": 0.0,
  "accuracy": 0.0,
  "n_examples": 0,
  "is_causal": true,
  "use_cache": false,
  "sparse_audit": "pending"
}
```

For calibration runs, add dense-teacher comparison fields but label them as offline diagnostics. Do not conflate teacher-trace availability with runtime behavior.

---

## 12. Main risks and mitigations

### Risk 1: Confidence false negatives

A confident gate may skip the exact branch when a distant needle matters.

**Mitigation:** start conservatively; calibrate on multiple depths; log false-negative cases; use retained-mass and local/linear disagreement signals; require high selection recall before reducing the correction rate.

### Risk 2: Linear attention has poor retrieval fidelity

The project already saw that a simple linear branch alone is insufficient for NIAH.

**Mitigation:** use linear attention only as a background completion, never as the sole retrieval mechanism; use exact sparse correction for uncertain queries; combine in unnormalized space.

### Risk 3: Block aggregation dilutes needles

This caused failures in BiggerBird and early coarse-to-fine variants.

**Mitigation:** do not use a single block mean as the only representative; keep token-level refinement, use multiple representatives or a conservative candidate union, and measure block-level needle recall directly.

### Risk 4: The router remains quadratic

Per-query top-k selection can erase the theoretical speedup.

**Mitigation:** expose router time separately; use shared query/block routing, a bounded candidate bank, or a hierarchical/approximate index; never claim linear complexity without a formula and a profiler result.

### Risk 5: Irregular conditional work is slower than regular dense work

GPU kernels prefer regular shapes, and a sparse branch with many small launches can be worse than FlashAttention.

**Mitigation:** block-level gates, packed active rows, fixed shape buckets, fused kernels, and end-to-end measurements.

### Risk 6: Training backward is the real bottleneck

Inference-only gather/linear kernels do not establish faster training.

**Mitigation:** require forward and backward parity tests, profile backward separately, implement recomputation/checkpointing, and compare full training steps rather than only forward latency.

### Risk 7: Novelty overlap

LISA, LoLA, BASED, adaptive sparse methods, and recent index optimizations are close related work.

**Mitigation:** keep novelty claims narrow; cite related work; position the contribution around a reproducible R1/Llama study and system/accuracy analysis if the mechanism overlap is substantial.

### Risk 8: Small-sample conclusions

Long 128K jobs are expensive and partial results can look persuasive.

**Mitigation:** use small samples only for iteration; require a standardized final sample size and preserve source provenance in the dashboard.

---

## 13. Proposed implementation milestones

The following milestones are intentionally sequential. We should not begin a broad sweep until the preceding milestone passes.

### Milestone A — Review approval

- Confirm the architecture and no-fallback rule.
- Confirm target metrics and acceptance thresholds.
- Confirm whether the first version should support only causal RULER or causal plus bidirectional from the start.

### Milestone B — Tiny-tensor correctness

- Reference math passes causal/bidirectional, mask, duplicate, and overlap tests.
- No dense attention call in the Exp 18 module.
- Formula output matches direct reference within numerical tolerance.

### Milestone C — 4K sparse-only smoke test

- Model loads and generates.
- Exp 18 does not regress catastrophically relative to dense.
- Gate diagnostics and result JSON are emitted.
- Sparse audit passes.

### Milestone D — Accuracy characterization

- Complete ablation matrix through 32K.
- Verify whether the linear completion helps rather than harms `exp_1`.
- Tune confidence thresholds without learned routing.

### Milestone E — 64K/128K quality gate

- Test multiple needle depths.
- Measure selected-token recall and false-negative gate cases.
- Do not optimize for speed until the architecture has a credible accuracy path.

### Milestone F — Kernel optimization

- Implement/fix training-capable local, linear, and sparse kernels.
- Eliminate Python token loops and inactive-branch computation.
- Produce time breakdowns and memory profiles.

### Milestone G — Training benchmark

- Compare dense, `exp_1`, and Exp 18 at matched training settings.
- Report forward, backward, optimizer, total step time, tokens/s, memory, and quality.

### Milestone H — Finalist decision

At this point we decide whether Exp 18 is:

1. a practical finalist for long-context training/inference;
2. an accuracy-preserving but not faster method that needs a kernel redesign;
3. an informative negative result; or
4. a basis for a narrower paper/system contribution.

No new family of experiment folders should be opened before this decision unless Exp 18 is clearly blocked by a fundamental flaw.

---

## 14. Review questions before implementation

Please review these decisions before implementation begins:

1. Is the target method name and scope acceptable, or should the experiment be framed more narrowly as **confidence-gated global correction**?
2. Should the first implementation prioritize causal RULER only, then add bidirectional/classification, or require both paths immediately?
3. Is `W=256`, `d_phi=32`, and `k_max=2048` a reasonable initial configuration?
4. Should the first gate be fully deterministic, with learned/distilled gating postponed until after zero-shot evaluation?
5. Are the proposed speed and accuracy thresholds appropriate for deciding whether Exp 18 is promising?
6. Do we want the final contribution to emphasize a new mechanism, a high-quality R1/Llama systems study, or whichever is supported by the results?

**No Exp 18 code should be written until this design is reviewed and approved.**
