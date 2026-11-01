# Default memory configuration for sparse long-context runs

Decision for the merged sparse-prefill memory stack (#17 A → #18 B → #19 C `kv_evict` → #21 D `--kv-int8`),
plus the exact memory savers added in this PR. All numbers: one **A100-SXM4-40GB**, torch 2.13.0, triton 3.7.1,
transformers 5.15.1, R1-Distill-Llama-8B BF16, `modal_rd.eval_niah`, `--warm first`, 2026-10-12.

## Decision

| Piece | Default? | Why |
|---|---|---|
| A (GQA in-kernel, kernel tail) | always on (code path) | exact; sparse peak = dense |
| B `--mlp-chunk 8192` | **on** | exact; now also chunks RMSNorm (this PR) |
| In-place RoPE (this PR) | always on (code path) | bit-identical |
| `--cpu-embed` (this PR) | **on** | bit-identical, −1.05 GB, ~free |
| C `kv_evict` | **on**, `evict_pool=8`, `evict_keep=16384` | ~127K: single NIAH 20/20 (same as no eviction), peak 38.1 → 26.3 GB, decode ~2× faster |
| D `--kv-int8` | **off** (opt-in) | only −1.6 GB once C is on, decode +29%, not lossless |

## Default command

Local (A100/H100, model under `$SCRATCH/models`):

```bash
python -m modal_rd.eval_niah --arm vs \
  --attn '{"sink":64,"window":16384,"vertical":512,"exact_tail":256,"kv_evict":true,"evict_pool":8,"evict_keep":16384}' \
  --mlp-chunk 8192 --cpu-embed --bytes 512000 --n 10 --tag default_127k
```

Modal:

```bash
modal run modal_rd/app.py --module modal_rd.eval_niah --args "--arm vs --attn '{\"sink\":64,\"window\":16384,\"vertical\":512,\"exact_tail\":256,\"kv_evict\":true,\"evict_pool\":8,\"evict_keep\":16384}' --mlp-chunk 8192 --cpu-embed --bytes 512000 --n 10 --tag default_127k"
```

Memory-first variant (only if the default does not fit): add `--kv-int8`.

The attention part is the established best all-sparse config at ~127K (`AGENTS.md`); `evict_window` is not needed
because the prefill window is already 16384.

## Should we use C (`kv_evict`)? Yes

At ~127K tokens (`--bytes 512000`, single NIAH, depths .1 and .5, n=10 each), all with `--mlp-chunk 8192 --cpu-embed`:

| Config | exact | peak GB | decode ms/token |
|---|---:|---:|---:|
| no eviction | 20/20 | 38.11 | 58.3 * |
| eviction, `evict_keep=8192` (the 65K-tuned T1) | **18/20** | 25.17 | 31.6 * |
| **eviction, `evict_keep=16384` (default)** | **20/20** | **26.26** | **29.4** |
| eviction, `evict_keep=32768` (depth .5 only) | 10/10 | 28.44 | 33.8 * |

\* measured before the `--cpu-embed` decode fix in this PR, so ~3–6 ms/token too high; the ratios between these
rows still hold (same overhead in each).

- **The 65K tuning does not carry over to 127K.** T1 keeps a fixed ~24.6K tokens per KV head: 38% of a 65K prompt,
  19% of a 127K one. At 127K it failed 2/20 with the same truncated/off-by-one needles as the original default at 65K
  (`2627354` for `2627353`, `758, 758, …` for `7585133`). Doubling the routed keep to 16384 (~32.8K kept, 26%) fixes
  both, for +1.1 GB.
- **Memory:** −11.9 GB peak at 127K versus no eviction. Without eviction, 127K only just fits on 40 GB.
- **Speed:** decode attends ~32.8K instead of 127K keys per layer, so it is ~2× faster.
- **At 65K** the tuned eviction (T1) matched or beat no-eviction on all five NIAH tasks (n=10–30, #19). The default
  keeps more than T1, so it is at least as conservative there.

**Limits (by design):** eviction keeps the keys the *current* question routes to, so a new question on the same
context needs a fresh prefill; prefill must be a single pass; long chain-of-thought decode was not measured
(answers here are ≤64 tokens).

### Multi-needle tasks at ~125K are a prefill problem, not an eviction problem

| Task (~125K tokens, depth .5, n=10) | eviction (default) | no eviction |
|---|---:|---:|
| niah_multiquery | 0/10 (partial 0.17) | 0/10 (partial 0.17) |
| niah_multikey_3 | 0/10 | not finished |

Multiquery fails identically with and without eviction: the model returns one of the four values and then copies
haystack text. So the sparse prefill config (window 16384, vertical 512) is the limit at this length, not the cache.
The dense baselines and the multikey_3 / multivalue no-eviction runs were stopped for time; finish them before
claiming anything about multi-needle accuracy at 127K:

```bash
python -m modal_rd.eval_niah --arm dense_rd --mlp-chunk 8192 --cpu-embed --task niah_multiquery --bytes 619000 --n 10 --depths 0.5 --max-new 64 --tag dense_mq_125k
```

(`dense_rd` now fits at ~125K on 40 GB thanks to the savers below; `niah_multikey_3` needs `--bytes 386000 --max-new 40`
for ~125K tokens.)

## Should we use D (`--kv-int8`)? Not by default

| | peak GB | decode ms/token | exact | tokens identical to BF16 |
|---|---:|---:|---:|---:|
| ~65K, T1 eviction, BF16 | 21.81 | 30.2 | 30/30 | — |
| ~65K, T1 eviction, INT8 | 20.22 | 40.5 | 30/30 | 25/30 * |
| ~127K, eviction keep 8192, BF16 | 25.17 | 31.6 | 18/20 | — |
| ~127K, eviction keep 8192, INT8 | 23.58 | 40.7 | 18/20 | 17/20 |

\* from #21; this PR's runs reproduce #21's INT8 tokens exactly (30/30).

- Once C is on, the cache is already small (~32.8K tokens per KV head), so INT8 only saves **~1.6 GB**.
- It costs **~30% slower decode** with eviction on (per-layer dequantise launches).
- It is lossy: across #21's 130 examples one answer changed (multivalue, no eviction); here no answer changed.
- So: off by default; turn it on only when the default command does not fit (e.g. a smaller GPU or longer context).
  Without eviction INT8 is worth more (−4 GB at 65K, ~−8 GB of cache at 127K) at BF16 decode speed.

## New exact memory savers (this PR)

Profiling the allocator trace at the 65K T1 + INT8 peak showed ~5.7 GB of non-weight memory, ~4 GB of it temporaries:

| Live at peak (65K, T1 + INT8) | before | after |
|---|---:|---:|
| weights | 16.12 GB | 15.07 GB (embedding table on CPU) |
| INT8 KV cache | 1.59 GB | 1.64 GB |
| RoPE temporaries (`apply_rotary_pos_emb`, `rotate_half`) | 1.60 GB | 0 |
| RMSNorm FP32 temporaries (became the peak after the RoPE fix) | 2.13 GB | 0 |
| **peak allocated** | **21.74 GB** | **20.21 GB** |
| peak reserved (what `nvidia-smi` shows) | 23.87 GB | 21.06 GB |

1. **In-place RoPE** (`modal_rd/attention.py: rope_inplace`): RoPE is applied to Q/K in place, 8192 positions at a
   time. Position-wise, so bit-identical to HF's `apply_rotary_pos_emb`, without its 3–4 full-size copies of Q.
   Always on in `RDAttention` (both `vs` and `dense_rd`); falls back to the HF op when autograd is recording.
2. **Chunked RMSNorm** (`--mlp-chunk` now also wraps `input_layernorm`, `post_attention_layernorm` and the final
   norm): LlamaRMSNorm upcasts the whole `[n, 4096]` input to FP32 twice (~1.1 GB each at 65K). Row-wise, so
   bit-identical. `install_chunked_mlp(model, chunk, norms=False)` keeps the old MLP-only behaviour.
3. **CPU embedding table** (`--cpu-embed`, `modal_rd/cpu_embed.py`): the 1.05 GB `[128256, 4096]` table stays in
   pinned CPU memory; prefill copies n rows, decode one. Bit-identical. Lookup uses `F.embedding`: BF16 advanced
   indexing on CPU took 6 ms per decode step, `F.embedding` takes 0.09 ms (GPU table: 0.04 ms). Prefill pays ~0.13 s
   at 65K (~2%).

**Verification:**
- `pytest tests/test_exact_memory_savers.py tests/test_mlp_chunk.py tests/test_rd_attention_gqa.py tests/test_kv_evict.py tests/test_kv_int8.py`:
  61/61 on CUDA (BF16); the new file passes 10/10 on CPU. The new tests assert `torch.equal` for RoPE (incl. strided K
  and whole-model logits through the sparse arm), all three norms, and logits + `generate` with the CPU table.
- Real model, ~65K, T1 eviction: generated tokens identical to #19's runs on **30/30** (BF16) and to #21's INT8 runs
  on **30/30**; BF16 peak 23.29 → 21.81 GB.
- With all savers, **~127K now fits on 40 GB even without eviction** (38.11 GB, 20/20); previously the 127K config
  needed ~48 GB (Modal H100 numbers).

## Further memory improvements

Done since: item 2 (strided kernel output, PR #24) and items 1 + 3 together as `modal_rd/inplace_residual.py`
(`--inplace-residual` in eval_niah and exp 20): residual adds go in place into the layer input, so the embedding
output is the only residual-stream buffer and HF's extra `inputs_embeds` reference costs nothing. Bit-identical
(tests/test_inplace_residual.py, and on Nemotron-UltraLong NF4 at ~13K tokens for v3, v3 + eviction and exp 21:
identical logits and tokens). On that model it removes ~2 `[n, 4096]` BF16 tensors from the prefill peak
(0.21 GB at 12,959 tokens), i.e. ~2 GB at 127K and ~16 GB at ~1M by linear scaling (not yet measured there). Item 4 (FP8 weights) is being
tried as `modal_rd/fp8.py` (weight-only).

Original list:

What is still live at the 65K peak after this PR, beyond weights and the cache, is ~3.4 GB of `[n, 4096]`-sized
BF16 tensors (0.53 GB each at 65K, ~1.05 GB at 127K):

1. **HF keeps `inputs_embeds` alive for the whole forward** (`LlamaModel.forward` holds the variable while the layer
   loop reassigns `hidden_states`). Exact fix: a patched `LlamaModel.forward` that drops the reference. −0.5 GB at 65K,
   −1 GB at 127K.
2. **Kernel output transpose copy**: `out.transpose(1, 2).reshape(...)` after `bigger_bird_flash` copies the
   `[H, n, d]` output into `[n, H·d]` while the original is alive. Exact fix: let the kernel write `[n, H, d]`
   directly (strided output). −0.5 / −1 GB.
3. **Residual adds** in `LlamaDecoderLayer` allocate a new `[n, 4096]` each time; in-place adds would remove one.
4. **Weights are now 75% of peak (15.1 GB).** Weight-only INT8 (e.g. torchao) would roughly halve them (−7 GB) and
   FP8 on H100 likewise. Lossy, and needs a new dependency plus an accuracy eval; the biggest remaining lever.
5. **Multi-needle at 127K** is limited by the sparse prefill config (above), not by memory.
