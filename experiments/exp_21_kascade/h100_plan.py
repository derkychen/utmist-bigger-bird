"""Exp 21 H100 plan in one process (one model load): accuracy, long generation, second question.

Nemotron-UltraLong-1M (BB_MODEL=nemotron) at ~128K tokens (512000 source bytes, Thomas's prompts),
v3 final attention config. Exp 21 = v3 prefill + sparse decode with anchor-layer reuse, head remap
and KV-head pooling (`decodeonly`); anchors and head maps come from the Phase 0 run --map-tag
(default: the local 16K UltraLong run, same checkpoint, uploaded to the volume).

  accuracy  30 single-needle prompts, exp 21 dpK (dense / v3 exist: nemotron_confirm30_20261028);
            niah_multikey_1 n=6: v3 vs exp 21 with 3, 5 and 8 anchors
  long      ~384-token answers: v3, v3+evict, exp 21 with decode window 16384 / 4096 / 1024 (+ budget 1024)
  followup  second question on the same cache, same arms (question 2 = another multikey needle)
  speed     decode kernel: single-layer microbenchmark at 32K-512K, then the followup_fact test (n=6) with
            v3 first and last (first-arm timing is noisy), v3+evict and exp 21 (window 4096 / 1024 + 1024)
  graph     CUDA-graph decode (no per-token Python / launch overhead), n=6: followup_fact and long with
            v3 first and last, v3+evict, exp 21 (window 4096 / 1024 + 1024), all with --graph
  graph_evict  as graph, arms v3, v3+evict, exp 21 (1024 + 1024), v3; static buffers sized from the kept
            cache, so peak memory is comparable across arms (exp 21 + eviction is not supported: see model.py)
  evict_1m  v3 + eviction at ~979K tokens (Thomas's calibrated niah_multikey_1 size), CPU embeddings, no warm-up:
            followup_fact (needle first, fact second), the control with the fact first, and long answers
  fp8       FP8 weights + activations (modal_rd/fp8.py): BF16 vs FP8 linear-layer timing and last-logit agreement
            on a 16K prompt, then (FP8) Thomas's 30 single-needle prompts with v3 and the graph stage arms
  turbo_1m  v3 with no eviction over a TurboQuant-style compressed full cache (modal_rd/turbo_kv.py, --kv-bits)
            on the evict_1m prompts and tests; evict_1m --evict-keep N tests a larger eviction budget
  offload_1m  exp 21 (decode window 1024 + routed 1024) with the CPU-offloaded KV cache (offload_cache.py) on the
            evict_1m prompts and tests, for a full-cache vs eviction comparison at ~1M; needs BB_MEMORY_MB~200000
  followup_fact  second question about a differently worded fact at 25% depth (n=10): v3, v3+evict,
            exp 21 with decode window 4096 / 1024 (+ budget 1024)

  python -m experiments.exp_21_kascade.h100_plan --stages accuracy,long,followup
  python -m experiments.exp_21_kascade.h100_plan --smoke     # ~16K, 1-2 prompts, small window: plumbing only
"""
import argparse
import json
import signal

from experiments.exp_21_kascade import bench_decode, eval_decode, eval_reuse
from experiments.exp_21_kascade.common import MODEL, OUT, load_model, load_tok

ATTN = {"sink": 64, "window": 16384, "vertical": 512, "exact_tail": 256}


def turbo_selfcheck():
    """Fused compressed-cache decode vs dequantize-then-attend on this GPU; stops the run on a mismatch."""
    import torch
    from kernels.turbo_decode import turbo_decode
    from modal_rd.turbo_kv import TurboQuantizer
    torch.manual_seed(0)
    for bits in (4, 3):
        tq = TurboQuantizer(128, bits, "cuda")
        k, v = torch.randn(8, 20000, 128, device="cuda") * 2, torch.randn(8, 20000, 128, device="cuda")
        (kc, kn), (vc, vn) = tq.encode(k), tq.encode(v)
        q = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        out = turbo_decode(q, kc, kn, vc, vn, tq.c, tq.rot, scale=128 ** -0.5).float()
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.float()[None, :, None], tq.decode(kc, kn, torch.float32)[None], tq.decode(vc, vn, torch.float32)[None],
            scale=128 ** -0.5, enable_gqa=True)[0, :, 0]
        err = (out - ref).abs().max().item()
        print(f"TURBO SELFCHECK bits={bits} max_err={err:.4f}", flush=True)
        if err > 2e-2:
            raise RuntimeError("fused compressed-cache decode disagrees with the reference")


def fp8_checks(loaded, path, nbytes):
    """BF16 vs FP8 linear timing (one token and an 8K chunk, CUDA-graph GPU time) and last-position logits on
    one NIAH prompt; then quantizes the loaded model in place (install_fp8)."""
    import torch
    from experiments.exp_21_kascade.bench_decode import graphed
    from modal_rd import attention as rd
    from modal_rd.fp8 import FP8Linear, install_fp8
    from modal_rd.model_config import tokenize_prompt
    from modal_rd.niah import prepare_task
    tok, model = loaded
    layer = model.model.layers[0]
    base = getattr(layer.self_attn, "_rd_base", layer.self_attn)
    timing = {}
    with torch.inference_mode():
        for name, lin in (("q_proj", base.q_proj), ("k_proj", base.k_proj), ("gate_proj", layer.mlp.gate_proj),
                          ("down_proj", layer.mlp.down_proj)):
            f8 = FP8Linear(lin)
            for m in (1, 8192):         # one token = FP8 GEMV kernel; 8192 = dequantize + BF16 GEMM
                x = torch.randn(1, m, lin.in_features, device="cuda", dtype=torch.bfloat16)
                timing[f"{name}_m{m}"] = {"bf16_ms": graphed(lambda: lin(x), 50), "fp8_ms": graphed(lambda: f8(x), 50)}
            del f8
        row = prepare_task(tok, "niah", int(nbytes), 1)[0][0]
        ids = tokenize_prompt(tok, row["text"], return_tensors="pt")["input_ids"].cuda()
        rd.install(model, "dense_rd", **ATTN)
        ref = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
        install_fp8(model)
        out = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
    kl = torch.nn.functional.kl_div(out.log_softmax(-1), ref.log_softmax(-1), log_target=True, reduction="sum").item()
    report = {"timing": timing, "tokens": ids.shape[1], "top1_same": bool(out.argmax() == ref.argmax()),
              "max_abs_logit_diff": (out - ref).abs().max().item(), "kl_bf16_to_fp8": kl,
              "top5_bf16": ref.topk(5).indices.tolist(), "top5_fp8": out.topk(5).indices.tolist(),
              "weights_gb_after": torch.cuda.memory_allocated() / 1e9}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    print("FP8 CHECKS", json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", default="accuracy,long,followup")
    p.add_argument("--map-tag", default="ul_p0_16k_tail0_pool")
    p.add_argument("--anchors", default="dp5", help="anchor spec for the main exp 21 arm")
    p.add_argument("--decode", default="dw1024:dv1024", help="offload_1m: exp 21 decode flags, e.g. dw1024:dv8192:sp8")
    p.add_argument("--evict-keep", type=int, default=0, help="evict_1m: routed keys kept per KV head (default 16384)")
    p.add_argument("--kv-bits", type=int, default=4, help="turbo_1m: bits per value of the compressed full cache")
    p.add_argument("--fp8", action="store_true", help="FP8 weights (modal_rd/fp8.py) for every stage")
    p.add_argument("--tests", default="followup_fact,followup_fact_first,long", help="1M stages: which tests to run")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-bytes", default="64000", help="smoke prompt size (32000 ~ 8K tokens fits a 10 GB GPU)")
    p.add_argument("--max-minutes", type=float, default=0,
                   help="stop after this many minutes (results so far are kept; 0 = no limit)")
    a = p.parse_args()
    if a.max_minutes:
        def stop(*_):
            raise TimeoutError(f"--max-minutes {a.max_minutes} reached")
        signal.signal(signal.SIGALRM, stop)
        signal.alarm(int(a.max_minutes * 60))
    attn, nbytes, n, suffix = dict(ATTN), "512000", 6, ""
    if a.smoke:  # window small enough that the sparse paths run on ~16K prompts
        attn.update(window=2048)
        nbytes, n, suffix = a.smoke_bytes, 1, "_smoke"
    loaded = (load_tok(MODEL), load_model())
    if a.fp8:
        from modal_rd.fp8 import install_fp8
        install_fp8(loaded[1])
    exp21 = f"{a.anchors}:remap:pool:decodeonly"
    common = ["--map-tag", a.map_tag, "--attn", json.dumps(attn), "--bytes", nbytes]
    for stage in a.stages.split(","):
        print(f"##### {stage}", flush=True)
        if stage == "accuracy":
            # prompts 0-5 are Thomas's tuning prompts; report 6-29 separately
            eval_reuse.main(["--task", "niah", "--sets", "v3;" + exp21 if a.smoke else exp21, *common, "--start", "0",
                             "--n", str(1 if a.smoke else 30), "--max-new", "64", "--tag", "nemotron_niah_128k" + suffix],
                            loaded)
            sweep = ";".join(f"dp{k}:remap:pool:decodeonly" for k in (3, 5, 8))
            eval_reuse.main(["--task", "niah_multikey_1", "--sets", "v3;" + sweep, *common, "--start", "0",
                             "--n", str(n), "--max-new", "64", "--tag", "nemotron_multikey_128k" + suffix], loaded)
        elif stage in ("long", "followup"):
            sets = f"v3;v3+evict;{exp21};{exp21}:dw4096;{exp21}:dw1024:dv1024"
            if a.smoke:
                sets = f"v3;v3+evict;{exp21};{exp21}:dw1024:dv1024"
            eval_decode.main(["--mode", stage, *(["--q2", "needle"] if stage == "followup" else []), "--sets", sets, *common, "--n", str(n),
                              *(["--max-new", "48"] if a.smoke and stage == "long" else []),
                              "--tag", f"nemotron_{stage}_128k{suffix}"], loaded)
        elif stage == "speed":
            (OUT / "phase4").mkdir(parents=True, exist_ok=True)
            for w, r in ((1024, 1024), (4096, 512), (16384, 512)):
                bench_decode.main(["--n", "32768" if a.smoke else "32768,131072,262144,524288", "--window", str(w),
                                   "--routed", str(r), "--out", str(OUT / "phase4" / f"bench_decode_h100_w{w}_r{r}{suffix}.json")])
            sets = f"v3;v3+evict;{exp21}:dw4096;{exp21}:dw1024:dv1024;v3"
            eval_decode.main(["--mode", "followup", "--q2", "fact", "--sets", sets, *common, "--n", str(n),
                              "--tag", f"nemotron_speed_kernel_128k{suffix}"], loaded)
        elif stage == "graph":
            sets = f"v3;v3+evict;{exp21}:dw4096;{exp21}:dw1024:dv1024;v3"
            for mode in ("followup", "long"):
                eval_decode.main(["--mode", mode, "--graph", *(["--q2", "fact"] if mode == "followup" else []),
                                  "--sets", sets, *common, "--n", str(n),
                                  *(["--max-new", "48"] if a.smoke and mode == "long" else []),
                                  "--tag", f"nemotron_graph_{mode}_128k{suffix}"], loaded)
        elif stage == "graph_evict":
            e = f"{exp21}:dw1024:dv1024"
            for mode in ("followup", "long"):
                eval_decode.main(["--mode", mode, "--graph", *(["--q2", "fact"] if mode == "followup" else []),
                                  "--sets", f"v3;v3+evict;{e};v3", *common, "--n", str(n),
                                  *(["--max-new", "48"] if a.smoke and mode == "long" else []),
                                  "--tag", f"nemotron_graph_evict_{mode}_128k{suffix}"], loaded)
        elif stage in ("evict_1m", "offload_1m", "turbo_1m"):
            import modal_rd.hard_1m_pilot  # noqa: F401  (installs the fast byte decoder for ~4.8 MB prompts)
            from modal_rd.cpu_embed import install_cpu_embedding
            install_cpu_embedding(loaded[1])
            attn_1m = {**ATTN, **({"evict_keep": a.evict_keep} if a.evict_keep else {})}
            big = ["--attn", json.dumps(attn_1m), "--bytes", a.smoke_bytes if a.smoke else "4848732", "--warm", "0"]
            k = 1 if a.smoke else 5
            if stage == "turbo_1m":
                turbo_selfcheck()
            for extra, tag in ((["--mode", "followup", "--q2", "fact"], "followup_fact"),
                               (["--mode", "followup", "--q2", "fact", "--fact-first"], "followup_fact_first"),
                               (["--mode", "long", "--max-new", "48" if a.smoke else "384"], "long")):
                if tag not in a.tests.split(","):
                    continue
                if stage == "evict_1m":
                    arm, variant = ["--sets", "v3+evict"], f"_keep{a.evict_keep}" if a.evict_keep else ""
                elif stage == "turbo_1m":
                    arm, variant = ["--sets", "v3", "--kv-bits", str(a.kv_bits)], f"_{a.kv_bits}bit"
                else:
                    arm = ["--sets", f"{exp21}:{a.decode}", "--map-tag", a.map_tag, "--cpu-cache"]
                    variant = "" if a.decode == "dw1024:dv1024" else "_" + a.decode.replace(":", "_")
                variant += "_fp8" if a.fp8 else ""
                eval_decode.main([*extra, *arm, *big, "--n", str(k if tag != "long" else max(1, k - 1)),
                                  "--tag", f"nemotron_{stage[:-3]}_1m{variant}_{tag}{suffix}"], loaded)
        elif stage == "fp8":
            fp8_checks(loaded, OUT / "phase4" / f"fp8_checks{suffix}.json", "16000" if a.smoke else "64000")
            eval_reuse.main(["--task", "niah", "--sets", "v3", *common, "--start", "0", "--n", str(1 if a.smoke else 30),
                             "--max-new", "64", "--tag", f"nemotron_fp8_niah_128k{suffix}"], loaded)
            e = f"{exp21}:dw1024:dv1024"
            for mode in ("followup", "long"):
                eval_decode.main(["--mode", mode, "--graph", *(["--q2", "fact"] if mode == "followup" else []),
                                  "--sets", f"v3;v3+evict;{e};v3", *common, "--n", str(n),
                                  *(["--max-new", "48"] if a.smoke and mode == "long" else []),
                                  "--tag", f"nemotron_fp8_graph_{mode}_128k{suffix}"], loaded)
        elif stage == "followup_fact":
            sets = f"v3;v3+evict;{exp21}:dw4096;{exp21}:dw1024:dv1024"
            eval_decode.main(["--mode", "followup", "--q2", "fact", "--sets", sets, *common,
                              "--n", str(n if a.smoke else 10), "--tag", f"nemotron_followup_fact_128k{suffix}"], loaded)
        else:
            raise ValueError(stage)


if __name__ == "__main__":
    main()
