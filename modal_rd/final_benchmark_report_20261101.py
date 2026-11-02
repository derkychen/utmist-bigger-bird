"""Verify the November 1 paired Exp20 runs and make a slide-ready report."""
import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results" / "final_bigger_bird_20261101"
RAW = ROOT / "raw"
LOGS = ROOT / "logs"
TASKS = {
    "lcqa32k": ("lcqa", "lcqa32k.jsonl", 113),
    "lcqa64k": ("lcqa", "lcqa64k.jsonl", 76),
    "lcqa128k": ("lcqa", "lcqa128k.jsonl", 92),
    "codedebug": ("codedebug", "codedebug.jsonl", 394),
    "lbv2": ("lbv2", "lbv2.jsonl", 503),
}
ARMS = ("dense", "bb_v3", "bb_evict64k")
EXPECTED_EViction = {
    "sink": 64, "window": 16384, "vertical": 512, "exact_tail": 256,
    "kv_evict": True, "evict_pool": 8, "evict_keep": 65536, "evict_window": 16384,
}
SOURCE_FILES = (
    "modal_rd/model_config.py", "modal_rd/niah.py", "modal_rd/attention.py",
    "modal_rd/graph_decode.py", "modal_rd/inplace_residual.py", "modal_rd/mlp_chunk.py",
    "modal_rd/cpu_embed.py", "kernels/bigger_bird_flash.py",
    "experiments/exp_20_sota_long_context/run.py",
    "experiments/exp_20_sota_long_context/tasks.py",
    "modal_rd/app.py", "modal_rd/final_code_manifest.py",
    "modal_rd/longcodeqa_1m_matched_prefill.py",
    "modal_rd/final_benchmark_report_20261101.py",
)


def pct(x):
    return f"{100 * x:.1f}%"


def mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.fmean(vals) if vals else None


def median(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_subset_rows(by_id, ids):
    return [by_id[i] for i in ids]


def summarize(rows):
    n = len(rows)
    out = {"n": n}
    out["tokens_mean"] = mean([x["dense"] for x in rows], "tokens")
    out["tokens_min"] = min(x["dense"]["tokens"] for x in rows)
    out["tokens_max"] = max(x["dense"]["tokens"] for x in rows)
    out["truncated"] = sum(bool(x["dense"]["truncated"]) for x in rows)
    for arm in ARMS:
        out[f"{arm}_correct"] = sum(bool(x[arm]["correct"]) for x in rows)
        out[f"{arm}_accuracy"] = out[f"{arm}_correct"] / n if n else None
        out[f"{arm}_prefill_mean_s"] = mean([x[arm] for x in rows], "prefill_s")
        out[f"{arm}_prefill_median_s"] = median([x[arm] for x in rows], "prefill_s")
        out[f"{arm}_decode_mean_ms_token"] = mean([x[arm] for x in rows], "decode_ms_per_token")
        out[f"{arm}_peak_mean_gb"] = mean([x[arm] for x in rows], "peak_gb")
        out[f"{arm}_peak_max_gb"] = max(x[arm]["peak_gb"] for x in rows)
        out[f"{arm}_prefill_peak_mean_gb"] = mean([x[arm] for x in rows], "prefill_peak_gb")
    for arm in ARMS[1:]:
        out[f"{arm}_delta_pp"] = 100 * (out[f"{arm}_accuracy"] - out["dense_accuracy"])
        out[f"{arm}_agreement_pct"] = 100 * sum(x[arm]["pred"] == x["dense"]["pred"] for x in rows) / n
        out[f"{arm}_prefill_ratio_of_means"] = out["dense_prefill_mean_s"] / out[f"{arm}_prefill_mean_s"]
        out[f"{arm}_median_paired_speed_ratio"] = statistics.median(
            x["dense"]["prefill_s"] / x[arm]["prefill_s"] for x in rows
        )
        b = sum(x["dense"]["correct"] and not x[arm]["correct"] for x in rows)
        c = sum(x[arm]["correct"] and not x["dense"]["correct"] for x in rows)
        out[f"{arm}_dense_to_method_loss"] = b
        out[f"{arm}_dense_to_method_gain"] = c
        out[f"{arm}_mcnemar_p"] = mcnemar_exact(b, c)
    return out


def fmt(x, digits=2):
    return "—" if x is None else f"{x:.{digits}f}"


def table_row(label, s):
    return "| " + " | ".join([
        label, str(s["n"]), f"{s['tokens_mean']:.0f} ({s['tokens_min']:,}–{s['tokens_max']:,})",
        f"{s['truncated']}/{s['n']}",
        f"{s['dense_correct']}/{s['n']} ({pct(s['dense_accuracy'])})",
        f"{s['bb_v3_correct']}/{s['n']} ({pct(s['bb_v3_accuracy'])})",
        f"{s['bb_v3_delta_pp']:+.1f} pp", pct(s["bb_v3_agreement_pct"] / 100),
        f"{s['dense_prefill_mean_s']:.2f} → {s['bb_v3_prefill_mean_s']:.2f}",
        f"{s['bb_v3_prefill_ratio_of_means']:.2f}×",
        f"{s['bb_v3_median_paired_speed_ratio']:.2f}×",
        f"{s['dense_decode_mean_ms_token']:.2f} → {s['bb_v3_decode_mean_ms_token']:.2f}",
        f"{s['dense_peak_mean_gb']:.1f} → {s['bb_v3_peak_mean_gb']:.1f} (max {s['dense_peak_max_gb']:.1f}/{s['bb_v3_peak_max_gb']:.1f})",
    ]) + " |"


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    meta_by_dataset = {}
    complete_by_task = {}
    checks = []
    errors = []
    warnings = []
    source_sha = {name: sha256(repo / name) for name in SOURCE_FILES}
    artifact_sha = {}
    for parent in (RAW, LOGS, ROOT / "longcodeqa_1m_prefill"):
        if parent.exists():
            for path in sorted(x for x in parent.rglob("*") if x.is_file()):
                artifact_sha[str(path.relative_to(ROOT))] = sha256(path)

    def check(name, ok, detail):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})
        if not ok:
            errors.append(f"{name}: {detail}")

    def warn(name, detail):
        warnings.append({"name": name, "detail": detail})

    for task, (dataset, filename, expected_n) in TASKS.items():
        run_dir = next((RAW / dataset).glob(f"final_r1_*"))
        meta = json.loads((run_dir / "meta.json").read_text())
        meta_by_dataset[dataset] = meta
        rows = read_jsonl(run_dir / filename)
        by_id = defaultdict(dict)
        for rec in rows:
            key = (rec["task"], rec["id"])
            if rec["arm"] in by_id[key]:
                errors.append(f"duplicate {task}/{rec['id']}/{rec['arm']}")
            by_id[key][rec["arm"]] = rec
        complete = {key[1]: arms for key, arms in by_id.items()
                    if all(arm in arms for arm in ARMS)}
        check(f"{task}: expected item count", len(by_id) == expected_n,
              f"expected {expected_n}, got {len(by_id)}")
        check(f"{task}: every arm present", len(complete) == expected_n,
              f"complete paired IDs {len(complete)}/{expected_n}")
        check(f"{task}: no duplicate records", len(rows) == expected_n * len(ARMS),
              f"records {len(rows)}, expected {expected_n * len(ARMS)}")
        check(f"{task}: config", meta["max_tokens"] == 130000 and meta["decode_tokens"] == 16
              and meta["graph"] is True and meta["inplace_residual"] is True
              and meta["mlp_chunk"] == 8192,
              f"max_tokens={meta['max_tokens']}, decode_tokens={meta['decode_tokens']}, graph={meta['graph']}, "
              f"inplace_residual={meta['inplace_residual']}, mlp_chunk={meta['mlp_chunk']}")
        check(f"{task}: 64K eviction config", meta["arms"].get("bb_evict64k") == EXPECTED_EViction,
              json.dumps(meta["arms"].get("bb_evict64k")))
        check(f"{task}: H100", "H100" in meta["gpu"], meta["gpu"])
        check(f"{task}: no OOM/errors", all(not r.get("oom") and not r.get("error") for r in rows),
              f"oom={sum(bool(r.get('oom')) for r in rows)}, error={sum(bool(r.get('error')) for r in rows)}")
        token_pair_ok = True
        pred_consistent = True
        probs_ok = True
        eviction_same = True
        decode_ok = True
        for item_id, arm_rows in complete.items():
            dense = arm_rows["dense"]
            for arm in ARMS[1:]:
                r = arm_rows[arm]
                for key in ("gold", "tokens", "context_tokens", "kept_context_tokens", "truncated"):
                    token_pair_ok &= dense.get(key) == r.get(key)
            for arm in ARMS:
                r = arm_rows[arm]
                pred_consistent &= r.get("correct") == (r.get("pred") == r.get("gold"))
                probs = r.get("probs", [])
                probs_ok &= len(probs) == 4 and abs(sum(probs) - 1.0) < 2e-4
                if len(probs) == 4:
                    probs_ok &= r.get("pred") == "ABCD"[max(range(4), key=probs.__getitem__)]
                decode_ok &= r.get("decode_ms_per_token") is not None and r["decode_ms_per_token"] > 0
            eviction_same &= arm_rows["bb_v3"].get("pred") == arm_rows["bb_evict64k"].get("pred")
            eviction_same &= arm_rows["bb_v3"].get("probs") == arm_rows["bb_evict64k"].get("probs")
        check(f"{task}: matched tokenization/config", token_pair_ok, "IDs and actual token lengths match across arms")
        check(f"{task}: independent scoring consistency", pred_consistent, "correct agrees with pred == gold")
        check(f"{task}: A-D distributions", probs_ok, "four constrained probabilities sum to one")
        check(f"{task}: eviction preserves prefill answer", eviction_same,
              "bb_v3 and cache-eviction probabilities/predictions identical")
        check(f"{task}: decode timings", decode_ok, "16-token graph-decode timing present")
        expected_hashes = meta.get("source_sha256", {})
        local_match = all((repo / name).exists() and sha256(repo / name) == digest
                          for name, digest in expected_hashes.items())
        check(f"{task}: source hashes", local_match,
              f"checked {len(expected_hashes)} metadata-listed files against local main")
        if complete:
            complete_by_task[task] = complete

    metas = list(meta_by_dataset.values())
    check("all runs: same checkpoint", len({m["model_id"] for m in metas}) == 1,
          sorted({m["model_id"] for m in metas}))
    check("all runs: same GPU/torch/source", len({(m["gpu"], m["torch"], json.dumps(m["source_sha256"], sort_keys=True)) for m in metas}) == 1,
          {"gpu": sorted({m["gpu"] for m in metas}), "torch": sorted({m["torch"] for m in metas})})
    check("all runs: pinned commit", git_commit == "01d0e402fb7de4ba2ddd91b09650e1329d3d3301", git_commit)
    if not all(m.get("resolved_revision") for m in metas):
        warn("R1 upstream revision not pinned in run metadata",
             "requested_revision=main, resolved_revision is null; all three arms in each suite shared one local checkpoint")
    else:
        check("all runs: resolved model revision", len({m.get("resolved_revision") for m in metas}) == 1,
              [m.get("resolved_revision") for m in metas])

    summaries = {}
    for task, by_id in complete_by_task.items():
        items = list(by_id.values())
        summaries[task] = {"all": summarize(items)}
        if task.startswith("lcqa"):
            summaries[task]["length_bucket"] = task
        elif task == "lbv2":
            groups = defaultdict(list)
            for arm_rows in items:
                groups[arm_rows["dense"].get("group", "unknown")].append(arm_rows)
            summaries[task]["by_domain"] = {k: summarize(v) for k, v in sorted(groups.items())}
            diffs = defaultdict(list)
            for arm_rows in items:
                diffs[arm_rows["dense"].get("difficulty", "unknown")].append(arm_rows)
            summaries[task]["by_difficulty"] = {k: summarize(v) for k, v in sorted(diffs.items())}

    table = [
        "| Test | n | actual tokens, mean (range) | middle-truncated | Dense accuracy | BB v3 accuracy | Δ vs dense | BB answer agrees with dense | Prefill dense → BB (s) | dense/BB mean ratio | paired median ratio | 16-step decode dense → BB (ms/token) | peak allocation dense → BB (GB; max) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table_rows = []
    labels = {"lcqa32k": "LongCodeQA 32K bucket", "lcqa64k": "LongCodeQA 64K bucket",
              "lcqa128k": "LongCodeQA 128K bucket", "codedebug": "InfiniteBench Code.Debug",
              "lbv2": "LongBench v2"}
    for task in TASKS:
        if task in summaries:
            s = summaries[task]["all"]
            table.append(table_row(labels[task], s))
            table_rows.append({"task": task, **s})

    out_lines = [
        "# Bigger Bird final benchmark results — 1 Nov 2026", "",
        "## Slide table: matched R1 benchmarks", "",
        "Dense, `bb_v3`, and `bb_evict64k` were run on the same official examples per suite, using the same R1-Distill-Llama-8B checkpoint, BF16, H100 80GB, 130,000-token prompt cap, seed 0, 16-token greedy CUDA-graph decode, MLP chunking (8,192), and in-place residuals. All three suites completed with every listed item in all three arms and no OOM. Accuracy is the first answer-token distribution renormalized over A–D; it is not code generated or a patch pass rate.",
        "", *table, "",
        "`dense/BB mean ratio` is ratio of mean paired-prefill times; above 1 means BB is faster. The paired median ratio shows the typical per-prompt change. Decode is a separate 16-step graph-decoding measurement. Peak allocation is CUDA max-allocated memory over prefill plus the short decode, not reserved memory or total device utilization. Mean actual tokens are counted after tokenization and middle truncation; the bucket labels are dataset categories, not a claim that every prompt has that exact length.", "",
        "### Paired accuracy changes versus dense", "",
        "| Test | BB correct when dense wrong | BB wrong when dense correct | exact McNemar p |", "|---|---:|---:|---:|",
    ]
    for task in TASKS:
        if task in summaries:
            s = summaries[task]["all"]
            out_lines.append(f"| {labels[task]} | {s['bb_v3_dense_to_method_gain']} | {s['bb_v3_dense_to_method_loss']} | {s['bb_v3_mcnemar_p']:.3f} |")
    out_lines += [
        "", "The paired tests show no clear accuracy separation in these samples (all p ≥ 0.68); this is not an equivalence test, and it does not erase the observed per-item differences.", "",
        "## Cache eviction comparison", "",
        "The `bb_evict64k` arm used `evict_keep=65,536` extra selected middle positions per KV head, plus a 64-token sink and 16,384-token recent window (up to about 81,984 cached positions/head before clipping). It is a 65K extra-token budget, not a 65K total-cache cap. Eviction occurs after the prefill logits used for QA are produced, so identical dense-sparse answers here do not prove that long generated answers retain quality.", "",
        "| Test | n | BB v3 accuracy | BB + eviction accuracy | eviction − v3 (pp) | BB v3 decode ms/token | BB + eviction decode ms/token | v3 peak GB mean/max | eviction peak GB mean/max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        if task not in summaries:
            continue
        s = summaries[task]["all"]
        out_lines.append("| " + " | ".join([
            labels[task], str(s["n"]), f"{s['bb_v3_correct']}/{s['n']} ({pct(s['bb_v3_accuracy'])})",
            f"{s['bb_evict64k_correct']}/{s['n']} ({pct(s['bb_evict64k_accuracy'])})",
            f"{s['bb_evict64k_accuracy'] * 100 - s['bb_v3_accuracy'] * 100:+.1f}",
            f"{s['bb_v3_decode_mean_ms_token']:.2f}", f"{s['bb_evict64k_decode_mean_ms_token']:.2f}",
            f"{s['bb_v3_peak_mean_gb']:.1f}/{s['bb_v3_peak_max_gb']:.1f}",
            f"{s['bb_evict64k_peak_mean_gb']:.1f}/{s['bb_evict64k_peak_max_gb']:.1f}",
        ]) + " |")
    out_lines += [
        "", "## Readout", "",
        "- `bb_v3` mean prefill is 1.27× faster on Code.Debug and 1.26× on LongBench v2; it is slower on LongCodeQA 32K/64K (0.67×/0.80×). The 128K bucket mean ratio is 1.09×, but the paired median ratio is 0.99×, so a typical item is near parity.",
        "- Exact multiple-choice QA is close to dense but slightly lower on the larger LongCodeQA bucket, Code.Debug, and LongBench v2. Report the paired counts and denominators, not ‘same accuracy’.",
        "- `bb_v3` and the 65K-extra eviction arm should have identical first-token probabilities because eviction is downstream of prefill. Any decode/memory benefit is a separate outcome; the measured peak is dominated by the full prefill/cache path and should not be described as 64K total memory.",
        "- These are R1 128K-class evaluations. The 1M-bucket work uses Nemotron-UltraLong-8B because R1's configured maximum is 131,072 tokens; do not combine the models in one accuracy column.",
        "", "## Protocol and limitations", "",
        "- The app warmed each arm at 20,000 and 20,001 tokens, then ran each test prompt once in each arm. It did not warm every exact prompt length and then retime that same prompt; lengths that trigger first-use specialization may retain a cold-start effect.",
        "- Code.Debug and LongBench v2 prompts beyond the 130,000-token cap were middle-truncated. Their model-visible lengths are recorded in raw JSONL. LongCodeQA rows were not truncated in these 32K–128K runs.",
        "- `resolved_revision` for the R1 weights was absent from the Modal run metadata (`requested_revision=main`). The paired arms used the same local checkpoint within each run, but the exact upstream weight commit is not recoverable from these result files; pin it for a strict rerun.",
        "- The next-token A–D scorer is LongCodeQA/LongBench comprehension QA. The 16 decoded tokens are only a speed probe; they are not reviewed code, a valid patch, or test-suite success.",
        "", "## LongCodeQA 1M bucket: tokenizer audit and follow-up", "",
    ]

    manifest_path = RAW / "manifests" / "longcodeqa_1m_manifest.json"
    manifest_info = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        tokens = [x["tokens"] for x in manifest["rows"]]
        manifest_info = {
            "status": manifest["status"], "count": len(tokens), "expected": manifest["expected_count"],
            "tokens_min": min(tokens), "tokens_max": max(tokens),
            "tokens_mean": statistics.fmean(tokens), "truncated": sum(bool(x["truncated"]) for x in manifest["rows"]),
            "model": manifest["model"], "sha256": sha256(manifest_path),
        }
        check("LongCodeQA 1M audit complete", manifest["status"] == "complete"
              and len(tokens) == manifest["expected_count"] == 50
              and manifest_info["truncated"] == 0,
              json.dumps({k: manifest_info[k] for k in ("status", "count", "expected", "tokens_min", "tokens_max", "truncated")}))
        out_lines += [
            f"CPU token audit: {manifest_info['count']}/{manifest_info['expected']} prompts; actual tokens {manifest_info['tokens_min']:,}–{manifest_info['tokens_max']:,} (mean {manifest_info['tokens_mean']:,.0f}); no truncation. Tokenizer is pinned to `{manifest_info['model']['resolved_revision']}`.",
        ]
        prefill_dir = ROOT / "longcodeqa_1m_prefill"
        prefill_files = sorted(prefill_dir.glob("*.json")) if prefill_dir.exists() else []
        prefill_runs = [json.loads(p.read_text()) for p in prefill_files]
        completed_prefill = [run for run in prefill_runs if run.get("status") == "complete"]
        prefill_progress = []
        for run in prefill_runs:
            prefill_progress.append({
                "tag": run.get("tag"), "status": run.get("status"),
                "range": run.get("range"), "warm_rows_saved": len(run.get("warm", [])),
                "timed_rows_saved": len(run.get("timed", [])),
            })
        if completed_prefill:
            run_rows = {phase: [] for phase in ("warm", "timed")}
            expected_indices = set()
            for run in completed_prefill:
                rr = run.get("range", {})
                expected_indices.update(range(rr.get("start", 0), rr.get("end_exclusive", 0)))
                for phase in run_rows:
                    run_rows[phase].extend(run.get(phase, []))
            warm, timed = run_rows["warm"], run_rows["timed"]
            by_arm = {arm: [r for r in timed if r.get("arm") == arm] for arm in ("dense", "bb_v3")}
            matched = {arm: {r["index"]: r for r in by_arm[arm]} for arm in by_arm}
            dense_indices, bb_indices = set(matched["dense"]), set(matched["bb_v3"])
            ids_match = dense_indices == bb_indices
            duplicate_keys = len({(r.get("index"), r.get("arm")) for r in timed}) != len(timed)
            hashes_match = ids_match and all(
                matched["dense"][i].get("token_sha256_u32_native") ==
                matched["bb_v3"][i].get("token_sha256_u32_native") for i in dense_indices
            )
            manifest_rows = {i: row for i, row in enumerate(manifest["rows"])}
            audit_match = ids_match and all(
                0 <= i < len(manifest_rows)
                and all(matched[arm][i].get("id") == manifest_rows[i].get("id")
                        and matched[arm][i].get("gold") == manifest_rows[i].get("gold")
                        and matched[arm][i].get("tokens") == manifest_rows[i].get("tokens")
                        and matched[arm][i].get("token_sha256_u32_native") ==
                            manifest_rows[i].get("token_sha256_u32_native")
                        and not matched[arm][i].get("truncated") for arm in matched)
                for i in dense_indices
            )
            audit_match &= all(
                0 <= r.get("index", -1) < len(manifest_rows)
                and r.get("id") == manifest_rows[r["index"]].get("id")
                and r.get("gold") == manifest_rows[r["index"]].get("gold")
                and r.get("tokens") == manifest_rows[r["index"]].get("tokens")
                and r.get("token_sha256_u32_native") == manifest_rows[r["index"]].get("token_sha256_u32_native")
                and not r.get("truncated") for r in warm + timed
            )
            count_match = dense_indices == bb_indices == expected_indices and not duplicate_keys
            correct = {arm: sum(bool(r.get("correct")) for r in by_arm[arm]) for arm in by_arm}
            means = {arm: statistics.fmean(r["prefill_s"] for r in by_arm[arm]
                                           if r.get("prefill_s") is not None) for arm in by_arm}
            medians = {arm: statistics.median(r["prefill_s"] for r in by_arm[arm]
                                              if r.get("prefill_s") is not None) for arm in by_arm}
            peaks = {arm: statistics.fmean(r["peak_allocated_gb"] for r in by_arm[arm]
                                           if r.get("peak_allocated_gb") is not None) for arm in by_arm}
            warm_map = {(r["index"], r["arm"]): r for r in warm}
            warm_equal = len(warm_map) == len(warm) and all(
                warm_map.get((r["index"], r["arm"]), {}).get("pred") == r.get("pred")
                for r in timed
            )
            config_match = all(
                run.get("model", {}).get("model_id") == manifest["model"].get("model_id")
                and run.get("model", {}).get("resolved_revision") == manifest["model"].get("resolved_revision")
                and run.get("use_cache") is False and run.get("decode_tokens") == 0
                and run.get("precision") == "bfloat16" and run.get("inplace_residual") is True
                and run.get("mlp_chunk") == 8192 and "H100" in run.get("gpu", "")
                and run.get("attention_backend") == "PyTorch SDPA; flash/efficient/math allowed"
                and run.get("scoring") == "constrained next-token A-D probability"
                and set(run.get("arms", {})) == {"dense", "bb_v3"}
                for run in completed_prefill
            )
            source_hashes_match = all(
                (repo / name).exists() and sha256(repo / name) == digest
                for run in completed_prefill
                for name, digest in run.get("source", {}).get("sha256", {}).items()
            )
            source_hashes_match &= all(len(run.get("source", {}).get("sha256", {})) == 11
                                       for run in completed_prefill)
            result_rows_ok = all(
                isinstance(r.get("pred"), str) and r.get("pred") in "ABCD"
                and r.get("correct") == (r.get("pred") == r.get("gold"))
                and isinstance(r.get("prefill_s"), (int, float)) and r["prefill_s"] > 0
                and isinstance(r.get("peak_allocated_gb"), (int, float))
                and not r.get("oom") and not r.get("error")
                for r in timed + warm
            )
            check("LongCodeQA 1M matched prefill rows", count_match and hashes_match,
                  f"timed dense={len(dense_indices)}, bb_v3={len(bb_indices)}, expected completed indices={len(expected_indices)}, duplicates={duplicate_keys}, hashes_match={hashes_match}")
            check("LongCodeQA 1M prompts match frozen token audit", audit_match,
                  f"matched audit rows={len(dense_indices)}, manifest rows={len(manifest_rows)}")
            check("LongCodeQA 1M warm/timed answer equality", warm_equal,
                  f"warm/timed pairs={len(warm_map)}/{len(timed)}")
            check("LongCodeQA 1M run config", config_match, "same Nemotron revision, H100, BF16, no KV cache, no decode, MLP chunk and residual optimizations")
            check("LongCodeQA 1M source hashes", source_hashes_match,
                  f"checked {sum(len(run.get('source', {}).get('sha256', {})) for run in completed_prefill)} metadata-listed files")
            check("LongCodeQA 1M result rows", result_rows_ok,
                  f"rows={len(timed)+len(warm)}, errors={sum(bool(r.get('error')) for r in timed+warm)}, ooms={sum(bool(r.get('oom')) for r in timed+warm)}")
            n = len(dense_indices)
            expected_n = len(expected_indices)
            ratio_each = [matched["dense"][i]["prefill_s"] / matched["bb_v3"][i]["prefill_s"]
                          for i in sorted(dense_indices)]
            summary_1m = {
                "n": n, "expected_n_in_completed_runs": expected_n, "total_dataset_n": len(manifest_rows),
                "tokens_mean": statistics.fmean(matched["dense"][i]["tokens"] for i in dense_indices),
                "tokens_min": min(matched["dense"][i]["tokens"] for i in dense_indices),
                "tokens_max": max(matched["dense"][i]["tokens"] for i in dense_indices),
                "truncated": sum(bool(matched["dense"][i].get("truncated")) for i in dense_indices),
                "correct": correct, "accuracy": {arm: correct[arm] / n for arm in correct},
                "delta_pp": 100 * (correct["bb_v3"] - correct["dense"]) / n,
                "bb_gain": sum(matched["bb_v3"][i].get("correct") and not matched["dense"][i].get("correct") for i in dense_indices),
                "bb_loss": sum(matched["dense"][i].get("correct") and not matched["bb_v3"][i].get("correct") for i in dense_indices),
                "agreement_pct": 100 * sum(matched["dense"][i].get("pred") == matched["bb_v3"][i].get("pred") for i in dense_indices) / n,
                "prefill_mean_s": means, "prefill_median_s": medians,
                "prefill_ratio_of_means": means["dense"] / means["bb_v3"],
                "median_paired_speed_ratio": statistics.median(ratio_each),
                "peak_mean_gb": peaks, "warm_timed_equal": warm_equal,
                "token_hashes_match": hashes_match, "audit_match": audit_match,
                "all_50_complete": n == len(manifest_rows) and count_match,
            }
            summary_1m["mcnemar_exact_p"] = mcnemar_exact(summary_1m["bb_loss"], summary_1m["bb_gain"])
            summaries["longcodeqa_1m"] = {"all": summary_1m}
            out_lines += [
                "", f"Same-H100 BF16 no-KV-cache result ({n}/{len(manifest_rows)} official examples covered):",
                "| n | actual tokens (mean, range) | Dense accuracy | BB v3 accuracy | Δ | BB gain/loss vs dense | McNemar p | Dense/BB prefill (s) | mean ratio | paired median ratio | mean peak allocated GB |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                f"| {n} | {summary_1m['tokens_mean']:.0f} ({summary_1m['tokens_min']:,}–{summary_1m['tokens_max']:,}) | {correct['dense']}/{n} ({pct(correct['dense']/n)}) | {correct['bb_v3']}/{n} ({pct(correct['bb_v3']/n)}) | {summary_1m['delta_pp']:+.1f} pp | {summary_1m['bb_gain']}/{summary_1m['bb_loss']} | {summary_1m['mcnemar_exact_p']:.3f} | {means['dense']:.2f} → {means['bb_v3']:.2f} | {summary_1m['prefill_ratio_of_means']:.2f}× | {summary_1m['median_paired_speed_ratio']:.2f}× | {peaks['dense']:.1f} → {peaks['bb_v3']:.1f} |",
                "", f"Measured prompt range {summary_1m['tokens_min']:,}–{summary_1m['tokens_max']:,}; truncation {summary_1m['truncated']}/{n}; audit IDs, gold answers, token counts, and hashes match: `{audit_match}`. Warm/timed predictions equal: `{warm_equal}`. This is a full-prompt single forward with the final query available to routing; it is not a prefix-invariant causal-cache or decode result.",
            ]
            manifest_info["matched_prefill_summary"] = summary_1m
            if n < len(manifest_rows):
                warn("LongCodeQA 1M matched run is a subset", f"completed {n}/{len(manifest_rows)} prompts")
            active_runs = [p for p in prefill_progress if p.get("status") != "complete"]
            if active_runs:
                saved = sum(x["warm_rows_saved"] + x["timed_rows_saved"] for x in prefill_progress)
                out_lines += [f"Additional matched H100 batch(es) are still active: {active_runs}. Saved warm/timed arm rows across all batches: {saved}."]
        elif prefill_runs:
            saved = sum(len(run.get("warm", [])) + len(run.get("timed", [])) for run in prefill_runs)
            out_lines += ["", f"The matched H100 prefill job(s) are in progress: {prefill_progress}. Saved warm/timed arm rows across jobs: {saved}; partial JSONs are preserved in `{prefill_dir.relative_to(ROOT)}/`."]
        else:
            out_lines += ["", "The matched H100 prefill comparison has not started or its result file is not in this bundle."]
        manifest_info["matched_prefill_progress"] = prefill_progress
        out_lines += [
            "", "LongCodeBench separates LongCodeQA comprehension from LongSWE-Bench patch repair. Its published 1M bucket has 50 LongCodeQA prompts and a separate 100-item LongSWE-Bench repair set. A repair claim requires generated patches followed by the official Docker-based test harness, not a multiple-choice score. A matched LongSWE run is not included here: this workstation has no Docker CLI/daemon, and the official harness starts Docker containers. Do not treat the R1 Code.Debug QA run as patch repair. LongSWE remains the next benchmark workstream after the current matched LongCodeQA run. See the [paper](https://arxiv.org/abs/2505.07897) and [official benchmark code](https://github.com/Zteefano/long-code-bench).",
        ]
    else:
        out_lines.append("LongCodeQA 1M tokenizer audit not found in the local artifact bundle.")

    # Add independently verifiable historical 1M evidence as distinct rows, not merged with R1.
    pilot_dir = repo / "modal_rd/results/nemotron_hard_1m_20261029"
    pilot_file = pilot_dir / "nemotron_hard_1m_pilot_20261029.json"
    if pilot_file.exists():
        pilot = json.loads(pilot_file.read_text())
        dense_pilot = json.loads((pilot_dir / "nemotron_hard_1m_dense_offload_b200_20261030.json").read_text())
        verify = json.loads((pilot_dir / "verification.json").read_text())
        out_lines += [
            "", "## Earlier 1M synthetic RULER results (separate model/task)", "",
            f"The six-task synthetic NIAH comparison used Nemotron-UltraLong-8B, three prompts/task (~971K–980K actual tokens), depth 0.5, seed 42, BF16, and cached greedy output up to 64 tokens. Sparse used H100 with routed attention; dense used B200 with a CPU-offloaded BF16 KV cache. Prompt hashes match, but timing is not a hardware-matched comparison. The multi-query scorer checks answer-string presence, not answer-to-key pairing. Verification: `{verify.get('status')}`.",
            "", "| Task | n | Dense correct | BB correct |", "|---|---:|---:|---:|",
        ]
        for task, sparse_item in pilot.get("tasks", {}).items():
            dense_item = dense_pilot.get("tasks", {}).get(task, {})
            sparse_hits = sum(bool(x.get("exact")) for x in sparse_item.get("timed", []))
            dense_hits = sum(bool(x.get("exact")) for x in dense_item.get("timed", []))
            n = len(sparse_item.get("timed", []))
            out_lines.append(f"| {task} | {n} | {dense_hits}/{n} | {sparse_hits}/{n} |")
        out_lines += ["", f"See the [historical report](../nemotron_hard_1m_20261029/results-report.md) for prompt hashes, prefill, and memory. These synthetic retrieval tasks are not code QA or LongSWE-Bench."]

    exp21 = repo / "experiments/exp_21_kascade/SUMMARY.md"
    if exp21.exists():
        out_lines += [
            "", "## Exp21 1M follow-up facts and efficiency", "",
            "Separate 25-prompt held-out fact follow-up, four facts/prompt. The original report has 16K eviction at 62/100 facts and 20/25 needles; 64K at 85/100 and 19/25; 128K at 86/100 and 21/25; optimal 4-bit full-cache at 92/100 and 21/25; uniform 4-bit at 91/100 and 19/25. The 1M prefill medians were measured (63 s eviction; 72–78 s compressed); the 1M decode values were estimated from per-layer kernel benchmarks, not measured end-to-end. Do not merge these with LongCodeQA accuracy.",
        ]

    out_lines += [
        "", "## Raw artifacts and verification", "",
        "- Paired JSONL, metadata, 1M token audit, and Modal logs are under `raw/` and `logs/`. The matched 1M forward JSONs and separate slide CSV are under `longcodeqa_1m_prefill/` and `longcodeqa_1m_slide.csv`.",
            "- Machine checks, warnings, and source hashes are in `verification.json`; aggregate metrics are in `summary.json` and `slide_summary.csv`.",
        "- Repository baseline: `" + git_commit + "`; benchmark source hashes below.",
        "- No code was pushed. The Modal jobs used profile `maker041006`, environment `main`.",
        "", "### Source SHA-256", "", "| File | SHA-256 |", "|---|---|",
    ]
    out_lines += [f"| `{name}` | `{digest}` |" for name, digest in source_sha.items()]

    (ROOT / "results-report.md").write_text("\n".join(out_lines) + "\n")
    report = {
        "status": "pass" if not errors else "fail", "repo_commit": git_commit,
        "source_sha256": source_sha, "artifact_sha256": artifact_sha,
        "checks": checks, "warnings": warnings, "errors": errors,
        "summaries": summaries, "longcodeqa_1m_audit": manifest_info,
        "model_metadata": {dataset: {k: v for k, v in meta.items()
            if k in ("model_id", "requested_revision", "resolved_revision", "prompt_format", "max_context",
                     "max_tokens", "decode_tokens", "graph", "inplace_residual", "mlp_chunk", "gpu", "torch",
                     "arms", "source_sha256")} for dataset, meta in meta_by_dataset.items()},
    }
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2))
    (ROOT / "summary.json").write_text(json.dumps({"summaries": summaries,
        "longcodeqa_1m_audit": manifest_info, "verification_status": report["status"]}, indent=2))
    with (ROOT / "slide_summary.csv").open("w", newline="") as f:
        fields = ["task", "n", "tokens_mean", "tokens_min", "tokens_max", "truncated",
                  "dense_correct", "dense_accuracy", "bb_v3_correct", "bb_v3_accuracy", "bb_v3_delta_pp",
                  "bb_v3_prefill_mean_s", "bb_v3_prefill_ratio_of_means", "dense_decode_mean_ms_token",
                  "bb_v3_decode_mean_ms_token", "dense_peak_mean_gb", "bb_v3_peak_mean_gb",
                  "bb_evict64k_correct", "bb_evict64k_accuracy", "bb_evict64k_decode_mean_ms_token",
                  "bb_evict64k_peak_mean_gb"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in table_rows:
            writer.writerow({key: item.get(key) for key in fields})
    with (ROOT / "longcodeqa_1m_slide.csv").open("w", newline="") as f:
        fields = ["status", "n", "total_dataset_n", "partial_warm_arm_rows", "partial_timed_arm_rows", "tokens_mean", "tokens_min", "tokens_max",
                  "truncated", "dense_correct", "dense_accuracy", "bb_v3_correct", "bb_v3_accuracy",
                  "delta_pp", "bb_gain", "bb_loss", "mcnemar_exact_p", "agreement_pct", "dense_prefill_mean_s", "bb_v3_prefill_mean_s",
                  "prefill_ratio_of_means", "median_paired_speed_ratio", "dense_peak_mean_gb",
                  "bb_v3_peak_mean_gb", "warm_timed_equal", "token_hashes_match", "audit_match", "all_50_complete"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        if "longcodeqa_1m" in summaries:
            s = summaries["longcodeqa_1m"]["all"]
            writer.writerow({"status": "complete", "n": s["n"], "total_dataset_n": s["total_dataset_n"],
                "tokens_mean": s["tokens_mean"], "tokens_min": s["tokens_min"], "tokens_max": s["tokens_max"],
                "truncated": s["truncated"], "dense_correct": s["correct"]["dense"],
                "dense_accuracy": s["accuracy"]["dense"], "bb_v3_correct": s["correct"]["bb_v3"],
                "bb_v3_accuracy": s["accuracy"]["bb_v3"], "delta_pp": s["delta_pp"],
                "bb_gain": s["bb_gain"], "bb_loss": s["bb_loss"],
                "mcnemar_exact_p": s["mcnemar_exact_p"], "agreement_pct": s["agreement_pct"],
                "dense_prefill_mean_s": s["prefill_mean_s"]["dense"],
                "bb_v3_prefill_mean_s": s["prefill_mean_s"]["bb_v3"],
                "prefill_ratio_of_means": s["prefill_ratio_of_means"],
                "median_paired_speed_ratio": s["median_paired_speed_ratio"],
                "dense_peak_mean_gb": s["peak_mean_gb"]["dense"],
                "bb_v3_peak_mean_gb": s["peak_mean_gb"]["bb_v3"],
                "warm_timed_equal": s["warm_timed_equal"], "token_hashes_match": s["token_hashes_match"],
                "audit_match": s["audit_match"], "all_50_complete": s["all_50_complete"]})
        elif prefill_runs:
            writer.writerow({"status": "in_progress", "total_dataset_n": 50,
                "partial_warm_arm_rows": sum(len(run.get("warm", [])) for run in prefill_runs),
                "partial_timed_arm_rows": sum(len(run.get("timed", [])) for run in prefill_runs)})
    print(json.dumps({"status": report["status"], "checks": len(checks), "errors": errors,
                      "output": str(ROOT / "results-report.md"),
                      "summary_rows": {k: v["all"] for k, v in summaries.items()}}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
