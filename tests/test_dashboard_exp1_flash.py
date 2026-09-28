"""Dashboard ingestion of scripts/bench_exp1_flash.py (and dense) result files.

Runs under pytest or `python scripts/run_tests_plain.py tests/test_dashboard_exp1_flash.py`.
"""
import importlib.util
import json
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("build_dashboard", ROOT / "scripts" / "build_dashboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _diag():
    return {"scanned_files": 0, "parsed_files": 0, "parse_errors": [], "unrecognized_files": [],
            "ignored_files": [], "source_kind_counts": Counter(), "track_counts": Counter(),
            "model_counts": Counter()}


def _backend(acc=1.0, n=5, mean=21.9, warm=22.0, peak=6.42, eff="flash"):
    return {"status": "ok", "accuracy": acc, "n": n, "warmup_seconds": warm, "mean_seconds": mean,
            "peak_gb": peak, "effective_backend": eff, "predictions": []}


def _parse(rel, payload):
    mod = _mod()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / rel
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload))
        return mod.parse_exp1_flash_bench(path, payload, _diag())


def _cfg(top_k=512):
    return {"top_k": top_k, "low_rank_dim": 128, "quantization": "4bit", "dtype": "bfloat16",
            "gpu": "NVIDIA GeForce RTX 3080"}


def test_headline_flash_row():
    rows = _parse("exp_1_flash_chunk/bench_seq32768_20260927_090457.json",
                  {"seq_len": 32768, "config": _cfg(), "backends": {"flash": _backend()}})
    assert len(rows) == 1
    r = rows[0]
    assert (r["exp_num"], r["track"], r["task"], r["seq_length"], r["depth"]) == (1, "ruler", "niah", 32768, 0.5)
    assert r["model"] == "r1-llama-8b-nf4-rtx3080"
    assert r["accuracy"] == 1.0 and r["n_examples"] == 5 and r["complete"]
    assert r["latency_ms"] == 21900.0
    assert abs(r["peak_memory_mb"] - 6.42 * 1024) < 0.01
    assert r["analysis_eligible"] and r["sparse_validity"] == "sparse_only"
    assert "top-k 512" in r["variant"] and r["timestamp"] == "20260927_090457"


def test_dense_row_is_exp0():
    rows = _parse("exp_1_flash/dense_seq32768_20260927_091159.json",
                  {"seq_len": 32768, "exp": 0, "config": {"quantization": "4bit"},
                   "backends": {"dense": _backend(mean=23.8)}})
    assert [(r["exp_num"], r["analysis_eligible"], r["sparse_validity"]) for r in rows] == [(0, True, "dense_baseline")]


def test_ablations_are_kept_but_excluded_from_analysis():
    cases = {
        "exp_1_flash_dashcfg/bench_seq32768_20260927_100620.json": ({"flash": _backend(acc=0.07)}, 128, "ablation_top_k_128"),
        "exp_1_flash_rt_chunk/bench_seq4096_20260927_090529.json": ({"flash": _backend()}, 512, "ablation_kernel_variant"),
        "exp_1_flash_dashcfg_ab/bench_seq32768_20260927_105338.json": ({"torch": _backend(eff="torch")}, 128, "ablation_torch_path"),
        "exp_1_flash/bench_seq4096_20260926_173344.json": ({"flash": _backend()}, 512, "pre_driver_fix"),
    }
    for rel, (backends, k, validity) in cases.items():
        rows = _parse(rel, {"seq_len": 4096, "config": _cfg(k), "backends": backends})
        assert [(r["sparse_validity"], r["analysis_eligible"]) for r in rows] == [(validity, False)], rel


def test_flash_that_fell_back_counts_as_torch():
    rows = _parse("exp_1_flash_chunk/bench_seq4096_20260927_090203.json",
                  {"seq_len": 4096, "config": _cfg(), "backends": {"flash": _backend(eff="torch")}})
    assert rows[0]["sparse_validity"] == "ablation_torch_path"


def test_oom_backend_is_recorded_as_incomplete():
    rows = _parse("exp_1_flash_chunk/bench_seq131072_20260927_080256.json",
                  {"seq_len": 131072, "config": _cfg(2048), "backends": {"flash": {"status": "oom"}}})
    assert len(rows) == 1 and not rows[0]["complete"]


def test_real_benchmarks_are_loaded():
    mod = _mod()
    diag = _diag()
    runs = [r for r in mod.load_runs(diag) if r["model"] == "r1-llama-8b-nf4-rtx3080"]
    assert runs, "no RTX 3080 exp1 flash rows loaded"
    assert not [u for u in diag["unrecognized_files"] if "exp_1_flash" in u["file"]]
    headline = {(r["exp_num"], r["seq_length"]) for r in runs if r["analysis_eligible"]}
    assert {(1, 131072), (0, 131072), (1, 32768), (0, 32768)} <= headline
