"""Matched dense vs Bigger Bird prefill-only check on LongCodeQA's 1M bucket.

This is an inference-mode, no-KV-cache single-forward comparison. It scores the
next-token A-D answer distribution, so it is multiple-choice QA, not generated
patch success or decode speed. The tokenizer manifest is checked before any GPU
work, and every selected prompt is warmed for both arms before the timed pass.

  BB_MODEL=nemotron MODAL_PROFILE=maker041006 modal run --detach --env main \
    modal_rd/app.py --module modal_rd.longcodeqa_1m_matched_prefill \
    --args "--start 0 --count 6 --tag lcqa1m_matched_h100_20261101"
"""
import argparse
import array
import hashlib
import json
import subprocess
from pathlib import Path

import modal
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.exp_20_sota_long_context.run import BB_V3, Prompter, build_model, set_arm
from experiments.exp_20_sota_long_context.tasks import load_task
from modal_rd.cpu_embed import install_cpu_embedding
from modal_rd.model_config import MAX_CONTEXT, MODEL_ID, NEMOTRON_ID, model_metadata

ROOT = Path("/vol/results/final_bigger_bird_20261101")
MANIFEST = ROOT / "longcodeqa_1m_manifest.json"
OUT_DIR = ROOT / "longcodeqa_1m_prefill"
ARMS = {"dense": None, "bb_v3": BB_V3}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_hash(ids):
    return hashlib.sha256(array.array("I", ids).tobytes()).hexdigest()


def source_metadata(source_commit):
    files = (
        "modal_rd/longcodeqa_1m_matched_prefill.py",
        "modal_rd/final_code_manifest.py",
        "modal_rd/model_config.py",
        "modal_rd/niah.py",
        "modal_rd/attention.py",
        "kernels/bigger_bird_flash.py",
        "modal_rd/cpu_embed.py",
        "modal_rd/mlp_chunk.py",
        "modal_rd/inplace_residual.py",
        "experiments/exp_20_sota_long_context/run.py",
        "experiments/exp_20_sota_long_context/tasks.py",
    )
    repo = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        commit = None
    return {"git_commit": source_commit or commit,
            "sha256": {name: sha256(repo / name) for name in files}}


def forward_one(model, ids, letter_ids):
    x = torch.tensor([ids], device="cuda", dtype=torch.long)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    try:
        with torch.inference_mode(), sdpa_kernel([
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]):
            start.record()
            out = model(input_ids=x, use_cache=False, logits_to_keep=1)
            end.record()
            torch.cuda.synchronize()
            logits = out.logits[0, -1].float()
            letter_logp = logits.log_softmax(-1)[letter_ids]
            probs = letter_logp.softmax(-1).cpu().tolist()
            pred_i = max(range(len(probs)), key=probs.__getitem__)
            result = {
                "pred": "ABCD"[pred_i],
                "probs_abcd": [round(float(p), 7) for p in probs],
                "prefill_s": round(start.elapsed_time(end) / 1000, 6),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 4),
                "oom": False,
            }
            del out, logits, letter_logp
            return result
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"pred": None, "probs_abcd": None, "prefill_s": None,
                "peak_allocated_gb": None, "oom": True}
    finally:
        del x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="first row in the frozen 1M manifest")
    parser.add_argument("--count", type=int, default=6, help="number of matched prompts")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-commit", default="",
                         help="local repository commit baked into this Modal image")
    args = parser.parse_args()
    if MODEL_ID != NEMOTRON_ID:
        raise ValueError("Run with BB_MODEL=nemotron")
    if args.start < 0 or args.count < 1:
        raise ValueError("start must be non-negative and count positive")
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)
    frozen = json.loads(MANIFEST.read_text())
    current_model = model_metadata()
    if frozen.get("status") != "complete":
        raise ValueError("LongCodeQA 1M tokenizer audit is not complete")
    if frozen.get("model", {}).get("model_id") != MODEL_ID:
        raise ValueError("Checkpoint differs from the tokenizer audit")
    if frozen.get("model", {}).get("resolved_revision") != current_model.get("resolved_revision"):
        raise ValueError("Resolved checkpoint revision differs from the tokenizer audit")
    if frozen.get("model", {}).get("source_sha256") != current_model.get("source_sha256"):
        raise ValueError("Model/tokenizer source hashes differ from the tokenizer audit")
    frozen_rows = frozen["rows"]
    end = min(args.start + args.count, len(frozen_rows))
    if args.start >= end:
        raise ValueError(f"start {args.start} is outside the {len(frozen_rows)}-row manifest")

    from modal_rd.niah import load_tokenizer
    tokenizer = load_tokenizer()
    prompter = Prompter(tokenizer, 1_000_000)
    rows = load_task("lcqa1m")
    if len(rows) != len(frozen_rows):
        raise ValueError(f"Dataset count changed: {len(rows)} vs {len(frozen_rows)}")
    selected = []
    for i in range(args.start, end):
        row = rows[i]
        ids, info = prompter.build(row)
        audit = frozen_rows[i]
        digest = token_hash(ids)
        if row["id"] != audit["id"] or row["gold"] != audit["gold"]:
            raise ValueError(f"row identity/gold mismatch at index {i}")
        if len(ids) != audit["tokens"] or digest != audit["token_sha256_u32_native"]:
            raise ValueError(f"token IDs differ from frozen audit at index {i}")
        if info["truncated"] or audit["truncated"] or len(ids) > MAX_CONTEXT:
            raise ValueError(f"Unexpected truncation/context overflow at index {i}")
        selected.append({"index": i, "row": row, "ids": ids, "info": info,
                         "manifest": audit, "token_sha256_u32_native": digest})

    out_path = OUT_DIR / f"{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(out_path)
    volume = modal.Volume.from_name("bb-vol")
    report = {
        "status": "running", "tag": args.tag, "task": "LongCodeQA official 1M bucket",
        "model": current_model, "frozen_manifest_sha256": sha256(MANIFEST),
        "source": source_metadata(args.source_commit), "range": {"start": args.start, "end_exclusive": end,
            "count": len(selected)},
        "arms": ARMS, "precision": "bfloat16", "attention_backend": "PyTorch SDPA; flash/efficient/math allowed",
        "use_cache": False, "decode_tokens": 0, "scoring": "constrained next-token A-D probability",
        "mlp_chunk": 8192, "inplace_residual": True, "cpu_embedding": True,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "warm": [], "timed": [],
    }
    out_path.write_text(json.dumps(report, indent=2))
    volume.commit()
    model = build_model(mlp_chunk=8192, inplace_residual=True)
    install_cpu_embedding(model)
    letter_ids = torch.tensor(prompter.letter_ids, device="cuda", dtype=torch.long)
    try:
        for phase in ("warm", "timed"):
            for item_no, item in enumerate(selected):
                arm_names = list(ARMS)
                # Alternate order across examples and phases to reduce systematic order bias.
                if (item_no + (phase == "timed")) % 2:
                    arm_names.reverse()
                for arm_name in arm_names:
                    set_arm(model, ARMS[arm_name])
                    try:
                        result = forward_one(model, item["ids"], letter_ids)
                    except Exception as exc:
                        result = {"pred": None, "probs_abcd": None, "prefill_s": None,
                                  "peak_allocated_gb": None, "oom": False,
                                  "error": f"{type(exc).__name__}: {exc}"}
                        report["status"] = "failed"
                        report["error"] = result["error"]
                        report["traceback"] = __import__("traceback").format_exc()
                        report[phase].append({"index": item["index"], "id": item["row"]["id"],
                            "arm": arm_name, "gold": item["row"]["gold"], **result})
                        out_path.write_text(json.dumps(report, indent=2))
                        volume.commit()
                        raise
                    rec = {
                        "index": item["index"], "id": item["row"]["id"], "repo": item["row"].get("repo"),
                        "arm": arm_name, "gold": item["row"]["gold"], "correct": result["pred"] == item["row"]["gold"],
                        "tokens": len(item["ids"]), "token_sha256_u32_native": item["token_sha256_u32_native"],
                        "context_tokens": item["info"]["context_tokens"],
                        "kept_context_tokens": item["info"]["kept_context_tokens"],
                        "truncated": item["info"]["truncated"], **result,
                    }
                    report[phase].append(rec)
                    print(f"{phase} {rec['id']} arm={arm_name} tokens={rec['tokens']} "
                          f"gold={rec['gold']} pred={rec['pred']} prefill={rec['prefill_s']} "
                          f"peak={rec['peak_allocated_gb']} oom={rec['oom']}", flush=True)
                    out_path.write_text(json.dumps(report, indent=2))
                volume.commit()
                if phase == "timed":
                    torch.cuda.empty_cache()
        warm = {(x["index"], x["arm"]): x for x in report["warm"]}
        report["warm_timed_equal"] = all(
            x.get("pred") is not None
            and warm.get((x["index"], x["arm"]), {}).get("pred") is not None
            and warm[(x["index"], x["arm"])].get("pred") == x.get("pred")
            for x in report["timed"]
        )
        report["status"] = "complete"
        volume.commit()
    finally:
        out_path.write_text(json.dumps(report, indent=2))
        volume.commit()
        del model
        torch.cuda.empty_cache()
    print(json.dumps({"status": report["status"], "range": report["range"],
                      "warm_timed_equal": report.get("warm_timed_equal")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
