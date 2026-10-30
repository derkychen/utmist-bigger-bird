"""Exp 21 H100 plan in one process (one model load): accuracy, long generation, second question.

Nemotron-UltraLong-1M (BB_MODEL=nemotron) at ~128K tokens (512000 source bytes, Thomas's prompts),
v3 final attention config. Exp 21 = v3 prefill + sparse decode with anchor-layer reuse, head remap
and KV-head pooling (`decodeonly`); anchors and head maps come from the Phase 0 run --map-tag
(default: the local 16K UltraLong run, same checkpoint, uploaded to the volume).

  accuracy  30 single-needle prompts, exp 21 dpK (dense / v3 exist: nemotron_confirm30_20261028);
            niah_multikey_1 n=6: v3 vs exp 21 with 3, 5 and 8 anchors
  long      ~384-token answers: v3, v3+evict, exp 21 with decode window 16384 / 4096 / 1024 (+ budget 1024)
  followup  second question on the same cache, same arms

  python -m experiments.exp_21_kascade.h100_plan --stages accuracy,long,followup
  python -m experiments.exp_21_kascade.h100_plan --smoke     # ~16K, 1-2 prompts, small window: plumbing only
"""
import argparse
import json
import signal

from experiments.exp_21_kascade import eval_decode, eval_reuse
from experiments.exp_21_kascade.common import MODEL, load_model, load_tok

ATTN = {"sink": 64, "window": 16384, "vertical": 512, "exact_tail": 256}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", default="accuracy,long,followup")
    p.add_argument("--map-tag", default="ul_p0_16k_tail0_pool")
    p.add_argument("--anchors", default="dp5", help="anchor spec for the main exp 21 arm")
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
            eval_decode.main(["--mode", stage, "--sets", sets, *common, "--n", str(n),
                              *(["--max-new", "48"] if a.smoke and stage == "long" else []),
                              "--tag", f"nemotron_{stage}_128k{suffix}"], loaded)
        else:
            raise ValueError(stage)


if __name__ == "__main__":
    main()
