"""Check CUDA-graph decode (`graph_decode.py`) against the eager decode path, and time both.

Per arm, on the same prompt: (1) eager decode through the installed attention modules (eval_decode.Session),
(2) the static step run eagerly, (3) the static step replayed from a CUDA graph. Reports whether the
generated tokens match and ms/token for each.

  BB_MODEL=nemotron python -m experiments.exp_21_kascade.check_graph --bytes 32000 --steps 64
"""
import argparse
import json
import time

import torch

from experiments.exp_21_kascade import model as kascade
from experiments.exp_21_kascade.common import MODEL, OUT, QUANT, load_model, load_tok
from experiments.exp_21_kascade.eval_decode import Session
from experiments.exp_21_kascade.model import EVICT
from experiments.exp_21_kascade.graph_decode import StaticDecoder
from modal_rd.niah import prepare_task


def timed(fn):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main(argv=None, loaded=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sets", default="v3;v3+evict;dp5:remap:pool:decodeonly:dw1024:dv1024")
    p.add_argument("--map-tag", default="ul_p0_16k_tail0_pool")
    p.add_argument("--attn", default='{"sink":64,"window":2048,"vertical":512,"exact_tail":256}')
    p.add_argument("--bytes", type=int, default=32000)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--tag", default="")
    a = p.parse_args(argv)
    attn = json.loads(a.attn)
    tok, model = loaded or (load_tok(MODEL), load_model())
    row = prepare_task(tok, "niah_multikey_1", a.bytes, 1)[0][0]
    ids = tok(row["text"], add_special_tokens=False)["input_ids"]
    report = {"tokens": len(ids), "steps": a.steps, "quant": QUANT, "gpu": torch.cuda.get_device_name(0), "arms": []}
    for spec in a.sets.split(";"):
        kascade.install_spec(model, "v3", {**EVICT, **attn}) if spec == "v3+evict" else \
            kascade.install_spec(model, spec, attn, a.map_tag)
        res = {"spec": spec}
        # (1) eager reference; EOS disabled so every arm runs exactly `steps` decode forwards
        for rep in range(2):                                        # second pass is timed (Triton compiled)
            sess = Session(model, eos=set())
            first, _ = sess.prefill(ids)
            eager, t = timed(lambda: sess.generate(first, a.steps + 1)[0])
        res["eager_ms"] = 1000 * t / a.steps
        # (2) static step, eager; (3) static step, CUDA graph
        for mode in ("static", "graph"):
            sess = Session(model, eos=set())
            first, _ = sess.prefill(ids)
            dec = StaticDecoder(model, capacity=sess.cache.layers[0].keys.shape[-2] + a.steps + 8)
            if mode == "graph":
                dec.load(sess.cache, len(ids), first)
                dec.capture()
            dec.load(sess.cache, len(ids), first)
            _, t = timed(lambda: dec.run(a.steps, graph=mode == "graph"))
            out = [first] + dec.generated()
            res[f"{mode}_ms"] = 1000 * t / a.steps
            res[f"{mode}_match"] = out == eager
            res[f"{mode}_first_diff"] = next((j for j, (x, y) in enumerate(zip(out, eager)) if x != y), None)
            del dec
        report["arms"].append(res)
        print(json.dumps(res), flush=True)
        torch.cuda.empty_cache()
    if a.tag:
        (OUT / "phase4").mkdir(parents=True, exist_ok=True)
        (OUT / "phase4" / f"{a.tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
