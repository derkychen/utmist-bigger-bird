"""GPU self-check for CUDA-graph decode (modal_rd/graph_decode.py) before using `--graph` in long runs.

For each arm (dense_rd, vs, vs + KV eviction) and a few NIAH prompts, greedy generation through HF `generate`
(eval_niah.run_one) and through the captured graph must return identical token ids. Exit code 1 on any
mismatch.

  python -m modal_rd.graphcheck --bytes 64000 --n 2 --cpu-embed
  modal run modal_rd/app.py --module modal_rd.graphcheck --args "--bytes 512000 --n 2 --cpu-embed"
"""
import argparse
import json
import sys

from modal_rd import attention as rd
from modal_rd.cpu_embed import install_cpu_embedding
from modal_rd.eval_niah import build, run_one
from modal_rd.mlp_chunk import install_chunked_mlp
from modal_rd.niah import load_tokenizer, prepare_task

ATTN = {"sink": 64, "window": 16384, "vertical": 512, "exact_tail": 256}
EVICT = {"kv_evict": True, "evict_pool": 8, "evict_keep": 16384}


def check(model, tokenizer, nbytes, n, max_new, attn=ATTN, evict=EVICT, task="niah_multikey_1"):
    """Returns a list of per-(arm, prompt) results; `same` must be True everywhere."""
    rows = prepare_task(tokenizer, task, nbytes, n)[0]
    out = []
    for arm, cfg in (("dense_rd", attn), ("vs", attn), ("vs", {**attn, **evict})):
        rd.install(model, arm, **cfg)
        for row in rows:
            eager = run_one(model, tokenizer, row, arm, max_new)
            graph = run_one(model, tokenizer, row, arm, max_new, graph=True)
            out.append({"arm": arm, "evict": cfg.get("kv_evict", False), "idx": row["idx"], "tokens": row["tokens"],
                        "same": eager["generated_ids"] == graph["generated_ids"],
                        "exact": [eager["exact"], graph["exact"]], "peak_gb": [eager["peak_gb"], graph["peak_gb"]]})
            print(json.dumps(out[-1]), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bytes", type=int, default=64000)
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--mlp-chunk", type=int, default=8192)
    p.add_argument("--cpu-embed", action="store_true")
    a = p.parse_args()
    tokenizer = load_tokenizer()
    model = build("dense", {}, [])
    if a.mlp_chunk:
        install_chunked_mlp(model, a.mlp_chunk)
    if a.cpu_embed:
        install_cpu_embedding(model)
    ok = all(r["same"] for r in check(model, tokenizer, a.bytes, a.n, a.max_new))
    print("GRAPHCHECK " + ("PASS" if ok else "FAIL"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
