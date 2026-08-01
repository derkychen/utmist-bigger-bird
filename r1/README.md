# R1 — DeepSeek-R1-Distill inference & chat

Serves DeepSeek-R1-Distill-Llama-8B via vLLM on the Compute Canada `fir`
cluster (H100 80GB, MIG-partitioned into 40GB slices) and provides an
interactive chat client that streams the model's `思考` reasoning separately
from its final answer.

This is the **inference / chat** half of the R1 integration. The experiment
harness (`run_experiment.py`, `exp_*`) currently uses BART-base as the base
model; the plan is to later rebase the experiments onto R1-Distill-Llama-8B
with LoRA. The serving code here is written so that swapping the base model is
a one-line change (see *Swapping models* below).

## Layout

```
r1/
├── serve.sh        # launch the vLLM OpenAI-compatible server
├── chat.py         # interactive streaming chat client (renders 思考 dim)
├── smoke_test.py   # one-shot test: send a prompt, print the response
└── README.md       # this file
```

## One-time setup (already done on this cluster)

These steps were run once and don't need repeating:

1. **Modules + venv** (in `/scratch/$USER/r1-venv`):
   ```bash
   module load StdEnv/2023 python/3.11.5 cuda/12.6 gcc opencv/4.14.0
   python -m venv /scratch/$USER/r1-venv
   source /scratch/$USER/r1-venv/bin/activate
   pip install --no-deps vllm          # CC build, 0.25.0
   pip install <vllm runtime deps>     # see setup log; opencv skipped (text-only)
   pip install uvloop                  # not pulled in by --no-deps
   ```
   `opencv-python-headless` is intentionally **not** installed — vLLM only
   needs it for vision models, and CC ships a dummy wheel that breaks `pip`.
   Text-only R1 inference works without it.

2. **Model snapshot** (in `/scratch/$USER/models/DeepSeek-R1-Distill-Llama-8B`):
   ```bash
   export HF_HOME=/scratch/$USER/hf-cache
   hf download deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
       --local-dir /scratch/$USER/models/DeepSeek-R1-Distill-Llama-8B
   ```

## Running a chat session

From a login node, allocate a 40GB H100 MIG slice and start the server:

```bash
# 1. Get a GPU (interactive, 3h walltime on gpubase_bygpu_b1)
salloc --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
       --cpus-per-task=8 --mem=64G --time=3:00:00 \
       --account=<your-account> gpubase_bygpu_b1

# 2. On the allocated node, start vLLM (blocks this terminal)
bash r1/serve.sh
```

Then from **another shell on the same GPU node** (e.g. `ssh` into the node
shown by `squeue`, or open a second terminal):

```bash
# 3a. Quick smoke test
source /scratch/$USER/r1-venv/bin/activate
python r1/smoke_test.py

# 3b. Interactive chat
python r1/chat.py
```

Inside the chat session: `/reset` clears history, `/system` changes the
system prompt, `/quit` exits. The model's `思考...` reasoning is rendered
dim cyan and the final answer in the default colour.

## Swapping models

`serve.sh` exposes the model and tensor-parallel size as env vars, so you can
serve a different R1-Distill variant without editing any code:

```bash
# R1-Distill-Llama-70B FP8 on 4x 40GB MIG (flagship, ~70GB)
salloc --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:4 --cpus-per-task=32 --mem=128G --time=3:00:00 --account=<your-account> gpubase_bygpu_b1
MODEL_ID=neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8 \
MODEL_PATH=/scratch/$USER/models/DeepSeek-R1-Distill-Llama-70B-FP8 \
TENSOR_PARALLEL=4 \
bash r1/serve.sh
# then: python r1/chat.py --model r1-distill-llama-70b   (adjust --served-model-name in serve.sh)

# R1-Distill-Qwen-32B BF16 on 1x 40GB MIG (fits with TP? — check; may need 2x)
MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
MODEL_PATH=/scratch/$USER/models/DeepSeek-R1-Distill-Qwen-32B \
TENSOR_PARALLEL=1 \
bash r1/serve.sh
```

When swapping, also update `--served-model-name` in `serve.sh` and `--model`
in `chat.py`/`smoke_test.py` to match.

## Notes / gotchas

- **MIG slices**: `fir`'s H100s are partitioned. Requestable GRES are
  `nvidia_h100_80gb_hbm3_1g.10gb`, `..._2g.20gb`, `..._3g.40gb`. There is no
  full-80GB GRES, so 70B-BF16 (~140GB) is not servable here — use 70B-FP8
  (~70GB) on 4× 40GB slices with TP=4 instead.
- **Triton warning on login node**: vLLM prints "Triton not installed or not
  compatible" on the login node because there's no GPU driver. This resolves
  automatically inside the salloc'd GPU node.
- **Training note**: 70B is servable but **not trainable** on fir's MIG
  slices (no NCCL P2P across MIG). For the experiment phase, use the 8B
  distill + LoRA on 1× 40GB MIG.
