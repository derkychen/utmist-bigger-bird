"""NoLiMa compute presets and default sweep axes."""

NOLIMA_COMPUTE = {
    "nolima-smoke": {
        "train_samples": 256,
        "eval_samples": 128,
        "batch_size": 8,
        "grad_accum": 1,
        "epochs": 2,
        "desc": "Quick pipeline sanity check",
    },
    "nolima-oom": {
        "train_samples": 32,
        "eval_samples": 16,
        "batch_size": 1,
        "grad_accum": 1,
        "epochs": 1,
        "desc": "Tiny budget for OOM/survival probes at long context",
    },
    "nolima-report": {
        "train_samples": 1000,
        "eval_samples": 200,
        "batch_size": 4,
        "grad_accum": 2,
        "epochs": 4,
        "desc": "Moderate budget for depth × context retention sweeps",
    },
    "nolima-full": {
        "train_samples": 5000,
        "eval_samples": 500,
        "batch_size": 8,
        "grad_accum": 2,
        "epochs": 8,
        "desc": "Full synthetic training budget",
    },
}

ALL_TASKS = ["onehop", "twohop", "all", "hard", "direct", "mc", "distractor"]
CORE_TASKS = ["onehop", "twohop", "all", "hard"]

_DEFAULT_SEQ_LEN = 4096
DEFAULT_SEQ = {t: _DEFAULT_SEQ_LEN for t in ALL_TASKS}
DEFAULT_DEPTH = 0.5

DEFAULT_SEQS = {t: [2048, 4096, 8192] for t in ALL_TASKS}
DEFAULT_DEPTHS = [0.1, 0.5, 0.9]

TRACK = "nolima"


def task_uses_depth(task: str) -> bool:
    return task in ALL_TASKS
