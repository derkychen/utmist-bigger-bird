"""RULER compute presets and default sweep axes (full 13-task suite)."""

RULER_COMPUTE = {
    "ruler-smoke": {
        "train_samples": 256,
        "eval_samples": 128,
        "batch_size": 8,
        "grad_accum": 1,
        "epochs": 2,
        "desc": "Quick pipeline sanity check",
    },
    "ruler-oom": {
        "train_samples": 32,
        "eval_samples": 16,
        "batch_size": 1,
        "grad_accum": 1,
        "epochs": 1,
        "desc": "Tiny budget for OOM/survival probes at long context",
    },
    "ruler-report": {
        "train_samples": 1000,
        "eval_samples": 200,
        "batch_size": 4,
        "grad_accum": 2,
        "epochs": 4,
        "desc": "Moderate budget for depth × context retention sweeps",
    },
    "ruler-full": {
        "train_samples": 5000,
        "eval_samples": 500,
        "batch_size": 8,
        "grad_accum": 2,
        "epochs": 8,
        "desc": "Full synthetic training budget",
    },
}

# Official NVIDIA RULER synthetic.yaml task names (+ legacy aliases).
ALL_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]

_ALIAS_TASKS = ["niah", "mq_niah"]
_DEPTH_TASKS = set(ALL_TASKS + _ALIAS_TASKS) - {"cwe", "fwe"}

_DEFAULT_SEQ_LEN = 4096
DEFAULT_SEQ = {t: _DEFAULT_SEQ_LEN for t in ALL_TASKS + _ALIAS_TASKS}
DEFAULT_DEPTH = 0.5

DEFAULT_SEQS = {t: [2048, 4096, 8192] for t in ALL_TASKS + _ALIAS_TASKS}
DEFAULT_DEPTHS = [0.1, 0.5, 0.9]

TRACK = "ruler"


def task_uses_depth(task: str) -> bool:
    return task in _DEPTH_TASKS
