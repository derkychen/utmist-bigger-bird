"""RULER compute presets and default sweep axes."""

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

DEFAULT_SEQ = {"niah": 4096, "mq_niah": 4096}
DEFAULT_DEPTH = 0.5

DEFAULT_SEQS = {"niah": [2048, 4096, 8192], "mq_niah": [2048, 4096, 8192]}
DEFAULT_DEPTHS = [0.1, 0.5, 0.9]

TRACK = "ruler"
