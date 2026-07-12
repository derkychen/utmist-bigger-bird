"""LRA compute presets and default sequence lengths."""

LRA_COMPUTE = {
    "lra-smoke": {
        "train_samples": 256,
        "eval_samples": 128,
        "batch_size": 8,
        "grad_accum": 1,
        "epochs": 2,
        "desc": "Quick pipeline sanity check (small data, few steps)",
    },
    "lra-oom": {
        "train_samples": 32,
        "eval_samples": 16,
        "batch_size": 1,
        "grad_accum": 1,
        "epochs": 1,
        "desc": "Tiny budget to probe OOM/survival at large context windows (fast per run)",
    },
    "lra-report": {
        "train_samples": 500,
        "eval_samples": 200,
        "batch_size": 4,
        "grad_accum": 2,
        "epochs": 3,
        "desc": "Moderate run for the context-window report (fits a 6GB GPU)",
    },
    "lra-full": {
        "train_samples": 8000,
        "eval_samples": 1000,
        "batch_size": 8,
        "grad_accum": 2,
        "epochs": 8,
        "desc": "LRA-scale from-scratch training",
    },
}

DEFAULT_SEQ = {"listops": 2048, "text": 4096, "retrieval": 4096}

DEFAULT_SEQS = {
    "listops": [512, 1024, 2048],
    "text": [1024, 2048, 4096],
    "retrieval": [1024, 2048, 4096],
}

TRACK = "lra"
