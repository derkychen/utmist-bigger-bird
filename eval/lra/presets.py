"""LRA compute presets and default sequence lengths (full LRA suite)."""

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

# seq_len includes the leading [CLS] token used by the shared encoder.
DEFAULT_SEQ = {
    "listops": 2048,
    "text": 4096,
    "retrieval": 4096,
    "image": 1025,          # 32x32 pixels + [CLS]
    "pathfinder": 1025,     # 32x32 + [CLS]
    "pathfinder_x": 16385,  # 128x128 + [CLS]
}

DEFAULT_SEQS = {
    "listops": [512, 1024, 2048],
    "text": [1024, 2048, 4096],
    "retrieval": [1024, 2048, 4096],
    "image": [257, 1025],           # 16x16 and 32x32
    "pathfinder": [257, 1025],
    "pathfinder_x": [4097, 16385],  # 64x64 smoke / full 128x128
}

ALL_TASKS = ["listops", "text", "retrieval", "image", "pathfinder", "pathfinder_x"]

TRACK = "lra"
