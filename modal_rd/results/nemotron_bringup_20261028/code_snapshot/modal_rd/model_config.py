"""Shared checkpoint selection for Modal runners; defaults preserve the R1 protocol."""
import hashlib
import json
import os
from pathlib import Path

R1_ID = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
NEMOTRON_ID = 'nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct'
ALIASES = {'r1': R1_ID, 'nemotron': NEMOTRON_ID}
choice = os.environ.get('BB_MODEL', 'r1')
MODEL_ID = ALIASES.get(choice, choice)
if MODEL_ID not in ALIASES.values():
    raise ValueError('BB_MODEL must be r1, nemotron, or one of their full Hugging Face IDs')
MODEL_DIR = str(Path(os.environ.get('SCRATCH', '/vol')) / 'models' / MODEL_ID.split('/')[-1])
MODEL_REVISION = os.environ.get('BB_MODEL_REVISION') or (
    'fa3a10b9c6ad02f42c81e1d68dcf6967afb230ef' if MODEL_ID == NEMOTRON_ID else 'main')
PROMPT_FORMAT = os.environ.get('BB_PROMPT_FORMAT', 'chat' if MODEL_ID == NEMOTRON_ID else 'raw')
if PROMPT_FORMAT not in ('raw', 'chat'):
    raise ValueError('BB_PROMPT_FORMAT must be raw or chat')
MAX_CONTEXT = 1073152 if MODEL_ID == NEMOTRON_ID else 131072


def model_metadata():
    marker = Path(MODEL_DIR) / 'bb_checkpoint.json'
    checkpoint = json.loads(marker.read_text()) if marker.exists() else {}
    return {'model_id': MODEL_ID, 'model_path': MODEL_DIR, 'requested_revision': MODEL_REVISION,
            'resolved_revision': checkpoint.get('resolved_revision'), 'prompt_format': PROMPT_FORMAT,
            'max_context': MAX_CONTEXT, 'source_sha256': source_hashes()}


def render_prompt(tokenizer, text):
    if PROMPT_FORMAT == 'raw':
        return text
    return tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                        tokenize=False, add_generation_prompt=True)


def tokenize_prompt(tokenizer, text, **kwargs):
    # Chat templates already contain their special tokens (including BOS).
    return tokenizer(text, add_special_tokens=PROMPT_FORMAT == 'raw', **kwargs)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = ('modal_rd/model_config.py', 'modal_rd/model_preflight.py', 'modal_rd/niah.py', 'modal_rd/attention.py',
             'modal_rd/eval_niah.py', 'modal_rd/selfcheck.py', 'modal_rd/nemotron_bringup.py',
             'kernels/bigger_bird_flash.py')
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files if (root / f).exists()}
