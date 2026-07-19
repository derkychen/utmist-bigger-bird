"""From-scratch encoders for the LRA / RULER long-context evaluation tracks.

Most experiments (0–14) use a randomly initialized BART-shaped encoder and patch
``BartAttention``. Exp 15 (BiggerBird) instead uses a from-scratch BigBird-RoBERTa
backbone so the existing BigBird attention wrapper can run unchanged.

For retrieval (document matching) a dual-tower wrapper runs the shared patched
encoder over both documents and combines [u, v, |u-v|, u*v] -> MLP -> 2 logits.
"""

import os
import sys

import torch
import torch.nn as nn
from transformers import (
    BartConfig,
    BartForSequenceClassification,
    BigBirdConfig,
    BigBirdForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# Make the repo root importable so we can reuse the canonical experiment registry.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from run_experiment import EXPERIMENT_CONFIGS  # noqa: E402  (maps exp_num -> (name, ModelClass, params))
from shared.patched_model import classification_forward  # noqa: E402

# LRA convention: small encoder trained from scratch (d=512, 8 heads, 6 layers, ffn=2048).
LRA_D_MODEL = 512
LRA_ENCODER_LAYERS = 6
LRA_HEADS = 8
LRA_FFN = 2048


def make_bart(vocab_size, seq_len, num_labels):
    """Construct a randomly-initialized BART-shaped sequence-classification model.

    The decoder is kept minimal (1 layer) for HF BART shape compatibility;
    classification uses encoder-only [CLS] pooling via ``classification_forward``.
    """
    cfg = BartConfig(
        vocab_size=vocab_size,
        max_position_embeddings=seq_len + 8,
        d_model=LRA_D_MODEL,
        encoder_layers=LRA_ENCODER_LAYERS,
        decoder_layers=1,
        encoder_attention_heads=LRA_HEADS,
        decoder_attention_heads=LRA_HEADS,
        encoder_ffn_dim=LRA_FFN,
        decoder_ffn_dim=LRA_FFN,
        num_labels=num_labels,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=2,
        dropout=0.1,
        classifier_dropout=0.1,
    )
    return BartForSequenceClassification(cfg)


def _choose_fragment_size(
    seq_len: int,
    preferred: int = 128,
    min_blocks: int = 8,
    num_random_blocks: int = 2,
) -> int:
    """Pick a fragment/block size compatible with BigBird block-sparse plans.

    HF keeps ``block_sparse`` only when
    ``seq_len > (5 + 2 * num_random_blocks) * block_size``. Using a larger
    fragment causes a silent switch to ``original_full``, which breaks once
    BiggerBirdAttention has replaced the self-attention module.
    """
    preferred = max(16, int(preferred))
    # Max fragment that still qualifies for block_sparse.
    max_frag = max(16, (seq_len - 1) // (5 + 2 * int(num_random_blocks)))
    candidates = []
    for frag in (preferred, 64, 32, 16):
        if frag > seq_len or frag > max_frag:
            continue
        if seq_len % frag == 0 and (seq_len // frag) >= min_blocks:
            candidates.append(frag)
    if candidates:
        return max(candidates)
    frag = min(preferred, max_frag)
    while frag > 16 and (seq_len % frag != 0 or seq_len // frag < 2):
        frag //= 2
    return max(16, frag)


def _exp15_kwargs(seq_len: int, kwargs: dict) -> dict:
    """Adapt exp_15 hyperparameters to the requested LRA/RULER context length."""
    out = dict(kwargs)
    out["context_len"] = seq_len
    preferred = int(out.get("fragment_size", 128))
    out["fragment_size"] = _choose_fragment_size(seq_len, preferred=preferred)
    out["dense_fallback_under"] = min(
        int(out.get("dense_fallback_under", 512)), max(64, seq_len // 2)
    )
    return out


def make_bigbird(vocab_size, seq_len, num_labels, block_size: int | None = None):
    """Construct a randomly-initialized BigBird classifier matching LRA size.

    Used only for exp_15 so the BigBird-native BiggerBirdAttention wrapper can
    attach to ``base_model.bert.encoder.layer``. Must use ``attention_type=block_sparse``
    so HF builds the band/random masks BiggerBirdAttention expects.
    """
    if block_size is None:
        block_size = _choose_fragment_size(seq_len)
    max_pos = max(seq_len + 2, 512)
    cfg = BigBirdConfig(
        vocab_size=vocab_size,
        hidden_size=LRA_D_MODEL,
        num_hidden_layers=LRA_ENCODER_LAYERS,
        num_attention_heads=LRA_HEADS,
        intermediate_size=LRA_FFN,
        max_position_embeddings=max_pos,
        num_labels=num_labels,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sep_token_id=2,
        cls_token_id=1,
        type_vocab_size=1,
        attention_type="block_sparse",
        block_size=block_size,
        num_random_blocks=2,  # match PatchedModel BiggerBirdConfig
        use_bias=True,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        classifier_dropout=0.1,
    )
    return BigBirdForSequenceClassification(cfg)


def _hidden_size(config) -> int:
    return int(getattr(config, "d_model", None) or getattr(config, "hidden_size"))


class DenseBaseline(nn.Module):
    """Dense (exp 0) baseline that pools the [CLS] slot via the shared classification path."""

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return classification_forward(self.model, input_ids, attention_mask, labels, **kwargs)

    @property
    def config(self):
        return self.model.config

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    @property
    def supports_gradient_checkpointing(self):
        return getattr(self.model, "supports_gradient_checkpointing", True)


class BigBirdClsWrapper(nn.Module):
    """Fair [CLS] readout over a BigBird (exp_15) PatchedModel for LRA/RULER.

    Runs the BigBird encoder and pools position 0, matching the BART [CLS] protocol.
    """

    def __init__(self, body, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.body = body
        self.num_labels = num_labels
        hidden = _hidden_size(body.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Prefer the patched PatchedModel forward (same path as IMDb exp_15), then
        # re-read encoder [CLS] from the backbone for a fair LRA/RULER pool.
        inner = self.body.model if hasattr(self.body, "model") else self.body
        bert_out = inner.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled = bert_out.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)

    @property
    def config(self):
        return self.body.config

    def gradient_checkpointing_enable(self, **kwargs):
        inner = self.body.model if hasattr(self.body, "model") else self.body
        if hasattr(inner, "gradient_checkpointing_enable"):
            inner.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        inner = self.body.model if hasattr(self.body, "model") else self.body
        if hasattr(inner, "gradient_checkpointing_disable"):
            inner.gradient_checkpointing_disable()

    @property
    def supports_gradient_checkpointing(self):
        return True


def _model_params(exp_num):
    """Return (exp_name, ModelClass, kwargs) for an experiment, dropping non-kwarg meta."""
    name, model_class, params = EXPERIMENT_CONFIGS[exp_num]
    kwargs = {k: v for k, v in params.items() if k != "attention"}
    return name, model_class, kwargs, dict(params)


def _encoder_module(body):
    """Return the encoder module for DualTower (BART or BigBird)."""
    if hasattr(body, "token_drop_encoder"):
        return body.token_drop_encoder
    if hasattr(body, "dynamic_encoder"):
        return body.dynamic_encoder
    if hasattr(body, "body"):
        body = body.body
    inner = body.model if (hasattr(body, "model") and not hasattr(body, "classification_head")) else body
    if hasattr(inner, "bert"):
        return inner.bert.encoder
    if hasattr(inner, "model") and hasattr(inner.model, "encoder"):
        return inner.model.encoder
    raise AttributeError(f"Cannot locate encoder on {type(body)!r} / {type(inner)!r}")


class DualTowerRetrieval(nn.Module):
    """Dual-tower matching head over a shared (patched) encoder."""

    def __init__(self, body, d_model, num_labels=2, dropout=0.1):
        super().__init__()
        self.body = body
        self.num_labels = num_labels
        self.head = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_labels),
        )

    def _encode(self, input_ids, attention_mask):
        # BigBird: run full bert so embeddings/masks are correct, then pool CLS.
        inner = self.body.model if (hasattr(self.body, "model") and hasattr(self.body.model, "bert")) else None
        if inner is not None and hasattr(inner, "bert"):
            enc = inner.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            return enc.last_hidden_state[:, 0, :]
        enc_mod = _encoder_module(self.body)
        enc = enc_mod(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if isinstance(enc, tuple):
            enc = enc[0]
        return enc.last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids_a=None,
        attention_mask_a=None,
        input_ids_b=None,
        attention_mask_b=None,
        labels=None,
        **kwargs,
    ):
        u = self._encode(input_ids_a, attention_mask_a)
        v = self._encode(input_ids_b, attention_mask_b)
        feats = torch.cat([u, v, (u - v).abs(), u * v], dim=-1)
        logits = self.head(feats)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.long().view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)

    @property
    def config(self):
        if hasattr(self.body, "config"):
            return self.body.config
        bfsc = self.body if hasattr(self.body, "classification_head") else self.body.model
        return bfsc.config

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.body, "gradient_checkpointing_enable"):
            self.body.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.body, "gradient_checkpointing_disable"):
            self.body.gradient_checkpointing_disable()

    @property
    def supports_gradient_checkpointing(self):
        return getattr(self.body, "supports_gradient_checkpointing", True)


def build_retrieval_model(exp_num, vocab_size, seq_len, num_labels=2):
    """Build a dual-tower LRA retrieval model for the given experiment."""
    name, model_class, kwargs, meta = _model_params(exp_num)
    if exp_num == 15:
        kw = _exp15_kwargs(seq_len, kwargs)
        base = make_bigbird(vocab_size, seq_len, num_labels, block_size=kw["fragment_size"])
        body = model_class(base, **kw)
        d_model = _hidden_size(base.config)
        meta = {**meta, "backbone": "bigbird_from_scratch", "fragment_size": kw["fragment_size"]}
    else:
        base = make_bart(vocab_size, seq_len, num_labels)
        body = base if model_class is None else model_class(base, **kwargs)
        d_model = _hidden_size(base.config)
    model = DualTowerRetrieval(body, d_model=d_model, num_labels=num_labels)
    return model, name, meta


def build_classification_model(exp_num, vocab_size, seq_len, num_labels):
    """Build a single-sequence LRA/RULER classifier for the given experiment.

    Exp 15 uses a from-scratch BigBird backbone + BiggerBirdAttention.
    """
    name, model_class, kwargs, meta = _model_params(exp_num)
    if exp_num == 15:
        kw = _exp15_kwargs(seq_len, kwargs)
        base = make_bigbird(vocab_size, seq_len, num_labels, block_size=kw["fragment_size"])
        body = model_class(base, **kw)
        model = BigBirdClsWrapper(body, num_labels=num_labels)
        meta = {**meta, "backbone": "bigbird_from_scratch", "fragment_size": kw["fragment_size"]}
        return model, name, meta
    base = make_bart(vocab_size, seq_len, num_labels)
    if model_class is None:
        return DenseBaseline(base), name, meta
    model = model_class(base, **kwargs)
    return model, name, meta


def build_lra_model(task, exp_num, vocab_size, seq_len, num_labels, pair=False):
    """Dispatch to the single-sequence or dual-tower builder based on the task."""
    if pair:
        return build_retrieval_model(exp_num, vocab_size, seq_len, num_labels)
    return build_classification_model(exp_num, vocab_size, seq_len, num_labels)
