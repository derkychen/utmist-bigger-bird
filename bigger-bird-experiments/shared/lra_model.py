"""From-scratch BART-shaped encoder for the LRA long-context evaluation track.

LRA tasks are symbolic / byte-level, so (unlike the IMDb experiments) there is no
useful pretrained checkpoint -- following LRA convention we build a *randomly
initialized* encoder and train from scratch. Crucially, the 13 sparse-attention
modules all patch ``BartAttention`` (encoder self-attention only) via each
experiment's ``PatchedModel`` wrapper, so they drop straight into a fresh
``BartForSequenceClassification`` with a task-native vocabulary.

For retrieval (document matching) a single classification head cannot express a
pairwise decision, so we add a dual-tower wrapper that runs the shared patched
encoder over both documents and combines [u, v, |u-v|, u*v] -> MLP -> 2 logits.
"""

import os
import sys

import torch
import torch.nn as nn
from transformers import BartConfig, BartForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

# Make the repo root importable so we can reuse the canonical experiment registry.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from run_experiment import EXPERIMENT_CONFIGS  # noqa: E402  (maps exp_num -> (name, ModelClass, params))
from shared.patched_model import classification_forward  # noqa: E402

# LRA convention: small encoder trained from scratch (d_model=512, 8 heads, 6 layers, ffn=2048).
LRA_D_MODEL = 512
LRA_ENCODER_LAYERS = 6
LRA_HEADS = 8
LRA_FFN = 2048


def make_bart(vocab_size, seq_len, num_labels):
    """Construct a randomly-initialized BART-shaped sequence-classification model.

    The decoder is kept minimal (1 layer); the experiments only patch encoder
    self-attention, and classification pools the [CLS] slot, so the decoder is just
    plumbing for the reused ``classification_forward`` path.
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
    model = BartForSequenceClassification(cfg)
    return model


class DenseBaseline(nn.Module):
    """Dense (exp 0) baseline that pools the [CLS] slot via the shared classification path.

    This matches the patched experiments' pooling (position 0) instead of HF's default
    EOS pooling, so the LRA datasets don't need an EOS token and the baseline is a fair
    reference for the sparse variants.
    """

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


def _model_params(exp_num):
    """Return (exp_name, ModelClass, kwargs) for an experiment, dropping non-kwarg meta."""
    name, model_class, params = EXPERIMENT_CONFIGS[exp_num]
    kwargs = {k: v for k, v in params.items() if k != "attention"}
    return name, model_class, kwargs, dict(params)


def build_classification_model(exp_num, vocab_size, seq_len, num_labels):
    """Build a single-sequence LRA classifier (listops / text) for the given experiment.

    Returns (model, exp_name, meta_params). ``exp_num == 0`` is the dense baseline.
    """
    base = make_bart(vocab_size, seq_len, num_labels)
    name, model_class, kwargs, meta = _model_params(exp_num)
    if model_class is None:
        return DenseBaseline(base), name, meta
    model = model_class(base, **kwargs)
    return model, name, meta


def _encoder_of(body):
    """Return the (patched) BartEncoder regardless of whether ``body`` is a PatchedModel."""
    bfsc = body if hasattr(body, "classification_head") else body.model
    return bfsc.model.encoder


class DualTowerRetrieval(nn.Module):
    """Dual-tower matching head over a shared (patched) encoder.

    Note: for exp_8 (token-drop) the mid-network token dropping lives in its own
    encoder wrapper and is bypassed here, so retrieval uses its dense-kernel attention;
    every other experiment's encoder self-attention patch applies normally.
    """

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
        enc = _encoder_of(self.body)(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        return enc.last_hidden_state[:, 0, :]  # [CLS]

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
    base = make_bart(vocab_size, seq_len, num_labels)
    name, model_class, kwargs, meta = _model_params(exp_num)
    body = base if model_class is None else model_class(base, **kwargs)
    model = DualTowerRetrieval(body, d_model=base.config.d_model, num_labels=num_labels)
    return model, name, meta


def build_lra_model(task, exp_num, vocab_size, seq_len, num_labels, pair=False):
    """Dispatch to the single-sequence or dual-tower builder based on the task."""
    if pair:
        return build_retrieval_model(exp_num, vocab_size, seq_len, num_labels)
    return build_classification_model(exp_num, vocab_size, seq_len, num_labels)
