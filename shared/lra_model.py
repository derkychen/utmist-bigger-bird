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

# BigBird is only needed for exp_15 (proper BiggerBird, BigBird-RoBERTa backbone).
# Import lazily so the rest of the LRA pipeline keeps working if BigBird is unavailable.
try:
    from transformers import BigBirdConfig, BigBirdForSequenceClassification
    _BIGBIRD_AVAILABLE = True
except Exception:  # pragma: no cover - depends on optional deps / PIL etc.
    BigBirdConfig = None
    BigBirdForSequenceClassification = None
    _BIGBIRD_AVAILABLE = False

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


# exp_15 (proper BiggerBird) is the only experiment on a BigBird backbone instead of
# BART. Its ``PatchedModel`` expects ``base_model.bert.encoder.layer`` and a
# ``BigBirdBlockSparseAttention``-compatible config, so the from-scratch LRA encoder
# must be a ``BigBirdForSequenceClassification`` for that one experiment.
BIGBIRD_EXPS = {15}
# BigBird block-sparse attention requires seq_len % block_size == 0. exp_15 uses
# fragment_size=128 as its block_size, so we match that here.
BIGBIRD_BLOCK_SIZE = 128


def _round_up_to_block(seq_len: int, block_size: int = BIGBIRD_BLOCK_SIZE) -> int:
    return ((seq_len + block_size - 1) // block_size) * block_size


def make_bigbird(vocab_size, seq_len, num_labels):
    """Construct a randomly-initialized BigBird-shaped sequence-classification model.

    Mirrors ``make_bart`` (same d_model / heads / layers / ffn) but on the BigBird
    architecture so exp_15's ``BiggerBirdAttention`` (which extends
    ``BigBirdBlockSparseAttention``) can drop in. ``max_position_embeddings`` is rounded
    up to a multiple of ``block_size``; BigBird falls back to dense attention for seqs
    shorter than ~9 blocks (1152 tokens at block_size=128), which is intrinsic to BigBird.
    """
    if not _BIGBIRD_AVAILABLE:
        raise ImportError(
            "exp_15 requires transformers.BigBirdForSequenceClassification; "
            "install a compatible transformers version (pinned: transformers==5.8.1)."
        )
    max_pos = _round_up_to_block(seq_len)
    cfg = BigBirdConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_pos,
        hidden_size=LRA_D_MODEL,
        num_hidden_layers=LRA_ENCODER_LAYERS,
        num_attention_heads=LRA_HEADS,
        intermediate_size=LRA_FFN,
        num_labels=num_labels,
        # BigBird block-sparse params; exp_15 reuses fragment_size as block_size.
        block_size=BIGBIRD_BLOCK_SIZE,
        num_random_blocks=2,
        attention_type="block_sparse",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        dropout=0.1,
        classifier_dropout=0.1,
    )
    return BigBirdForSequenceClassification(cfg)


def _is_bigbird_exp(exp_num: int) -> bool:
    return exp_num in BIGBIRD_EXPS


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


def _model_params(exp_num, seq_len=None):
    """Return (exp_name, ModelClass, kwargs) for an experiment, dropping non-kwarg meta.

    For exp_15 (proper BiggerBird), ``context_len`` is overridden to ``seq_len`` to match
    the compute window (mirroring what ``run_experiment.py`` does for the IMDb track).
    """
    name, model_class, params = EXPERIMENT_CONFIGS[exp_num]
    kwargs = {k: v for k, v in params.items() if k != "attention"}
    if _is_bigbird_exp(exp_num) and seq_len is not None:
        kwargs["context_len"] = seq_len
    return name, model_class, kwargs, dict(params)


def build_classification_model(exp_num, vocab_size, seq_len, num_labels):
    """Build a single-sequence LRA classifier (listops / text) for the given experiment.

    Returns (model, exp_name, meta_params). ``exp_num == 0`` is the dense baseline.
    exp_15 is built on a from-scratch BigBird encoder (see ``make_bigbird``); all other
    experiments use the BART-shaped encoder.
    """
    if _is_bigbird_exp(exp_num):
        base = make_bigbird(vocab_size, seq_len, num_labels)
    else:
        base = make_bart(vocab_size, seq_len, num_labels)
    name, model_class, kwargs, meta = _model_params(exp_num, seq_len=seq_len)
    if model_class is None:
        return DenseBaseline(base), name, meta
    model = model_class(base, **kwargs)
    return model, name, meta


def _encoder_of(body):
    """Return the (patched) encoder regardless of whether ``body`` is a PatchedModel.

    Works for both BART (``bfsc.model.encoder``) and BigBird (``bfsc.bert.encoder``)
    backbones, so exp_15 (proper BiggerBird) can reuse the dual-tower retrieval head.
    """
    bfsc = body if hasattr(body, "classification_head") else body.model
    if hasattr(bfsc, "model") and hasattr(bfsc.model, "encoder"):
        return bfsc.model.encoder  # BART
    if hasattr(bfsc, "bert") and hasattr(bfsc.bert, "encoder"):
        return bfsc.bert.encoder  # BigBird
    raise AttributeError(f"Could not locate encoder on {type(bfsc).__name__}")


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
    if _is_bigbird_exp(exp_num):
        raise NotImplementedError(
            f"exp_{exp_num} (proper BiggerBird / BigBird backbone) is not supported on "
            "the LRA retrieval (dual-tower) task yet: BigBird's block-sparse masks are "
            "prepared by the full BigBirdModel, not the bare encoder, so the shared "
            "DualTowerRetrieval head would need a BigBird-specific encode path. "
            "Use listops/text (LRA) or niah/mq_niah (RULER) for exp_15 instead."
        )
    base = make_bart(vocab_size, seq_len, num_labels)
    name, model_class, kwargs, meta = _model_params(exp_num, seq_len=seq_len)
    body = base if model_class is None else model_class(base, **kwargs)
    model = DualTowerRetrieval(body, d_model=base.config.d_model, num_labels=num_labels)
    return model, name, meta


def build_lra_model(task, exp_num, vocab_size, seq_len, num_labels, pair=False):
    """Dispatch to the single-sequence or dual-tower builder based on the task."""
    if pair:
        return build_retrieval_model(exp_num, vocab_size, seq_len, num_labels)
    return build_classification_model(exp_num, vocab_size, seq_len, num_labels)
