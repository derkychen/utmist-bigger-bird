"""Shared helpers for encoder-only BART classification patches."""

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput


def bart_first_token_pool(last_hidden: torch.Tensor) -> torch.Tensor:
    """Pool encoder position 0 ([CLS] / BOS slot used by LRA/RULER datasets)."""
    return last_hidden[:, 0, :]


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over non-pad tokens — more stable for from-scratch training."""
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1)


def classification_forward(
    base_model: nn.Module,
    input_ids=None,
    attention_mask=None,
    labels=None,
    pooling: str = "mean",
    **kwargs,
):
    """Encoder forward + pool + HF classification head.

    Calls the encoder directly (not the full encoder+decoder ``BartModel``) because:
      - the sparse-attention patches live on the encoder only;
      - the from-scratch decoder in LRA never learns useful cross-attention with
        small data, so reading ``last_hidden_state`` (decoder output) gave random
        chance accuracy; and
      - skipping the decoder saves compute.

    ``pooling`` defaults to ``"mean"`` which is more stable for from-scratch LRA
    training (first-token pooling requires the model to learn to aggregate info
    to position 0, which is hard without pretraining).
    """
    inner = base_model.model
    encoder = getattr(inner, "encoder", None)
    if encoder is not None:
        enc_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        enc_hidden = enc_outputs.last_hidden_state
    else:
        # Fallback for non-BART backbones (e.g. BigBird handled elsewhere).
        outputs = inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        enc_hidden = getattr(outputs, "encoder_last_hidden_state", outputs.last_hidden_state)

    if pooling == "mean":
        pooled = mean_pool(enc_hidden, attention_mask)
    else:
        pooled = bart_first_token_pool(enc_hidden)
    logits = base_model.classification_head(pooled)

    loss = None
    if labels is not None:
        if labels.dtype != torch.long:
            labels = labels.long()
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, base_model.config.num_labels),
            labels.view(-1),
        )
    return SequenceClassifierOutput(loss=loss, logits=logits)


def force_keep_cls_indices(top_idx: torch.Tensor) -> torch.Tensor:
    """Ensure token index 0 ([CLS]) is always among the kept indices (per batch row)."""
    # top_idx: [B, K]
    bsz, keep_n = top_idx.shape
    has_cls = (top_idx == 0).any(dim=-1)  # [B]
    if bool(has_cls.all()):
        return top_idx
    out = top_idx.clone()
    for b in range(bsz):
        if has_cls[b]:
            continue
        # Replace the last kept index with 0, then re-sort to preserve order.
        out[b, -1] = 0
        out[b], _ = torch.sort(out[b])
    return out


def compute_dataset_seq_stats(ds_split, sample_limit: int = 512):
    """Length stats from tokenized dataset (input_ids per row)."""
    lengths = []
    n = min(sample_limit, len(ds_split))
    for i in range(n):
        row = ds_split[i]
        if "input_ids" in row:
            lengths.append(int(row["input_ids"].shape[0]))
        elif "attention_mask" in row:
            lengths.append(int(row["attention_mask"].sum().item()))
    if not lengths:
        return {"sample_count": 0, "max_len": 0, "mean_len": 0.0, "p95_len": 0}
    lengths.sort()
    p95_idx = min(len(lengths) - 1, int(0.95 * (len(lengths) - 1)))
    return {
        "sample_count": len(lengths),
        "max_len": lengths[-1],
        "mean_len": round(sum(lengths) / len(lengths), 2),
        "p95_len": lengths[p95_idx],
    }
