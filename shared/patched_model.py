"""Shared helpers for encoder-only BART classification patches."""

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput


def bart_first_token_pool(last_hidden: torch.Tensor) -> torch.Tensor:
    """Pool encoder position 0 ([CLS] / BOS slot used by LRA/RULER datasets)."""
    return last_hidden[:, 0, :]


def classification_forward(
    base_model: nn.Module,
    input_ids=None,
    attention_mask=None,
    labels=None,
    **kwargs,
):
    """Encoder-only forward + [CLS] pool + HF classification head.

    Uses ``base_model.model.encoder`` (not the full encoder-decoder BartModel) so the
    pooled vector is the true input-position-0 representation, matching DualTower and
    the LRA/RULER data format.
    """
    encoder = base_model.model.encoder
    enc_out = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    pooled = bart_first_token_pool(enc_out.last_hidden_state)
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
