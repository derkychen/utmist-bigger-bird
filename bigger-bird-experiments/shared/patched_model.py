"""Shared helpers for encoder-only BART classification patches."""

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput


def bart_first_token_pool(last_hidden: torch.Tensor) -> torch.Tensor:
    """Match HuggingFace BartForSequenceClassification: pool encoder position 0."""
    return last_hidden[:, 0, :]


def classification_forward(
    base_model: nn.Module,
    input_ids=None,
    attention_mask=None,
    labels=None,
    **kwargs,
):
    """Encoder forward + first-token pool + HF classification head."""
    outputs = base_model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    pooled = bart_first_token_pool(outputs.last_hidden_state)
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
