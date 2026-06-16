from transformers import AutoTokenizer, BigBirdForSequenceClassification
from dataset import build_imdb_dataset, DataConfig
from runner import run_experiment, TrainConfig

import torch
import torch.nn as nn



def extend_bigbird_embeddings(model, context_len: int):
    """
    BigBird-RoBERTa was pretrained with max_position_embeddings around 4096.
    If we pass 8000 or larger token inputs, we must resize:
      1. position_embeddings
      2. position_ids buffer
      3. token_type_ids buffer

    Otherwise, we get:
    RuntimeError: expanded size X must match existing size 4096
    """
    emb = model.bert.embeddings

    old_pos_emb = emb.position_embeddings
    old_max, hidden_size = old_pos_emb.weight.shape

    padding_idx = old_pos_emb.padding_idx
    if padding_idx is None:
        padding_idx = 0

    # RoBERTa-style position ids can reach padding_idx + context_len.
    # Embedding size must therefore be at least context_len + padding_idx + 1.
    new_max = context_len + padding_idx + 1

    print(f"[pos-emb] old_max={old_max}, context_len={context_len}, new_max={new_max}")

    if new_max > old_max:
        new_pos_emb = nn.Embedding(
            new_max,
            hidden_size,
            padding_idx=padding_idx,
        ).to(
            device=old_pos_emb.weight.device,
            dtype=old_pos_emb.weight.dtype,
        )

        with torch.no_grad():
            # Copy pretrained positions.
            new_pos_emb.weight[:old_max] = old_pos_emb.weight

            # Initialize new positions by repeating the final pretrained position.
            # This is simple and avoids random initialization spikes.
            new_pos_emb.weight[old_max:] = old_pos_emb.weight[-1].unsqueeze(0)

        emb.position_embeddings = new_pos_emb

    # These buffers caused the expansion error
    emb.register_buffer(
        "position_ids",
        torch.arange(new_max, device=old_pos_emb.weight.device).expand((1, -1)),
        persistent=False,
    )

    emb.register_buffer(
        "token_type_ids",
        torch.zeros(
            (1, new_max),
            dtype=torch.long,
            device=old_pos_emb.weight.device,
        ),
        persistent=False,
    )

    model.config.max_position_embeddings = new_max
    model.bert.config.max_position_embeddings = new_max

    print("[pos-emb] position_embeddings:", emb.position_embeddings.weight.shape)
    print("[pos-emb] position_ids:", emb.position_ids.shape)
    print("[pos-emb] token_type_ids:", emb.token_type_ids.shape)

    return model



if __name__ == "__main__":
    CONTEXT_LEN = 8000

    tokenizer = AutoTokenizer.from_pretrained(
        "google/bigbird-roberta-base"
    )

    model = BigBirdForSequenceClassification.from_pretrained(
        "google/bigbird-roberta-base",
        num_labels=2,
    )

    if CONTEXT_LEN > 4096:
        tokenizer.model_max_length = CONTEXT_LEN
        # remember to adjust the pre-trained embeddings to account for larger token sizes. 
        model = extend_bigbird_embeddings(model, CONTEXT_LEN)

    data_cfg = DataConfig(
        max_length=CONTEXT_LEN,
        train_samples=6000,
        eval_samples=1000,
    )

    ds = build_imdb_dataset(
        tokenizer,
        data_cfg,
        fixed_length=CONTEXT_LEN,
    )

    train_cfg = TrainConfig(
        epochs=3,
        per_device_train_bs=2,
        grad_accum_steps=8,
    )

    results = run_experiment(
        exp_name="bigbird_8000",
        model=model,
        tokenizer=tokenizer,
        ds=ds,
        cfg=train_cfg,
    )

