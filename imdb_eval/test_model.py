from transformers import AutoTokenizer, BigBirdForSequenceClassification
from dataset import build_imdb_dataset, DataConfig
from runner import run_experiment, TrainConfig

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model import BiggerBirdAttention
from biggerbirdconfigs import BiggerBirdConfig

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
    tokenizer = AutoTokenizer.from_pretrained(
        "google/bigbird-roberta-base"
    )

    CONTEXT_LEN = 8000
    tokenizer.model_max_length = CONTEXT_LEN

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

    model = BigBirdForSequenceClassification.from_pretrained(
        "google/bigbird-roberta-base",
        num_labels=2,
    )

    # remember to adjust the pre-trained embeddings to account for larger token sizes. 
    model = extend_bigbird_embeddings(model, CONTEXT_LEN)

    bigger_bird_config = BiggerBirdConfig(

        fragment_size=128,          # slightly tighter window → cleaner local top-k
        r_target_softmax=0.02,      # for 0.02 * 8000 = 160 tokens relative to full sequence length.
        min_k=56,
        max_k=64,                   # locals per query (main quality driver)
        globals_per_head=6,
        teleports_per_head=4,       # a tiny bump helps long-range without much cost
        teleport_bias_frac=0.75,

        top_u=32,
        proto_count=48,

        mmr_prefilter_mult=3,
        mmr_diversity_steps=2,      # ↓ from 7 → less over-diversification, higher precision
        gamma_diversity=0.16,       # moderate penalty works best with steps=2

        alpha_pos_prior=0.12,       # restore a useful locality bias for IMDB
        share_stride_layers=2,

        dense_fallback_under=512,
        random_selection=False,
        debug_collect=False,
        log_once_pairs=True,

        # Ablation flags
        use_topk_mmr=True,
        use_dynamic_globals=True,
        use_random_attn=False,
        use_teleports=False,
    )

    for layer in model.bert.encoder.layer:
        old_attn = layer.attention.self

        new_attn = BiggerBirdAttention(bigger_bird_config)

        # optionally copy weights
        new_attn.load_state_dict(
            old_attn.state_dict(),
            strict=False
        )

        layer.attention.self = new_attn

    # Re-configure the number of samples loaded to just 1
    # to account for out-of-memory issues. Keep grad_accum to 16
    # to maintain effective batch size of 16.
    train_cfg = TrainConfig(
        epochs=3,
        per_device_train_bs=1,
        per_device_eval_bs=1,
        grad_accum_steps=16,
    )
    
    results = run_experiment(
        exp_name="biggerbird_topkMMR_globals_8000",
        model=model,
        tokenizer=tokenizer,
        ds=ds,
        cfg=train_cfg
    )