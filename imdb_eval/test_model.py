from transformers import AutoTokenizer, BigBirdForSequenceClassification
from dataset import build_imdb_dataset, DataConfig
from runner import run_experiment, TrainConfig

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model import BiggerBirdAttention
from biggerbirdconfigs import BiggerBirdConfig

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

    bigger_bird_config = BiggerBirdConfig(
        fragment_size=128,          # slightly tighter window → cleaner local top-k
        r_target_softmax=0.16,      # ensures k hits max_k at 896 tokens
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

    train_cfg = TrainConfig(
        epochs=3,
        per_device_train_bs=2,
        grad_accum_steps=8,
    )
    
    results = run_experiment(
        exp_name="biggerbird_topkMMR_globals_8000",
        model=model,
        tokenizer=tokenizer,
        ds=ds,
        cfg=train_cfg
    )

