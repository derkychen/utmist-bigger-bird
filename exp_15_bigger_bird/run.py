import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, BigBirdForSequenceClassification
from shared.dataset import build_imdb_dataset, DataConfig
from shared.runner import run_experiment, TrainConfig
from exp_15_bigger_bird.model_wrapper import PatchedModel


def main():
    context_len = 8000
    model_name = "google/bigbird-roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = BigBirdForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    params = dict(
        context_len=context_len,
        fragment_size=128,
        r_target_softmax=0.02,
        min_k=56,
        max_k=64,
        globals_per_head=6,
        teleports_per_head=4,
        teleport_bias_frac=0.75,
        top_u=32,
        proto_count=48,
        mmr_prefilter_mult=3,
        mmr_diversity_steps=2,
        gamma_diversity=0.16,
        alpha_pos_prior=0.12,
        use_topk_mmr=True,
        use_dynamic_globals=True,
        use_random_attn=True,
        use_teleports=False,
    )
    model = PatchedModel(base_model, **params)

    data_cfg = DataConfig(
        max_length=context_len,
        train_samples=6000,
        eval_samples=1000,
    )
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=context_len)

    train_cfg = TrainConfig(epochs=3, per_device_train_bs=2, grad_accum_steps=8)
    run_experiment(
        f"exp_15_bigger_bird_{context_len}",
        model, tokenizer, ds, train_cfg,
        extra_meta=params,
    )


if __name__ == "__main__":
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    main()
