from transformers import AutoTokenizer, BigBirdForSequenceClassification
from imdb_eval.dataset import build_imdb_dataset, DataConfig
from imdb_eval.runner import run_experiment, TrainConfig

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "google/bigbird-roberta-base"
    )

    data_cfg = DataConfig(
        max_length=768,
        train_samples=6000,
        eval_samples=1000,
    )

    ds = build_imdb_dataset(
        tokenizer,
        data_cfg,
        fixed_length=768
    )

    train_cfg = TrainConfig(
        epochs=3,
        per_device_train_bs=2,
        grad_accum_steps=8,
    )

    model = BigBirdForSequenceClassification.from_pretrained(
        "google/bigbird-roberta-base",
        num_labels=2,
    )

    results = run_experiment(
        exp_name="bigbird",
        model=model,
        tokenizer=tokenizer,
        ds=ds,
        cfg=train_cfg
    )

