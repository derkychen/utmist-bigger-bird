import torch
import os
import time
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback
from transformers.trainer_utils import EvalPrediction
import json
from datetime import datetime

@dataclass
class TrainConfig:
    epochs: int = 3
    per_device_train_bs: int = 2
    per_device_eval_bs: int = 2
    grad_accum_steps: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_pred):
    if isinstance(eval_pred, EvalPrediction):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        preds, labels = eval_pred
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

def device_flags():
    use_cuda = torch.cuda.is_available()
    use_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    fp16 = False
    bf16 = False
    torch_compile = False
    if use_cuda:
        bf16 = torch.cuda.is_bf16_supported()
        fp16 = not bf16
    return fp16, bf16, torch_compile, use_mps

class TrajectoryCallback(TrainerCallback):
    """Callback to capture per-epoch/step metrics for trajectory visualization."""
    def __init__(self):
        self.trajectory = []
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            point = {
                "epoch": state.epoch,
                "step": state.global_step,
                "train_loss": state.log_history[-1].get("loss", None) if state.log_history else None,
                "eval_loss": metrics.get("eval_loss", None),
                "eval_accuracy": metrics.get("eval_accuracy", None),
                "eval_f1": metrics.get("eval_f1", None),
            }
            self.trajectory.append(point)

def run_experiment(exp_name: str, model, tokenizer, ds, cfg: TrainConfig, extra_meta: dict = None, callbacks=None, save_weights: bool = False):
    fp16, bf16, torch_compile, use_mps = device_flags()
    eval_accum = 1 if use_mps else 8
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks", exp_name))
    os.makedirs(out_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_train_bs,
        per_device_eval_batch_size=cfg.per_device_eval_bs,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=fp16,
        bf16=bf16,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=False,
        torch_compile=torch_compile,
        optim="adamw_torch",
        eval_accumulation_steps=eval_accum,
    )

    # Setup trajectory tracking
    traj_callback = TrajectoryCallback()
    all_callbacks = [traj_callback] + (callbacks if callbacks else [])
    
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=64)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=all_callbacks
    )

    print(f"[{exp_name}] Starting training...", flush=True)
    start_time = time.time()
    train_res = trainer.train()
    train_time = time.time() - start_time
    
    print(f"[{exp_name}] Evaluating...", flush=True)
    eval_res = trainer.evaluate()
    
    # 📝 Prepare Rich Metadata and Structured Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment_metadata": {
            "name": exp_name,
            "timestamp": timestamp,
            "training_config": {
                "epochs": cfg.epochs,
                "batch_size": cfg.per_device_train_bs,
                "accumulation_steps": cfg.grad_accum_steps,
                "learning_rate": cfg.lr,
                "warmup": cfg.warmup_ratio
            },
            "dataset_info": {
                "train_size": len(ds["train"]),
                "eval_size": len(ds["validation"]),
                "max_seq_len": ds["train"][0]["input_ids"].shape[0] if "input_ids" in ds["train"][0] else "unknown"
            },
            "environment": {
                "use_mps": use_mps,
                "fp16": fp16
            },
            "model_config": extra_meta or {}
        },
        "performance_metrics": {
            "training_time_seconds": train_time,
            "train": train_res.metrics,
            "eval": eval_res,
            "trajectory": traj_callback.trajectory
        }
    }
    
    # Optionally save model weights
    weights_path = None
    if save_weights:
        weights_dir = os.path.join(out_dir, f"weights_{timestamp}")
        os.makedirs(weights_dir, exist_ok=True)
        # Unwrap PatchedModel wrapper to get the underlying HF model if possible
        save_model = getattr(model, "model", model)
        save_model.save_pretrained(weights_dir)
        tokenizer.save_pretrained(weights_dir)
        weights_path = weights_dir
        print(f"[{exp_name}] Weights saved to {weights_dir}")
    
    if weights_path:
        results["experiment_metadata"]["weights_path"] = weights_path

    # Save as timestamped JSON for scaling law analysis
    json_path = os.path.join(out_dir, f"eval_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    # Also update the legacy results.txt for quick human check
    res_path = os.path.join(out_dir, "results.txt")
    with open(res_path, "w") as f:
        f.write(f"Experiment: {exp_name} | Date: {timestamp}\n")
        f.write(f"Training Time: {train_time:.2f}s\n")
        f.write(f"Accuracy: {eval_res.get('eval_accuracy', 'N/A')}\n")
        f.write(f"F1: {eval_res.get('eval_f1', 'N/A')}\n")

    print(f"[{exp_name}] Results exported to {json_path}")
    return eval_res