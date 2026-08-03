"""Training runner for Llama-3 (R1-Distill-Llama-8B) experiments.

Wraps the existing ``shared.runner.run_experiment`` with Llama-specific defaults:
  - LoRA (only adapters + classification head are trainable)
  - Gradient checkpointing (essential for 8B on 40GB MIG)
  - Smaller batch sizes, higher LR (LoRA needs ~10x the LR of full FT)
  - bf16 mixed precision

Usage:
    from shared.llama_runner import run_llama_experiment, LlamaTrainConfig

    run_llama_experiment("exp_1_deepseek_topk", model, tokenizer, ds, cfg)
"""

import os
import time
import json
import torch
import numpy as np
from config_schema.trainer.llama import LlamaTrainConfig
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
)
from transformers.trainer_utils import EvalPrediction

from patches.original_patches.runner import (
    _reset_peak_memory,
    _peak_memory_mb,
    _compute_softmax_comparisons,
    _measure_inference_latency,
    TrajectoryCallback,
    preprocess_logits_for_metrics,
    compute_metrics,
)
from patches.original_patches.patched_model import compute_dataset_seq_stats


def run_llama_experiment(
    exp_name: str,
    model,
    tokenizer,
    ds,
    cfg: LlamaTrainConfig,
    extra_meta: dict | None = None,
    callbacks=None,
):
    """Train + evaluate a LlamaPatchedModel on the given dataset.

    The model should already be:
      1. Created via ``LlamaPatchedModel.from_pretrained(...)``
      2. Patched with the experiment's sparse attention
      3. LoRA applied via ``apply_lora(...)``
      4. Moved to the correct device

    This function handles gradient checkpointing, training, evaluation,
    and result logging — mirroring ``shared.runner.run_experiment`` but
    with Llama-8B-appropriate settings.
    """
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "benchmarks", exp_name)
    )
    os.makedirs(out_dir, exist_ok=True)

    # --- Enable gradient checkpointing (must be before Trainer) ---
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # PEFT + grad checkpointing: need input grads so LoRA adapters receive gradients
        if hasattr(model.model, "enable_input_require_grads"):
            model.model.enable_input_require_grads()

    # --- Print trainable params ---
    if hasattr(model.model, "print_trainable_parameters"):
        model.model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{exp_name}] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --- Training arguments ---
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
        max_grad_norm=cfg.max_grad_norm,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        bf16=True,                       # H100 supports bf16 natively
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=False,    # we enabled it manually above
        optim="adamw_torch",
        eval_accumulation_steps=4,
    )

    # --- Trainer ---
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
        callbacks=all_callbacks,
    )

    # --- Train ---
    _reset_peak_memory()
    print(f"[{exp_name}] Starting training...", flush=True)
    start_time = time.time()
    train_res = trainer.train()
    train_time = time.time() - start_time
    peak_mem_mb = _peak_memory_mb()
    print(f"[{exp_name}] Peak memory: {peak_mem_mb:.1f} MB")

    # --- Evaluate ---
    print(f"[{exp_name}] Evaluating...", flush=True)
    eval_res = trainer.evaluate()

    # --- Stats ---
    train_seq_stats = compute_dataset_seq_stats(ds["train"])
    seq_len = train_seq_stats["max_len"] or (
        ds["train"][0]["input_ids"].shape[0] if "input_ids" in ds["train"][0] else 256
    )

    # Inference latency
    device = next(model.parameters()).device
    try:
        inf_latency_ms = _measure_inference_latency(
            model, tokenizer, device, seq_len=min(seq_len, 512), n_trials=5
        )
        print(f"[{exp_name}] Inference latency: {inf_latency_ms:.2f} ms/seq")
    except Exception as e:
        inf_latency_ms = None
        print(f"[{exp_name}] Inference latency measurement failed: {e}")

    # Softmax comparisons
    softmax_comparisons = _compute_softmax_comparisons(seq_len, model, extra_meta)
    if softmax_comparisons:
        # Llama-3-8B: 32 layers, 32 heads
        baseline_comparisons = seq_len * seq_len * 32 * 32
        reduction_pct = (1 - softmax_comparisons / baseline_comparisons) * 100
        print(
            f"[{exp_name}] Softmax comparisons: {softmax_comparisons:,} "
            f"({reduction_pct:.1f}% vs baseline)"
        )

    # --- Save results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment_metadata": {
            "name": exp_name,
            "timestamp": timestamp,
            "base_model": "DeepSeek-R1-Distill-Llama-8B",
            "training_config": {
                "epochs": cfg.epochs,
                "batch_size": cfg.per_device_train_bs,
                "accumulation_steps": cfg.grad_accum_steps,
                "learning_rate": cfg.lr,
                "warmup": cfg.warmup_ratio,
                "lora_r": cfg.lora_r,
                "lora_alpha": cfg.lora_alpha,
                "gradient_checkpointing": cfg.gradient_checkpointing,
            },
            "dataset_info": {
                "train_size": len(ds["train"]),
                "eval_size": len(ds["validation"]),
                "max_seq_len": seq_len,
                "seq_stats_train": train_seq_stats,
            },
            "environment": {
                "peak_memory_mb": peak_mem_mb,
                "bf16": True,
            },
            "model_config": extra_meta or {},
        },
        "performance_metrics": {
            "training_time_seconds": train_time,
            "peak_memory_mb": peak_mem_mb,
            "inference_latency_ms": inf_latency_ms,
            "softmax_comparisons": softmax_comparisons,
            "train": train_res.metrics,
            "eval": eval_res,
            "trajectory": traj_callback.trajectory,
        },
    }

    results_path = os.path.join(out_dir, f"results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[{exp_name}] Results saved to {results_path}")

    # Optionally save weights
    if cfg.save_weights:
        weights_dir = os.path.join(out_dir, "weights")
        trainer.save_model(weights_dir)
        tokenizer.save_pretrained(weights_dir)
        print(f"[{exp_name}] Weights saved to {weights_dir}")

    return results
