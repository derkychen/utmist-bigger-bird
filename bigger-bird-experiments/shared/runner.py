import torch
import os
import time
import numpy as np
import psutil
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    default_data_collator,
)
from transformers.trainer_utils import EvalPrediction
import json
from datetime import datetime
from shared.patched_model import compute_dataset_seq_stats


def _reset_peak_memory():
    """Reset peak memory counters for the active accelerator."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _peak_memory_mb():
    """Return peak allocated memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        # Fallback: RSS of current process
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 ** 2)


def _compute_softmax_comparisons(seq_len, model, extra_meta):
    """Estimate total softmax comparisons across all layers/heads.

    Returns: int (total softmax key positions evaluated) or None if undetermined.
    Baseline reference = seq_len * seq_len * n_layers * n_heads.
    """
    # Read encoder shape from the model config so this is correct for both BART-base
    # (12 layers / 12 heads, IMDb) and the from-scratch LRA encoder (6 layers / 8 heads).
    cfg = getattr(model, "config", None)
    n_layers = getattr(cfg, "encoder_layers", None) or 12
    n_heads = getattr(cfg, "encoder_attention_heads", None) or 12
    base = seq_len * n_layers * n_heads
    meta = extra_meta or {}

    if meta.get("attention") == "full_dense" or model is None:
        return base * seq_len

    # Exp 5: Bigger Bird — local_k + num_globals + num_teleports
    if "local_k" in meta and "num_globals" in meta and "num_teleports" in meta:
        M = meta["local_k"] + meta["num_globals"] + meta["num_teleports"]
        return base * M

    # Exp 6: DeepSeek + PBS — same as top_k per query (sparse high-prec attention)
    if "top_k" in meta and "block_size" in meta and "num_blocks" in meta:
        return base * meta["top_k"]

    # Exp 7: Layer-adaptive — sum over layers using per-layer k schedule
    if "k_early" in meta and "k_mid" in meta and "k_late" in meta:
        ke, km, kl = meta["k_early"], meta["k_mid"], meta["k_late"]
        # Split layers into early / mid / late thirds (works for 12 or 6 layers).
        n_early = n_layers // 3
        n_late = n_layers // 3
        n_mid = n_layers - n_early - n_late
        per_layer_k = [ke] * n_early + [km] * n_mid + [kl] * n_late
        return seq_len * n_heads * sum(per_layer_k)

    # Exp 8: Token Drop — keep_ratio fraction after drop_after_layer
    if "drop_after_layer" in meta and "drop_ratio" in meta:
        dal = meta["drop_after_layer"]
        keep = 1.0 - meta["drop_ratio"]
        # Early layers: full attention. Late layers: attention over kept tokens.
        early = dal * n_heads * seq_len * seq_len
        late_len = int(seq_len * keep)
        late = (n_layers - dal) * n_heads * late_len * late_len
        return int(early + late)

    # Exp 13: Dynamic Context Window — fixed token budget after drop_after_layer
    if "drop_after_layer" in meta and "target_budget" in meta:
        dal = meta["drop_after_layer"]
        budget = meta["target_budget"]
        chunk_size = meta.get("chunk_size", 8192)
        # Early layers: if seq_len > chunk_size, chunks run independently
        if seq_len > chunk_size:
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            # Each chunk processes chunk_size tokens with full self-attention
            early = dal * n_heads * num_chunks * chunk_size * chunk_size
        else:
            early = dal * n_heads * seq_len * seq_len
        # Late layers: full attention over the fixed budget (always <= seq_len)
        late = (n_layers - dal) * n_heads * budget * budget
        return int(early + late)

    # Exp 14: Token Drop + DeepSeek Top-K — dense early, top-k late on shorter seq
    if "drop_after_layer" in meta and "drop_ratio" in meta and "top_k" in meta and "low_rank_dim" in meta:
        dal = meta["drop_after_layer"]
        keep = 1.0 - meta["drop_ratio"]
        top_k = meta["top_k"]
        early = dal * n_heads * seq_len * seq_len
        late_len = max(top_k, int(seq_len * keep))
        late = (n_layers - dal) * n_heads * late_len * top_k
        return int(early + late)

    # Exp 9: Attention Speculation — window + anchors per query
    if "window_size" in meta and "num_anchors" in meta:
        M = meta["window_size"] + meta["num_anchors"]
        return base * M

    # Exp 10: GQA + Sparse — same softmax count as top_k (GQA saves memory, not softmax)
    if "kv_groups" in meta and "top_k" in meta:
        return base * meta["top_k"]

    # Exp 1: DeepSeek Top-K
    if "top_k" in meta and "low_rank_dim" in meta and "block_size" not in meta:
        return base * meta["top_k"]

    # Exp 4: PBS — num_blocks * block_size per query
    if "block_size" in meta and "num_blocks" in meta:
        return base * meta["num_blocks"] * meta["block_size"]

    # Exp 3: Dynamic Globals — globals + window per query
    if "window_size" in meta and "num_globals" in meta:
        return base * (meta["num_globals"] + meta["window_size"])

    # Exp 2: Lightning Hybrid — local window only
    if "block_size" in meta:
        return base * meta["block_size"]

    return None


def _measure_inference_latency(model, tokenizer, device, seq_len=256, n_trials=10):
    """Measure average forward-pass latency (ms) on synthetic batch."""
    model.eval()
    dummy = tokenizer(
        "This is a test sentence for latency benchmarking. " * 50,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    dummy = {k: v.to(device) for k, v in dummy.items()}

    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            _ = model(**dummy)

    # Timed runs
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(n_trials):
                _ = model(**dummy)
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_trials):
                _ = model(**dummy)
        total_ms = (time.perf_counter() - t0) * 1000

    return total_ms / n_trials


def _measure_inference_latency_ids(model, device, seq_len, vocab_size, pair=False, n_trials=10):
    """Latency helper for LRA models (no HF tokenizer); builds synthetic id tensors."""
    model.eval()
    lo = 4  # skip special-token ids
    ids = torch.randint(lo, max(lo + 1, vocab_size), (1, seq_len), device=device)
    am = torch.ones(1, seq_len, dtype=torch.long, device=device)
    if pair:
        dummy = {
            "input_ids_a": ids, "attention_mask_a": am,
            "input_ids_b": ids.clone(), "attention_mask_b": am.clone(),
        }
    else:
        dummy = {"input_ids": ids, "attention_mask": am}

    with torch.no_grad():
        for _ in range(3):
            _ = model(**dummy)

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(n_trials):
                _ = model(**dummy)
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_trials):
                _ = model(**dummy)
        total_ms = (time.perf_counter() - t0) * 1000

    return total_ms / n_trials


@dataclass
class TrainConfig:
    epochs: int = 3
    per_device_train_bs: int = 2
    per_device_eval_bs: int = 2
    grad_accum_steps: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    use_cpu: bool = False
    torch_compile: bool = False

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_pred):
    if isinstance(eval_pred, EvalPrediction):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        preds, labels = eval_pred
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    # Binary tasks (IMDb, LRA text/retrieval) use binary F1; multiclass (LRA ListOps,
    # 10 classes) uses macro F1.
    n_classes = int(max(preds.max(initial=0), labels.max(initial=0))) + 1
    average = "binary" if n_classes <= 2 else "macro"
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average=average),
    }

def device_flags(force_cpu=False):
    if force_cpu:
        return False, False, False, False
    use_cuda = torch.cuda.is_available()
    use_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    fp16 = False
    bf16 = False
    torch_compile = False
    if use_cuda:
        # Only use bf16 when the GPU supports it *natively* (Ampere+, compute
        # capability >= 8.0). On pre-Ampere cards torch.cuda.is_bf16_supported()
        # can return True via slow emulation, which also makes torch.compile skip
        # bf16 nodes ("does not support bfloat16 compilation natively"). Prefer
        # fp16 there instead.
        major, _ = torch.cuda.get_device_capability()
        bf16 = major >= 8 and torch.cuda.is_bf16_supported()
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
    fp16, bf16, _torch_compile_default, use_mps = device_flags(force_cpu=cfg.use_cpu)
    eval_accum = 1 if use_mps else 8

    # torch.compile: opt-in via TrainConfig or env BIGGER_BIRD_TORCH_COMPILE=1.
    # Generates fused kernels across the whole graph (LayerNorm, FFN, attention
    # epilogues). Data-dependent gather/top-k paths will graph-break gracefully.
    env_compile = os.environ.get("BIGGER_BIRD_TORCH_COMPILE", "0").lower() in ("1", "true", "yes", "on")
    torch_compile = bool(cfg.torch_compile or env_compile) and not cfg.use_cpu
    if torch_compile:
        print(f"[{exp_name}] torch.compile: ENABLED")
    
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
        use_cpu=cfg.use_cpu,
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

    _reset_peak_memory()
    print(f"[{exp_name}] Starting training...", flush=True)
    start_time = time.time()
    train_res = trainer.train()
    train_time = time.time() - start_time
    peak_mem_mb = _peak_memory_mb()
    print(f"[{exp_name}] Peak memory: {peak_mem_mb:.1f} MB")

    print(f"[{exp_name}] Evaluating...", flush=True)
    eval_res = trainer.evaluate()

    # Sequence length stats (first row + distribution over train split)
    train_seq_stats = compute_dataset_seq_stats(ds["train"])
    seq_len = train_seq_stats["max_len"] or (
        ds["train"][0]["input_ids"].shape[0] if "input_ids" in ds["train"][0] else 256
    )

    # Inference latency
    device = next(model.parameters()).device
    try:
        inf_latency_ms = _measure_inference_latency(model, tokenizer, device, seq_len=seq_len, n_trials=10)
        print(f"[{exp_name}] Inference latency: {inf_latency_ms:.2f} ms/seq")
    except Exception as e:
        inf_latency_ms = None
        print(f"[{exp_name}] Inference latency measurement failed: {e}")

    # Softmax comparison count
    softmax_comparisons = _compute_softmax_comparisons(seq_len, model, extra_meta)
    if softmax_comparisons:
        baseline_comparisons = seq_len * seq_len * 12 * 12  # n² × heads × layers
        reduction_pct = (1 - softmax_comparisons / baseline_comparisons) * 100
        print(f"[{exp_name}] Softmax comparisons: {softmax_comparisons:,} ({reduction_pct:.1f}% vs baseline)")

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
                "max_seq_len": seq_len,
                "seq_stats_train": train_seq_stats,
                "fixed_length": (extra_meta or {}).get("fixed_length"),
            },
            "environment": {
                "use_mps": use_mps,
                "fp16": fp16,
                "peak_memory_mb": peak_mem_mb
            },
            "model_config": extra_meta or {}
        },
        "performance_metrics": {
            "training_time_seconds": train_time,
            "peak_memory_mb": peak_mem_mb,
            "inference_latency_ms": inf_latency_ms,
            "softmax_comparisons": softmax_comparisons,
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


def run_lra(
    task: str,
    exp_name: str,
    model,
    ds,
    cfg: TrainConfig,
    num_labels: int,
    seq_len: int,
    vocab_size: int,
    pair: bool = False,
    extra_meta: dict = None,
    callbacks=None,
    save_weights: bool = False,
    track: str = "lra",
):
    """Train/evaluate one long-context encoder experiment and write dashboard artifacts.

    Mirrors ``run_experiment`` but for the from-scratch encoder: data is already
    fixed-length integer ids (no HF tokenizer), so it uses the default data collator and
    a tensor-based latency probe. Artifacts land in ``benchmarks/<track>_<task>_<exp>/``
    in the same JSON schema the dashboard ingests, tagged with ``task`` and ``seq_length``.
    """
    fp16, bf16, _td, use_mps = device_flags(force_cpu=cfg.use_cpu)
    # The sparse-attention kernels use -1e9 masking sentinels that overflow fp16; for the
    # small from-scratch LRA encoder fp16 buys little and hurts stability, so train in fp32
    # (keep bf16 only where it is natively supported -- its range avoids the overflow).
    fp16 = False
    eval_accum = 1 if use_mps else 8
    env_compile = os.environ.get("BIGGER_BIRD_TORCH_COMPILE", "0").lower() in ("1", "true", "yes", "on")
    torch_compile = bool(cfg.torch_compile or env_compile) and not cfg.use_cpu

    bench_name = f"{track}_{task}_{exp_name}"
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks", bench_name))
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
        use_cpu=cfg.use_cpu,
        optim="adamw_torch",
        eval_accumulation_steps=eval_accum,
    )

    traj_callback = TrajectoryCallback()
    all_callbacks = [traj_callback] + (callbacks if callbacks else [])

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=all_callbacks,
    )

    _reset_peak_memory()
    print(f"[{bench_name}] Starting training...", flush=True)
    start_time = time.time()
    train_res = trainer.train()
    train_time = time.time() - start_time
    peak_mem_mb = _peak_memory_mb()
    print(f"[{bench_name}] Peak memory: {peak_mem_mb:.1f} MB")

    print(f"[{bench_name}] Evaluating...", flush=True)
    eval_res = trainer.evaluate()

    device = next(model.parameters()).device
    try:
        inf_latency_ms = _measure_inference_latency_ids(
            model, device, seq_len=seq_len, vocab_size=vocab_size, pair=pair, n_trials=10
        )
        print(f"[{bench_name}] Inference latency: {inf_latency_ms:.2f} ms/seq")
    except Exception as e:
        inf_latency_ms = None
        print(f"[{bench_name}] Inference latency measurement failed: {e}")

    softmax_comparisons = _compute_softmax_comparisons(seq_len, model, extra_meta)
    if softmax_comparisons:
        cfg_obj = getattr(model, "config", None)
        n_layers = getattr(cfg_obj, "encoder_layers", None) or 12
        n_heads = getattr(cfg_obj, "encoder_attention_heads", None) or 12
        baseline_comparisons = seq_len * seq_len * n_layers * n_heads
        reduction_pct = (1 - softmax_comparisons / baseline_comparisons) * 100
        print(f"[{bench_name}] Softmax comparisons: {softmax_comparisons:,} ({reduction_pct:.1f}% vs baseline)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment_metadata": {
            "name": exp_name,
            "task": f"{track}_{task}",
            "seq_length": seq_len,
            "timestamp": timestamp,
            "training_config": {
                "epochs": cfg.epochs,
                "batch_size": cfg.per_device_train_bs,
                "accumulation_steps": cfg.grad_accum_steps,
                "learning_rate": cfg.lr,
                "warmup": cfg.warmup_ratio,
            },
            "dataset_info": {
                "train_size": len(ds["train"]),
                "eval_size": len(ds["validation"]),
                "max_seq_len": seq_len,
                "num_labels": num_labels,
                "vocab_size": vocab_size,
                "fixed_length": True,
            },
            "environment": {
                "use_mps": use_mps,
                "fp16": fp16,
                "peak_memory_mb": peak_mem_mb,
            },
            "model_config": {**(extra_meta or {}), "task": f"{track}_{task}", "seq_length": seq_len},
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

    if save_weights:
        weights_dir = os.path.join(out_dir, f"weights_{timestamp}")
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(weights_dir, "model_state.pt"))
        results["experiment_metadata"]["weights_path"] = weights_dir
        print(f"[{bench_name}] Weights saved to {weights_dir}")

    json_path = os.path.join(out_dir, f"eval_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    res_path = os.path.join(out_dir, "results.txt")
    with open(res_path, "w") as f:
        f.write(f"Experiment: {bench_name} | Date: {timestamp}\n")
        f.write(f"Task: {track}_{task} | Seq: {seq_len}\n")
        f.write(f"Training Time: {train_time:.2f}s\n")
        f.write(f"Accuracy: {eval_res.get('eval_accuracy', 'N/A')}\n")
        f.write(f"F1: {eval_res.get('eval_f1', 'N/A')}\n")

    print(f"[{bench_name}] Results exported to {json_path}")
    return eval_res
