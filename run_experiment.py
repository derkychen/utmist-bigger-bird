#!/usr/bin/env python3
"""
Unified experiment runner with flexible compute configs.
Usage:
  python run_experiment.py --exp 3 --size small    # Quick test (1GB mem)
  python run_experiment.py --exp 3 --size medium   # Good GPU (8GB mem)  
  python run_experiment.py --exp 3 --size big      # Big GPU (24GB+ mem)
  python run_experiment.py --exp 3 --size big --epochs 5 --batch 4
  python run_experiment.py --exp 5 --size big
  python run_experiment.py --exp 6 --size small    
"""

import sys
import os
import argparse
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shared.dataset import build_imdb_dataset, DataConfig
from shared.runner import run_experiment, TrainConfig

# Import all experiment models
from exp_1_deepseek_topk.model import PatchedModel as DeepSeekModel
from exp_2_lightning_hybrid.model import PatchedModel as LightningModel
from exp_3_dynamic_globals.model import PatchedModel as DynamicGlobalsModel
from exp_4_pbs_attn.model import PatchedModel as PBSModel
from exp_5_bigger_bird.model import PatchedModel as BiggerBirdModel
from exp_6_deepseek_pbs.model import PatchedModel as DeepSeekPBSModel
from exp_7_layer_adaptive.model import PatchedModel as LayerAdaptiveModel
from exp_8_token_drop.model import PatchedModel as TokenDropModel
from exp_9_attn_specul.model import PatchedModel as AttnSpeculModel
from exp_10_gqa_sparse.model import PatchedModel as GQASparseModel
from exp_11_nsa.model import PatchedModel as NSAModel
from exp_12_s2_hhst.model import PatchedModel as S2HHSTModel
from exp_13_dynamic_context.model import PatchedModel as DynamicContextModel
from exp_14_token_drop_deepseek.model import PatchedModel as TokenDropDeepSeekModel

# Compute presets
COMPUTE_CONFIGS = {
    "small": {
        "train_samples": 500,
        "eval_samples": 100,
        "max_length": 256,
        "batch_size": 1,
        "grad_accum": 1,
        "epochs": 2,
        "desc": "Quick test / 8GB RAM / Fast iteration"
    },
    "medium": {
        "train_samples": 2000,
        "eval_samples": 400,
        "max_length": 512,
        "batch_size": 2,
        "grad_accum": 4,
        "epochs": 3,
        "desc": "Good GPU / 16GB VRAM / Solid training"
    },
    "big": {
        "train_samples": 6000,
        "eval_samples": 1000,
        "max_length": 768,
        "batch_size": 4,
        "grad_accum": 8,
        "epochs": 3,
        "desc": "Big GPU / 24GB+ VRAM / Full training"
    },
    "xl": {
        "train_samples": 25000,
        "eval_samples": 2500,
        "max_length": 1024,
        "batch_size": 8,
        "grad_accum": 16,
        "epochs": 5,
        "desc": "Full IMDb / Large GPU / Production"
    },
    "long": {
        "train_samples": 500,
        "eval_samples": 100,
        "max_length": 2048,
        "batch_size": 1,
        "grad_accum": 1,
        "epochs": 2,
        "desc": "Long-context stress test / Small samples / Fixed-length padding"
    }
}

def extend_position_embeddings(model, new_max_length):
    """Extend BART position embeddings to support sequences longer than 1024."""
    old_max = model.config.max_position_embeddings
    if new_max_length <= old_max:
        return
    
    model.config.max_position_embeddings = new_max_length
    offset = 2  # BART padding offset (indices 0,1 are padding)
    
    for embed in [model.model.encoder.embed_positions, model.model.decoder.embed_positions]:
        old_num = embed.weight.shape[0]  # Actual rows; don't assume formula
        new_num = new_max_length + offset + 1
        new_weight = embed.weight.new_zeros(new_num, embed.embedding_dim)
        new_weight[:old_num] = embed.weight.data
        
        # Tile learned positions [offset .. offset+old_max-1] cycling for new indices
        for i in range(old_num, new_num):
            src = offset + ((i - offset) % old_max)
            new_weight[i] = embed.weight.data[src]
        
        embed.num_embeddings = new_num
        embed.weight = nn.Parameter(new_weight)

EXPERIMENT_CONFIGS = {
    0: ("exp_0_baseline", None, {"attention": "full_dense"}),
    1: ("exp_1_deepseek_topk", DeepSeekModel, {"top_k": 64, "low_rank_dim": 16, "use_triton": True}),
    2: ("exp_2_lightning_hybrid", LightningModel, {"block_size": 128, "use_triton": True}),
    3: ("exp_3_dynamic_globals", DynamicGlobalsModel, {"window_size": 64, "num_globals": 16, "use_triton": True}),
    4: ("exp_4_pbs_attn", PBSModel, {"block_size": 64, "num_blocks": 2, "use_triton": True}),
    5: ("exp_5_bigger_bird", BiggerBirdModel, {"window_size": 64, "local_k": 32, "num_globals": 16, "num_teleports": 8, "diversity_lambda": 0.3, "teleport_bias": 0.5, "use_triton": True}),
    6: ("exp_6_deepseek_pbs", DeepSeekPBSModel, {"top_k": 64, "low_rank_dim": 16, "block_size": 32, "num_blocks": 4, "use_triton": True}),
    7: ("exp_7_layer_adaptive", LayerAdaptiveModel, {"k_early": 192, "k_mid": 64, "k_late": 32, "low_rank_dim": 16, "use_triton": True}),
    8: ("exp_8_token_drop", TokenDropModel, {"drop_after_layer": 3, "drop_ratio": 0.3, "use_triton": True}),
    9: ("exp_9_attn_specul", AttnSpeculModel, {"window_size": 64, "num_anchors": 4, "verify_every": 4, "verify_kl_weight": 0.1, "use_triton": True}),
    10: ("exp_10_gqa_sparse", GQASparseModel, {"kv_groups": 4, "top_k": 64, "low_rank_dim": 16, "use_triton": True}),
    11: ("exp_11_nsa", NSAModel, {"block_size": 32, "stride": 32, "topk_blocks": 4, "window_size": 128, "use_triton": True}),
    12: (
        "exp_12_s2_hhst",
        S2HHSTModel,
        {"shard_size": 32, "local_blocks": 2, "stride_blocks": 16, "use_sink": True, "dense_layers": [0]},
    ),
    13: (
        "exp_13_dynamic_context",
        DynamicContextModel,
        {"drop_after_layer": 3, "target_budget": 4096, "chunk_size": 8192},
    ),
    14: (
        "exp_14_token_drop_deepseek",
        TokenDropDeepSeekModel,
        {"drop_after_layer": 3, "drop_ratio": 0.3, "top_k": 64, "low_rank_dim": 16, "use_triton": True},
    ),
}

def main():
    parser = argparse.ArgumentParser(
        description="Run Bigger Bird experiments with flexible compute configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python run_experiment.py --exp 3 --size small
  
  # Full training on GPU  
  python run_experiment.py --exp 3 --size big
  
  # Custom override
  python run_experiment.py --exp 3 --size medium --epochs 5 --seq 768
  
  # List available configs
  python run_experiment.py --list
        """
    )
    
    parser.add_argument("--exp", type=int, choices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                       help="Experiment number (0=baseline, 1-4=original sparse methods, 5-10=new hybrid/advanced ideas, 11=NSA, 12=S2-HHST, 13=Dynamic Context)")
    parser.add_argument("--size", type=str, choices=["small", "medium", "big", "xl", "long"],
                       help="Compute size preset")
    parser.add_argument("--list", action="store_true",
                       help="List available compute presets")
    
    # Override flags
    parser.add_argument("--train-samples", type=int, help="Override training samples")
    parser.add_argument("--eval-samples", type=int, help="Override eval samples")
    parser.add_argument("--seq", type=int, help="Override sequence length")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--accum", type=int, help="Override gradient accumulation")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--fixed-length", action="store_true",
                       help="Pad all sequences to max_length (stress full context window)")
    parser.add_argument("--grad-checkpoint", action="store_true",
                       help="Enable gradient checkpointing (saves memory)")
    parser.add_argument("--save-weights", action="store_true",
                       help="Save model weights to benchmarks/<exp>/weights_<timestamp>/ for later eval")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU training (needed for seq>=2048 on Apple MPS due to buffer limits)")
    parser.add_argument("--compile", action="store_true",
                       help="Enable torch.compile for the training/eval graph (fused kernels)")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable compute presets:")
        print("=" * 70)
        for name, cfg in COMPUTE_CONFIGS.items():
            print(f"\n{name.upper()}: {cfg['desc']}")
            print(f"  Samples: {cfg['train_samples']} train / {cfg['eval_samples']} eval")
            print(f"  Seq length: {cfg['max_length']}")
            print(f"  Batch: {cfg['batch_size']} x {cfg['grad_accum']} accum = {cfg['batch_size'] * cfg['grad_accum']} effective")
            print(f"  Epochs: {cfg['epochs']}")
        print("=" * 70)
        print("\nExperiments:")
        for num, (name, _, params) in EXPERIMENT_CONFIGS.items():
            label = " [BASELINE]" if num == 0 else ""
            print(f"  {num}: {name}{label} ({params})")
        return
    
    if args.exp is None:
        parser.error("--exp is required (unless using --list)")
    if not args.size:
        parser.error("--size is required (unless using --list)")
    
    # Get compute config
    compute = COMPUTE_CONFIGS[args.size].copy()
    
    # Apply overrides
    if args.train_samples: compute["train_samples"] = args.train_samples
    if args.eval_samples: compute["eval_samples"] = args.eval_samples
    if args.seq: compute["max_length"] = args.seq
    if args.batch: compute["batch_size"] = args.batch
    if args.accum: compute["grad_accum"] = args.accum
    if args.epochs: compute["epochs"] = args.epochs
    
    # Get experiment config
    exp_name, ModelClass, model_params = EXPERIMENT_CONFIGS[args.exp]
    
    print(f"\n{'='*70}")
    print(f"Running: {exp_name}" + (" [BASELINE - full dense attention]" if args.exp == 0 else ""))
    print(f"Compute: {args.size.upper()} - {compute['desc']}")
    print(f"Config: {compute['train_samples']} samples, seq={compute['max_length']}, "
          f"batch={compute['batch_size']}x{compute['grad_accum']}, {compute['epochs']} epochs")
    print(f"{'='*70}\n")
    
    # Setup
    model_name = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    base_model.config.classifier_dropout = 0.1
    
    # Extend position embeddings if seq length exceeds BART's native 1024
    if compute["max_length"] > base_model.config.max_position_embeddings:
        print(f"Extending position embeddings from {base_model.config.max_position_embeddings} -> {compute['max_length']} tokens")
        extend_position_embeddings(base_model, compute["max_length"])
    
    # Enable gradient checkpointing if requested
    if args.grad_checkpoint:
        base_model.gradient_checkpointing_enable()
        print("Gradient checkpointing: ENABLED (saves ~30% memory)\n")
    
    # Baseline: use unpatched model directly
    if ModelClass is None:
        model = base_model
    else:
        model = ModelClass(base_model, **model_params)
    
    # Build dataset: use fixed-length padding for long-context stress tests
    use_fixed = args.fixed_length or args.size == "long"
    data_cfg = DataConfig(
        train_samples=compute["train_samples"],
        eval_samples=compute["eval_samples"],
        max_length=compute["max_length"]
    )
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=compute["max_length"] if use_fixed else None)
    
    # Training config
    train_cfg = TrainConfig(
        epochs=compute["epochs"],
        per_device_train_bs=compute["batch_size"],
        per_device_eval_bs=compute["batch_size"],
        grad_accum_steps=compute["grad_accum"],
        lr=args.lr,
        use_cpu=args.cpu,
        torch_compile=args.compile,
    )
    
    # Run
    run_experiment(
        exp_name,
        model,
        tokenizer,
        ds,
        train_cfg,
        extra_meta={
            **model_params,
            "compute_preset": args.size,
            "train_samples": compute["train_samples"],
            "seq_length": compute["max_length"],
            "fixed_length": use_fixed,
        },
        save_weights=args.save_weights,
    )

if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
