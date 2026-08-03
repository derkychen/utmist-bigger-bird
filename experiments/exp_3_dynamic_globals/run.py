import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from eval.imdb.dataset import build_imdb_dataset
from patches.original_patches.runner import run_experiment
from experiments.exp_3_dynamic_globals.model import PatchedModel

from experiments.exp_data_train_configs.original_patch_configs import load_data_config, load_train_config

def main():
    model_name = "facebook/bart-base"
    
    # 1. Load baseline model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    base_model.config.classifier_dropout = 0.1
    
    # 2. Patch with Dynamic Globals + Window
    # W=64 limits local block context. G=16 lets it broadcast the 16 most important tokens globally.
    model = PatchedModel(base_model, window_size=64, num_globals=16) 
    
    # 3. Build dataset (BIGGER RUN: full IMDb default config)
    data_cfg = load_data_config()
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=None)

    # 4. Run experiment (3 epochs to capture training trajectory)
    train_cfg = load_train_config()
    run_experiment(
        "exp_3_dynamic_globals", 
        model, 
        tokenizer, 
        ds, 
        train_cfg,
        extra_meta={"window_size": 64, "num_globals": 16}
    )

if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
