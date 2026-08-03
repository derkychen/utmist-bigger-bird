from dataclasses import dataclass

@dataclass
class LlamaTrainConfig:
    """Training config for R1-Distill-Llama-8B experiments.

    Defaults are tuned for a single 40GB H100 MIG slice with LoRA r=16.
    """
    epochs: int = 3
    per_device_train_bs: int = 1
    per_device_eval_bs: int = 2
    grad_accum_steps: int = 16
    lr: float = 2e-4          # LoRA needs ~10x higher LR than full FT
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    save_weights: bool = False