from dataclasses import dataclass

from transformers.models.big_bird.configuration_big_bird import (
    BigBirdConfig
)

@dataclass
class BiggerBirdConfig(BigBirdConfig):
    model_type = "bigger_bird"
    # Scope
    patch_encoder_only: bool = True

    # Candidate construction
    fragment_size: int = 64      # window size Fw — clamped to src_len
    k_per_query: int = 24         # locals picked from the window
    globals_per_head: int = 6    # g per head

    # Softmax target fraction (controls k on short sequences)
    r_target_softmax: float = 0.12
    min_k: int = 48
    max_k: int = 48

    # Teleports
    teleports_per_head: int = 3
    teleport_bias_frac: float = 0.4

    # Utility shaping for globals (facility-location proxy)
    w_mean: float = 1.0
    w_max: float = 0.6
    w_topk: float = 0.4
    w_std: float  = 0.2
    topk_frac: float = 0.2
    keynorm_exponent: float = 0.0

    # Scoring blend / locals
    alpha_pos_prior: float = 0.15
    gamma_diversity: float = 0.25   # diversity penalty

    # Globals (prefilter + prototypes)
    top_u: int = 16                 # per-head prefilter size U
    proto_count: int = 24          # query prototypes p (<= Tq)

    # Blocked-MMR params
    mmr_prefilter_mult: float = 3   # candidates = min(Fw, ceil(mult*k))
    mmr_diversity_steps: int = 7      # number of diversity rounds (≈MMR)

    # Amortization
    share_stride_layers: int = 2      # reuse globals across adjacent layers

    # Dense fallback threshold
    dense_fallback_under: int = 512   # if src_len <= this → use dense attention (super().forward)

    # Misc / Debug
    random_selection: bool = False
    debug_collect: bool = False
    log_once_pairs: bool = True