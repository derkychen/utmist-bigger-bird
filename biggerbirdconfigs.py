from transformers.models.big_bird.configuration_big_bird import (
    BigBirdConfig
)

class BiggerBirdConfig(BigBirdConfig):
    model_type = "bigger_bird"

    def __init__(
        self,
        patch_encoder_only=True,
        fragment_size=64,
        k_per_query=24,
        globals_per_head=6,
        r_target_softmax=0.12,
        min_k=48,
        max_k=48,
        teleports_per_head=3,
        teleport_bias_frac=0.4,
        w_mean=1.0,
        w_max=0.6,
        w_topk=0.4,
        w_std=0.2,
        topk_frac=0.2,
        keynorm_exponent=0.0,
        alpha_pos_prior=0.15,
        gamma_diversity=0.25,
        top_u=16,
        proto_count=24,
        mmr_prefilter_mult=3,
        mmr_diversity_steps=7,
        share_stride_layers=2,
        dense_fallback_under=512,
        random_selection=False,
        debug_collect=False,
        log_once_pairs=True,
        # Ablation flags: toggle components independently
        use_topk_mmr=True,       # False → full 3-block sliding window (BigBird baseline local)
        use_dynamic_globals=True, # False → no dynamic globals, only static first/last anchors
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.patch_encoder_only = patch_encoder_only
        self.fragment_size = fragment_size
        self.k_per_query = k_per_query
        self.globals_per_head = globals_per_head
        self.r_target_softmax = r_target_softmax
        self.min_k = min_k
        self.max_k = max_k
        self.teleports_per_head = teleports_per_head
        self.teleport_bias_frac = teleport_bias_frac
        self.w_mean = w_mean
        self.w_max = w_max
        self.w_topk = w_topk
        self.w_std = w_std
        self.topk_frac = topk_frac
        self.keynorm_exponent = keynorm_exponent
        self.alpha_pos_prior = alpha_pos_prior
        self.gamma_diversity = gamma_diversity
        self.top_u = top_u
        self.proto_count = proto_count
        self.mmr_prefilter_mult = mmr_prefilter_mult
        self.mmr_diversity_steps = mmr_diversity_steps
        self.share_stride_layers = share_stride_layers
        self.dense_fallback_under = dense_fallback_under
        self.random_selection = random_selection
        self.debug_collect = debug_collect
        self.log_once_pairs = log_once_pairs
        self.use_topk_mmr = use_topk_mmr
        self.use_dynamic_globals = use_dynamic_globals
    