import math
import numpy as np
import torch
import torch.nn as nn

from transformers.models.big_bird.modeling_big_bird import (
    BigBirdBlockSparseAttention
) 

class BiggerBirdAttention(BigBirdBlockSparseAttention):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
    
    # copied in block_sparse_attention from bigbird to modify
    def bigbird_block_sparse_attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        band_mask,
        from_mask,
        to_mask,
        from_blocked_mask,
        to_blocked_mask,
        n_heads,
        n_rand_blocks,
        attention_head_size,
        from_block_size,
        to_block_size,
        batch_size,
        from_seq_len,
        to_seq_len,
        seed,
        plan_from_length,
        plan_num_rand_blocks,
    ):
        # BigBird block-sparse attention as suggested in paper

        # ITC:
        #     global tokens: 2 x block_size
        #     window tokens: 3 x block_size
        #     random tokens: num_rand_tokens x block_size

        # ETC:
        #     global tokens: extra_globals_tokens + 2 x block_size
        #     window tokens: 3 x block_size
        #     random tokens: num_rand_tokens x block_size

        # Note:
        #     1) Currently, ETC is not supported.
        #     2) Window size is fixed to 3 blocks & it can be changed only by
        #     changing `block_size`.
        #     3) Number of global blocks are fixed (2 blocks here) & global tokens can be
        #     controlled only by `block_size`.

        # attention is calculated separately for q[0], q[1], q[2:-2], q[-2], q[-1] in order to use special trick of shifting tokens (for calculating sliding attention)
        # hence following code can be divided into 5 parts.

        if from_seq_len // from_block_size != to_seq_len // to_block_size:
            raise ValueError("Error the number of blocks needs to be same!")

        rsqrt_d = 1 / math.sqrt(attention_head_size)
        bsz = batch_size
        attn_mask_penalty = -10000.0

        blocked_query_matrix = query_layer.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, -1)
        blocked_key_matrix = key_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
        blocked_value_matrix = value_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)

        if self.config.use_random_attn:
            # generate random attention and corresponding masks
            np.random.seed(seed)
            if from_seq_len in [1024, 3072, 4096]:  # old plans used in paper
                rand_attn = [
                    self._bigbird_block_rand_mask(
                        self.max_seqlen, self.max_seqlen, from_block_size, to_block_size, n_rand_blocks, last_idx=1024
                    )[: (from_seq_len // from_block_size - 2)]
                    for _ in range(n_heads)
                       
                ]
            else:
                if plan_from_length is None:
                    plan_from_length, plan_num_rand_blocks = self._get_rand_attn_plan(
                        from_seq_len, from_block_size, n_rand_blocks
                    )

                rand_attn = self._bigbird_block_rand_mask_with_head(
                    from_seq_length=from_seq_len,
                    to_seq_length=to_seq_len,
                    from_block_size=from_block_size,
                    to_block_size=to_block_size,
                    num_heads=n_heads,
                    plan_from_length=plan_from_length,
                    plan_num_rand_blocks=plan_num_rand_blocks,
                )

            rand_attn = np.stack(rand_attn, axis=0)
            rand_attn = torch.tensor(rand_attn, device=query_layer.device, dtype=torch.long)
            rand_attn.unsqueeze_(0)
            rand_attn = torch.cat([rand_attn for _ in range(batch_size)], dim=0)

            rand_mask = self._create_rand_mask_from_inputs(
                from_blocked_mask, to_blocked_mask, rand_attn, n_heads, n_rand_blocks, bsz, from_seq_len, from_block_size
            )

            # preparing block for randn attn
            gathered_key = self.torch_gather_b2(blocked_key_matrix, rand_attn)
            gathered_key = gathered_key.view(
                bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1
            )  # [bsz, n_heads, to_seq_len//to_block_size-2, n_rand_blocks, to_block_size, -1]
            gathered_value = self.torch_gather_b2(blocked_value_matrix, rand_attn)
            gathered_value = gathered_value.view(
                bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1
            )  # [bsz, n_heads, to_seq_len//to_block_size-2, n_rand_blocks, to_block_size, -1]

        # 1st PART
        # 1st block (global block) attention scores
        # q[0] x (k[0], k[1], k[2], k[3], k[4] .... )

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, to_seq_len]
        first_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 0], key_layer, ndim=4)

        first_product = first_product * rsqrt_d
        first_product += (1.0 - to_mask) * attn_mask_penalty
        first_attn_weights = nn.functional.softmax(
            first_product, dim=-1
        )  # [bsz, n_heads, from_block_size, to_seq_len]

        # [bsz, n_heads, from_block_size, to_seq_len] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, -1]
        first_context_layer = self.torch_bmm_nd(first_attn_weights, value_layer, ndim=4)
        first_context_layer.unsqueeze_(2)

        # 2nd PART
        # 2nd block attention scores
        # q[1] x (sliding_keys, random_keys, global_keys)
        # sliding key blocks -> 2nd, 3rd blocks
        # global key blocks -> 1st block

        # Added ablation: use_random_attn flag=False → no random attention, only sliding + global

        second_key_list = [
            blocked_key_matrix[:, :, 0],
            blocked_key_matrix[:, :, 1],
            blocked_key_matrix[:, :, 2],
            blocked_key_matrix[:, :, -1],
        ]

        second_value_list = [
            blocked_value_matrix[:, :, 0],
            blocked_value_matrix[:, :, 1],
            blocked_value_matrix[:, :, 2],
            blocked_value_matrix[:, :, -1],
        ]

        if self.config.use_random_attn:
            second_key_list.append(gathered_key[:, :, 0])
            second_value_list.append(gathered_value[:, :, 0])

        second_key_mat = torch.cat(
            second_key_list,
            dim=2,
        )
        second_value_mat = torch.cat(
            second_value_list,
            dim=2,
        )

        
        second_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 1], second_key_mat, ndim=4)

        second_seq_pad_list = [
            to_mask[:, :, :, : 3 * to_block_size],
            to_mask[:, :, :, -to_block_size:]
        ]
        second_rand_pad_list = [
            torch.ones((bsz, n_heads, from_block_size, 4 * to_block_size), device=blocked_query_matrix.device)
        ]
        if self.config.use_random_attn:
            second_seq_pad_list.append(to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size]))
            second_rand_pad_list.append(rand_mask[:, :, 0])

        second_seq_pad = torch.cat(
            second_seq_pad_list,
            dim=3,
        )
        second_rand_pad = torch.cat(
            second_rand_pad_list,
            dim=3,
        )

        second_product = second_product * rsqrt_d
        second_product += (1.0 - torch.minimum(second_seq_pad, second_rand_pad)) * attn_mask_penalty
        second_attn_weights = nn.functional.softmax(
            second_product, dim=-1
        )  # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]

        second_context_layer = self.torch_bmm_nd(second_attn_weights, second_value_mat, ndim=4)

        second_context_layer.unsqueeze_(2)

        # 3rd PART: BiggerBird middle-block attention
        #
        # Original BigBird computes attention for middle query blocks q[2:-2] over:
        #   1) local sliding-window keys from neighboring blocks,
        #   2) random key blocks,
        #   3) static global anchor blocks: first and last blocks.
        #
        # Current BiggerBird modifies only the middle-block path:
        #   1) local sliding-window keys are reduced with content-aware TopK/MMR,
        #   2) extra content-aware global tokens are selected with a facility-location
        #      style selector over query prototypes,
        #   3) original first/last block anchors are kept as static global anchors,
        #   4) original BigBird random blocks are still kept.
        #
        # Final middle attention attends over (when both ablation flags are True):
        #   [dynamic_globals, first_block_anchor, MMR_selected_locals, random_blocks, last_block_anchor]

        # Build the 3-block local sliding window for each middle block.
        # For middle block i (0-indexed within the middle range), its local window consists of
        # its left neighbor (i-1), itself (i), and its right neighbor (i+1) in the full blocked
        # sequence. The three offset slices [1:-3], [2:-2], [3:-1] pick those three neighbors
        # for all middle blocks simultaneously and concatenate them along the key dimension,
        # producing a single [n_mid, 3*to_block_size] key/value matrix per batch and head.
        exp_blocked_key_matrix = torch.cat(
            [blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2], blocked_key_matrix[:, :, 3:-1]], dim=3
        )  # [bsz, n_heads, n_mid, 3*to_block_size, d]
        exp_blocked_value_matrix = torch.cat(
            [blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2], blocked_value_matrix[:, :, 3:-1]],
            dim=3,
        )  # [bsz, n_heads, n_mid, 3*to_block_size, d]
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2]  # [bsz, n_heads, n_mid, block_size, d]

        # --- Ablation: dynamic global token selection ---
        # use_dynamic_globals=True  → select g_eff content-aware globals via facility-location
        # use_dynamic_globals=False → g_eff=0, only static first/last block anchors remain
        g_eff = int(self.config.globals_per_head) # effective number of dynamic globals that will be attended to

        if self.config.use_dynamic_globals:
            global_idx = self.select_global_tokens(
                query=middle_query_matrix,
                key_layer=key_layer,
                to_mask=to_mask,
                g=g_eff,
                to_block_size=to_block_size,
                exclude_static_global_blocks=True
            )  # [bsz, n_heads, g_eff]

            g_eff = global_idx.size(-1)
            d_model = key_layer.shape[-1]
            global_idx_exp = global_idx.unsqueeze(-1).expand(-1, -1, -1, d_model)

            selected_global_keys = torch.gather(
                key_layer, dim=2, index=global_idx_exp,
            )  # [bsz, n_heads, g_eff, d]
            selected_global_values = torch.gather(
                value_layer, dim=2, index=global_idx_exp,
            )  # [bsz, n_heads, g_eff, d]
        else:
            g_eff = 0
            selected_global_keys = None
            selected_global_values = None

        # --- Ablation: local token selection ---
        # use_topk_mmr=True  → TopK/MMR-selected local tokens (k_to_select ≤ max_k)
        # use_topk_mmr=False → full 3-block sliding window, original BigBird local attention
        Fw = exp_blocked_key_matrix.shape[3]
        band_mask_expanded = band_mask.expand(-1, n_heads, -1, -1, -1)

        if self.config.use_topk_mmr:
            k_to_select = min(self.config.max_k, Fw)

            # Collapse the query-token dimension to get a per-key validity mask.
            # band_mask_expanded is [B, H, n_mid, block_size, Fw]: a key position is
            # considered reachable for a middle block if any query token in that block
            # can attend to it according to the band mask.
            local_valid_mask = band_mask_expanded.any(dim=-2)  # [B, H, n_mid, Fw]

            local_prior = self.build_local_positional_prior(
                Fw=Fw,
                device=middle_query_matrix.device,
                dtype=middle_query_matrix.dtype,
            ).view(1, 1, 1, Fw)

            # Returns indices of the k_to_select best local keys per middle block,
            # chosen to balance relevance (cosine similarity to the query block) with
            # diversity (MMR penalty against already-selected tokens).
            mmr_idx = self.top_k_mmr(
                query=middle_query_matrix,
                keys=exp_blocked_key_matrix,
                k=k_to_select,
                valid_mask=local_valid_mask,
                prior=local_prior,
            )  # [bsz, n_heads, n_mid, k_to_select]

            d = exp_blocked_key_matrix.shape[-1]
            mmr_idx_exp = mmr_idx.unsqueeze(-1).expand(-1, -1, -1, -1, d)

            selected_local_keys = torch.gather(
                exp_blocked_key_matrix, dim=3, index=mmr_idx_exp,
            )  # [bsz, n_heads, n_mid, k_to_select, d]
            selected_local_values = torch.gather(
                exp_blocked_value_matrix, dim=3, index=mmr_idx_exp,
            )  # [bsz, n_heads, n_mid, k_to_select, d]

            inner_band_product = self.torch_bmm_nd_transpose(
                middle_query_matrix, selected_local_keys, ndim=5,
            )
            inner_band_product = inner_band_product * rsqrt_d

            # Re-gather the band mask at the MMR-selected positions so padding penalties
            # are applied per query token, not just per block. The block-level valid_mask
            # above was enough to exclude padding from selection; this finer mask handles
            # the case where different query tokens within the same block have different
            # reachability for the same key (e.g., near sequence boundaries).
            selected_local_mask = torch.gather(
                band_mask_expanded,
                dim=-1,
                index=mmr_idx.unsqueeze(3).expand(-1, -1, -1, from_block_size, -1),
            )  # [bsz, n_heads, n_mid, block_size, k_to_select]
            inner_band_product += (1.0 - selected_local_mask) * attn_mask_penalty
        else:
            # Full sliding window — identical to original BigBird middle-block local path.
            # band_mask_expanded is already [B, H, n_mid, block_size, Fw], matching
            # the shape of the full inner_band_product, so it can be applied directly.
            k_to_select = Fw
            selected_local_values = exp_blocked_value_matrix

            inner_band_product = self.torch_bmm_nd_transpose(
                middle_query_matrix, exp_blocked_key_matrix, ndim=5,
            )
            inner_band_product = inner_band_product * rsqrt_d
            inner_band_product += (1.0 - band_mask_expanded) * attn_mask_penalty

        # randn attention scores for q[-2:2] with ablation
        if self.config.use_random_attn:
            rand_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, gathered_key[:, :, 1:-1], ndim=5)
            rand_band_product = rand_band_product * rsqrt_d

        # Including 1st block (since it's global)
        first_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, 0]
        )
        first_band_product = first_band_product * rsqrt_d

        # Including last block (since it's global)
        last_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, -1]
        )
        last_band_product = last_band_product * rsqrt_d

        first_band_product += (1.0 - to_mask[:, :, :, :to_block_size].unsqueeze(3)) * attn_mask_penalty
        last_band_product += (1.0 - to_mask[:, :, :, -to_block_size:].unsqueeze(3)) * attn_mask_penalty
        if self.config.use_random_attn:
            rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * attn_mask_penalty

        # Concatenate all attention score contributions into a single logit vector per
        # query token, then softmax jointly so probabilities sum to 1 across all attended
        # positions. Dynamic globals are prepended only when the ablation flag is on;
        # the static first/last anchors and randoms are always included.
        band_product_list = [
            first_band_product,
            inner_band_product,
            last_band_product
        ]
        if self.config.use_random_attn:
            band_product_list.insert(2, rand_band_product)
        
        if self.config.use_dynamic_globals:
            global_band_product = torch.einsum(
                "bhlqd,bhgd->bhlqg", middle_query_matrix, selected_global_keys,
            )  # [bsz, n_heads, n_mid, block_size, g_eff]
            global_band_product = global_band_product * rsqrt_d
            band_product_list.insert(0, global_band_product)
        
        band_product = torch.cat(band_product_list, dim = -1)

        attn_weights = nn.functional.softmax(band_product, dim=-1)

        # Track slice boundaries with an offset accumulator rather than hardcoded indices
        # so the slices stay correct regardless of which ablation flags are active
        # (g_eff=0 when use_global_selection=False, k_to_select=Fw when use_topk_mmr=False).
        offset = 0

        if self.config.use_dynamic_globals:
            global_end = offset + g_eff
            offset = global_end

        first_end = offset + to_block_size
        offset = first_end

        local_end = offset + k_to_select
        offset = local_end

        if self.config.use_random_attn:
            rand_end = offset + n_rand_blocks * to_block_size
            offset = rand_end

        last_end = offset + to_block_size

        # Reconstruct the context vector by accumulating weighted value contributions from
        # each group of attended positions. This is equivalent to a single matmul of the
        # full attn_weights against a concatenated value matrix, but kept split to avoid
        # materialising a large padded value tensor when group sizes differ (e.g. g_eff vs Fw).
        if self.config.use_dynamic_globals:
            context_layer = torch.einsum(
                "bhlqg,bhgd->bhlqd",
                attn_weights[:, :, :, :, :global_end],
                selected_global_values,
            )
            context_layer += torch.einsum(
                "bhlqk,bhkd->bhlqd",
                attn_weights[:, :, :, :, global_end:first_end],
                blocked_value_matrix[:, :, 0],
            )
        else:
            context_layer = torch.einsum(
                "bhlqk,bhkd->bhlqd",
                attn_weights[:, :, :, :, :first_end],
                blocked_value_matrix[:, :, 0],
            )

        # Local (MMR-selected or full sliding window) values
        context_layer += self.torch_bmm_nd(
            attn_weights[:, :, :, :, first_end:local_end],
            selected_local_values,
            ndim=5,
        )

        if self.config.use_random_attn:
            # Random block values
            context_layer += self.torch_bmm_nd(
                attn_weights[:, :, :, :, local_end:rand_end],
                gathered_value[:, :, 1:-1],
                ndim=5,
            )

            # Static last-block anchor values
            context_layer += torch.einsum(
                "bhlqk,bhkd->bhlqd",
                attn_weights[:, :, :, :, rand_end:last_end],
                blocked_value_matrix[:, :, -1],
            )
        else:
            # Static last-block anchor values (no randoms)
            context_layer += torch.einsum(
                "bhlqk,bhkd->bhlqd",
                attn_weights[:, :, :, :, local_end:last_end],
                blocked_value_matrix[:, :, -1],
            )

        # 4th PART
        # last 2nd token attention scores
        # q[-2] x (sliding_keys, random_keys, global_keys)
        # sliding key blocks -> last 3 blocks
        # global key block -> 1st block
        # random key block -> based on indices stored in `randn_attn`

        second_last_key_list = [
            blocked_key_matrix[:, :, 0],
            blocked_key_matrix[:, :, -3],
            blocked_key_matrix[:, :, -2],
            blocked_key_matrix[:, :, -1],
        ]
        second_last_value_list = [
            blocked_value_matrix[:, :, 0],
            blocked_value_matrix[:, :, -3],
            blocked_value_matrix[:, :, -2],
            blocked_value_matrix[:, :, -1],
        ]
        if self.config.use_random_attn:
            second_last_key_list.append(gathered_key[:, :, -1])
            second_last_value_list.append(gathered_value[:, :, -1])

        second_last_key_mat = torch.cat(
            second_last_key_list,
            dim=2,
        )
        second_last_value_mat = torch.cat(
            second_last_value_list,
            dim=2,
        )

        
        second_last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -2], second_last_key_mat, ndim=4)

        second_last_seq_pad_list = [
            to_mask[:, :, :, :to_block_size],
            to_mask[:, :, :, -3 * to_block_size :],
        ]
        second_last_rand_pad_list = [
            torch.ones((bsz, n_heads, from_block_size, 4 * to_block_size), device=blocked_query_matrix.device)
        ]
        if self.config.use_random_attn:
            second_last_seq_pad_list.append(to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size]))
            second_last_rand_pad_list.append(rand_mask[:, :, -1])
        
        second_last_seq_pad = torch.cat(
            second_last_seq_pad_list,
            dim=3,
        )
        second_last_rand_pad = torch.cat(
            second_last_rand_pad_list,
            dim=3,
        )
        second_last_product = second_last_product * rsqrt_d
        second_last_product += (1.0 - torch.minimum(second_last_seq_pad, second_last_rand_pad)) * attn_mask_penalty
        second_last_attn_weights = nn.functional.softmax(
            second_last_product, dim=-1
        ) 

        second_last_context_layer = self.torch_bmm_nd(second_last_attn_weights, second_last_value_mat, ndim=4)
        second_last_context_layer.unsqueeze_(2)

        # 5th PART
        # last block (global) attention scores
        # q[-1] x (k[0], k[1], k[2], k[3], .... )

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, to_seq_len]
        last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -1], key_layer, ndim=4)
        last_product = last_product * rsqrt_d
        last_product += (1.0 - to_mask) * attn_mask_penalty
        last_attn_weights = nn.functional.softmax(last_product, dim=-1)  # [bsz, n_heads, from_block_size, n]

        # [bsz, n_heads, from_block_size, to_seq_len] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, -1]
        last_context_layer = self.torch_bmm_nd(last_attn_weights, value_layer, ndim=4)
        last_context_layer.unsqueeze_(2)

        # combining representations of all tokens
        context_layer = torch.cat(
            [first_context_layer, second_context_layer, context_layer, second_last_context_layer, last_context_layer],
            dim=2,
        )
        context_layer = context_layer.view((bsz, n_heads, from_seq_len, -1)) * from_mask
        context_layer = torch.transpose(context_layer, 1, 2)


        return context_layer, None

    @staticmethod
    def F_normalize_safe(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=dim, eps=eps)

    @staticmethod
    def build_local_positional_prior(
        Fw: int,
        device: torch.device,
        dtype: torch.dtype,
        tau: float | None = None,
    ) -> torch.Tensor:
        """
        Center-biased prior over the local 3-block window.

        Fw = 3 * to_block_size in the current BigBird middle-block path.
        The center of this window corresponds roughly to the current query block,
        while the left/right sides correspond to neighboring blocks.
        """
        if tau is None:
            tau = max(Fw / 4.0, 1.0)

        positions = torch.arange(Fw, device=device, dtype=dtype)
        center = torch.tensor((Fw - 1) / 2.0, device=device, dtype=dtype)
        distance = (positions - center).abs()

        return torch.exp(-distance / tau)




    def top_k_mmr(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        k: int,
        valid_mask: torch.Tensor | None,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor: # returns [bsz, n_heads, num_middle_blocks, k] — indices into Fw
        """
        Two-phase blocked MMR over a sliding window of size Fw.
        Phase 1: topK prefilter reduces Fw candidates to Kc = ceil(mmr_prefilter_mult * k).
        Phase 2: mmr_diversity_steps diversity rounds on the candidate set, then greedy fill.
        All hyperparameters drawn from self.config.
        """
        bsz, n_heads, n_mid, block_size, d = query.shape
        Fw = keys.shape[3]
        device = query.device

        # normalize for cosine similarity
        q_norm = self.F_normalize_safe(query, dim=-1)   # [bsz, n_heads, n_mid, block_size, d]
        k_norm = self.F_normalize_safe(keys, dim=-1)   # [bsz, n_heads, n_mid, Fw, d]

        # mean relevance over the block_size query tokens -> one score per candidate
        # [bsz, n_heads, n_mid, Fw]
        relevance = torch.einsum('bnmqd,bnmkd->bnmk', q_norm, k_norm) / block_size

        # Add positional prior before TopK prefiltering.
        # This softly prefers candidates near the center of the local 3-block window.
        if prior is not None and self.config.alpha_pos_prior != 0.0:
            relevance = relevance + self.config.alpha_pos_prior * prior.to(
                device=relevance.device,
                dtype=relevance.dtype,
            )

        # mask padding tokens to -inf before prefilter so they can never enter the candidate set
        if valid_mask is not None:
            relevance = relevance.masked_fill(~valid_mask.bool(), torch.finfo(relevance.dtype).min)

        # PHASE 1: topK prefilter
        # reduce from Fw to Kc = min(Fw, ceil(mmr_prefilter_mult * k)) candidates
        Kc = min(
            Fw,
            max(k, int(np.ceil(self.config.mmr_prefilter_mult * k)))
        )
        cand_vals, cand_idx = torch.topk(relevance, k=Kc, dim=-1)  # [bsz, n_heads, n_mid, Kc]

        # gather candidate key vectors for use in diversity penalty
        # [bsz, n_heads, n_mid, Kc, d]
        cand_keys = torch.gather(
            k_norm,
            dim=3,
            index=cand_idx.unsqueeze(-1).expand(-1, -1, -1, -1, d),
        )

        # PHASE 2: diversity rounds
        # run 1 + mmr_diversity_steps iterations; remaining slots filled greedily
        # penalty from each round carries forward into `remaining` for all subsequent selections
        total_steps = min(k, 1 + int(self.config.mmr_diversity_steps))
        selected = torch.zeros(bsz, n_heads, n_mid, k, dtype=torch.long, device=device)
        remaining = cand_vals.clone()  # [bsz, n_heads, n_mid, Kc]

        for r in range(total_steps):
            # pick highest scoring remaining candidate (index within candidate set 0..Kc-1)
            j = remaining.argmax(dim=-1)                                        # [bsz, n_heads, n_mid]

            # map candidate-set index back to original Fw position and store
            selected[..., r] = torch.gather(cand_idx, -1, j.unsqueeze(-1)).squeeze(-1)

            if r < total_steps - 1:
                # pull the key vector of the just-selected token: [bsz, n_heads, n_mid, d]
                sel_vec = torch.gather(
                    cand_keys,
                    dim=3,
                    index=j.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, d),
                ).squeeze(3)

                # cosine similarity of every candidate to the just-selected token: [bsz, n_heads, n_mid, Kc]
                # clamp to 0 so anti-correlated tokens receive no diversity bonus
                cos = (cand_keys * sel_vec.unsqueeze(3)).sum(-1).clamp(min=0)

                # penalize candidates similar to what was just selected, scaled by gamma_diversity
                remaining = remaining - self.config.gamma_diversity * cos

                # mask the selected position so it cannot be picked again
                remaining.scatter_(-1, j.unsqueeze(-1), -1e9)

        # greedy fill for remaining k - total_steps slots
        # remaining scores already reflect diversity penalties from the rounds above
        if k > total_steps:
            _, fill_pos = torch.topk(remaining, k=k - total_steps, dim=-1)     # [bsz, n_heads, n_mid, k-total_steps]
            selected[..., total_steps:] = torch.gather(cand_idx, -1, fill_pos)

        return selected  # [bsz, n_heads, n_mid, k]
    

    def select_global_tokens(
        self,
        query: torch.Tensor,       # [B, H, n_mid, block_size, d]
        key_layer: torch.Tensor,   # [B, H, T, d]
        to_mask: torch.Tensor,     # [B, 1, 1, T]
        g: int,
        to_block_size: int | None = None,
        exclude_static_global_blocks: bool = True,
    ) -> torch.Tensor:
        """
        Select g content-aware global token indices per head using greedy facility-location.

        The goal is to pick g key tokens that collectively cover as much of the query
        space as possible. Coverage is measured by how similar each candidate key is to
        a set of query prototypes sampled evenly across the middle sequence. A two-stage
        pipeline is used: a utility-score prefilter narrows the field to top_u candidates,
        then greedy submodular maximisation picks the final g tokens.

        Returns:
            global_idx: [B, H, g] absolute token positions into T
        """
        B, H, n_mid, q_block, d = query.shape
        T = key_layer.shape[2]
        device = query.device

        # Normalize so all similarity scores are cosine similarities in [-1, 1].
        K = self.F_normalize_safe(key_layer, dim=-1)  # [B, H, T, d]

        # Flatten all middle-block query tokens, then sample p evenly-spaced prototypes.
        # Using a small fixed-size prototype set instead of all query tokens keeps the
        # similarity matrix S tractable while still capturing the diversity of the sequence.
        Q = query.reshape(B, H, n_mid * q_block, d)
        Q = self.F_normalize_safe(Q, dim=-1)  # [B, H, n_mid*block_size, d]

        p = min(int(self.config.proto_count), Q.shape[2])
        if p <= 0:
            p = 1

        proto_idx = torch.round(
            torch.linspace(0, Q.shape[2] - 1, steps=p, device=device)
        ).long()

        Qp = Q.index_select(2, proto_idx)  # [B, H, p, d]

        # S[b, h, t, i] = cosine similarity of key token t to query prototype i.
        # relu clamps negative similarities to 0: tokens that are semantically
        # opposite to a prototype contribute nothing to coverage.
        S = torch.relu(torch.einsum("bhtd,bhpd->bhtp", K, Qp))  # [B, H, T, p]

        # Build a validity mask that excludes padding tokens and, optionally, the
        # first and last blocks (which are already used as static global anchors).
        if to_mask is not None:
            valid = to_mask.squeeze(1).squeeze(1).bool()  # [B, T]
        else:
            valid = torch.ones(B, T, device=device, dtype=torch.bool)

        if exclude_static_global_blocks and to_block_size is not None:
            valid = valid.clone()
            valid[:, :to_block_size] = False
            valid[:, -to_block_size:] = False

        S = S.masked_fill(~valid[:, None, :, None], 0.0)

        # Compute a scalar utility score per key token by combining four statistics of
        # its similarity profile across all query prototypes:
        #   mean     — rewards tokens that are broadly relevant to the whole sequence
        #   max      — rewards tokens that are highly relevant to at least one prototype
        #   topk_mean — a blend of the two: mean over the top-kq prototype similarities
        #   std      — rewards tokens that are distinctive (high variance across prototypes),
        #              acting as a light diversity signal within the prefilter stage
        mean = S.mean(dim=-1)           # [B, H, T]
        mx = S.max(dim=-1).values       # [B, H, T]

        kq = max(1, int(round(p * self.config.topk_frac)))
        topk_mean = torch.topk(S, k=kq, dim=-1).values.mean(dim=-1)  # [B, H, T]

        std = S.std(dim=-1)             # [B, H, T]

        utility = (
            self.config.w_mean  * mean
            + self.config.w_max   * mx
            + self.config.w_topk  * topk_mean
            + self.config.w_std   * std
        )  # [B, H, T]

        utility = utility.masked_fill(~valid[:, None, :], torch.finfo(utility.dtype).min)

        # Prefilter to the top_u highest-utility candidates before running the greedy loop.
        # This keeps the per-round gain computation O(top_u * p) instead of O(T * p),
        # which matters when T is large (e.g. 4096 tokens).
        U = min(int(self.config.top_u), T)
        _, top_idx = torch.topk(utility, k=U, dim=-1)  # [B, H, U]

        S_sub = torch.gather(
            S,
            dim=2,
            index=top_idx.unsqueeze(-1).expand(-1, -1, -1, p),
        )  # [B, H, U, p]

        # Greedy facility-location: iteratively pick the candidate that maximises the
        # marginal gain in coverage. `covered[b, h, i]` tracks the maximum similarity
        # to prototype i seen so far across all already-selected tokens. The gain of
        # adding a new candidate is the sum of improvements over current coverage
        # (relu clamps negative improvements to zero — we never reduce coverage).
        g = min(g, U)
        covered = torch.zeros(B, H, p, device=device, dtype=S.dtype)   # [B, H, p]
        blocked = torch.zeros(B, H, U, device=device, dtype=torch.bool) # tracks selected candidates
        chosen_local = torch.zeros(B, H, g, device=device, dtype=torch.long)

        for r in range(g):
            gains = torch.relu(S_sub - covered.unsqueeze(2)).sum(dim=-1)  # [B, H, U]
            gains = gains.masked_fill(blocked, torch.finfo(gains.dtype).min)

            j = gains.argmax(dim=-1)  # [B, H] — index within the top-U candidate set
            chosen_local[:, :, r] = j

            blocked.scatter_(2, j.unsqueeze(-1), True)

            # Update covered: element-wise max so coverage can only grow.
            selected_cover = torch.gather(
                S_sub,
                dim=2,
                index=j.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, p),
            ).squeeze(2)  # [B, H, p]

            covered = torch.maximum(covered, selected_cover)

        # Map chosen indices from the top-U local space back to absolute token positions.
        global_idx = torch.gather(top_idx, dim=2, index=chosen_local)  # [B, H, g]
        return global_idx