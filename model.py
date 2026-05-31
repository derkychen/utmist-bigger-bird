import math
import numpy as np
import torch
import torch.nn as nn

from transformers.models.big_bird.modeling_big_bird import (
    BigBirdBlockSparseAttention
)

from biggerbirdconfigs import BiggerBirdConfig

bigger_bird_config = BiggerBirdConfig(
    fragment_size=128,          # slightly tighter window → cleaner local top-k
    r_target_softmax=0.16,      # ensures k hits max_k at 896 tokens
    min_k=56,
    max_k=64,                   # locals per query (main quality driver)
    globals_per_head=6,
    teleports_per_head=4,       # a tiny bump helps long-range without much cost
    teleport_bias_frac=0.75,

    top_u=32,
    proto_count=48,

    mmr_prefilter_mult=3.0,
    mmr_diversity_steps=2,      # ↓ from 7 → less over-diversification, higher precision
    gamma_diversity=0.16,       # moderate penalty works best with steps=2

    alpha_pos_prior=0.12,       # restore a useful locality bias for IMDB
    share_stride_layers=2,

    dense_fallback_under=512,
    random_selection=False,
    debug_collect=False,
    log_once_pairs=True
)

class BiggerBirdAttention(BigBirdBlockSparseAttention):
    def __init__(self, config):
        super().__init__(config)
    
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

        blocked_query_matrix = query_layer.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, -1)
        blocked_key_matrix = key_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
        blocked_value_matrix = value_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)

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

        second_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, 1],
                blocked_key_matrix[:, :, 2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, 0],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1]
        second_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, 1],
                blocked_value_matrix[:, :, 2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, 0],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1]

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]
        second_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 1], second_key_mat, ndim=4)
        second_seq_pad = torch.cat(
            [
                to_mask[:, :, :, : 3 * to_block_size],
                to_mask[:, :, :, -to_block_size:],
                to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size]),
            ],
            dim=3,
        )
        second_rand_pad = torch.cat(
            [
                rand_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]),
                rand_mask[:, :, 0],
            ],
            dim=3,
        )
        second_product = second_product * rsqrt_d
        second_product += (1.0 - torch.minimum(second_seq_pad, second_rand_pad)) * attn_mask_penalty
        second_attn_weights = nn.functional.softmax(
            second_product, dim=-1
        )  # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]

        # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, -1]
        second_context_layer = self.torch_bmm_nd(second_attn_weights, second_value_mat, ndim=4)

        second_context_layer.unsqueeze_(2)

        # 3rd PART
        # Middle blocks attention scores
        # q[-2:2] x (sliding_keys, random_keys, global_keys)
        # sliding attn is calculated using special trick of shifting tokens as discussed in paper
        # random keys are generated by taking random indices as per `rand_attn`
        # global keys -> 1st & last block

        exp_blocked_key_matrix = torch.cat(
            [blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2], blocked_key_matrix[:, :, 3:-1]], dim=3
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        exp_blocked_value_matrix = torch.cat(
            [blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2], blocked_value_matrix[:, :, 3:-1]],
            dim=3,
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2]


        ######### For top K MMR, we need to adjust the sliding window here:
        Fw = exp_blocked_key_matrix.shape[3] # get total number of candidate tokens in all of the sliding windows
        k_to_select = min(bigger_bird_config.max_k, Fw) # how many keys we want to select


        # Create filter so we can mask tokens before pre-filtering in topk MMR:
            # band_mask shape: [bsz, n_mid, from_block_size, 3*to_block_size]
            # We need [bsz, n_heads, n_mid, Fw] — average/any over the block_size query dim
            # since all queries in a block share the same candidate window
        band_mask_expanded = band_mask.expand(-1, n_heads, -1, -1, -1)
            # any() over from_block_size: a candidate is valid if any query in the block can see it
            # shape: [bsz, n_heads, n_mid, Fw]
        local_valid_mask = band_mask_expanded.any(dim=-2)

        # Perform topK mmr
        mmr_idx = self.top_k_mmr(
            query=middle_query_matrix,
            keys=exp_blocked_key_matrix,
            k=k_to_select,
            valid_mask=local_valid_mask, # use mask so we can already filter out invalid tokens to begin with
        )  # [bsz, n_heads, n_mid, k], returns selected local token indices

        # Gather selected keys and values by the selected mmr_idx
        d = exp_blocked_key_matrix.shape[-1]
        mmr_idx_exp = mmr_idx.unsqueeze(-1).expand(-1, -1, -1, -1, d)

        # Use torch.gather to select the keys we want
        selected_local_keys = torch.gather(
            exp_blocked_key_matrix,
            dim=3,
            index=mmr_idx_exp,
        )  # [bsz, n_heads, n_mid, k, d]

        selected_local_values = torch.gather(
            exp_blocked_value_matrix,
            dim=3,
            index=mmr_idx_exp,
        )  # [bsz, n_heads, n_mid, k, d]

        # sliding attention scores for q[-2:2]
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [b, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        #inner_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, exp_blocked_key_matrix, ndim=5)
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, 3*to_block_size]

        # Compute the sliding/local attention scores using only MMR-selected local tokens
        inner_band_product = self.torch_bmm_nd_transpose(
            middle_query_matrix,
            selected_local_keys,
            ndim=5,
        )
        inner_band_product = inner_band_product * rsqrt_d

        # randn attention scores for q[-2:2]
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, from_seq_len//from_block_size-4, n_rand_blocks*to_block_size, -1]
        rand_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, gathered_key[:, :, 1:-1], ndim=5)
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, n_rand_blocks*to_block_size]
        rand_band_product = rand_band_product * rsqrt_d

        # Including 1st block (since it's global)
        first_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, 0]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size]
        first_band_product = first_band_product * rsqrt_d

        # Including last block (since it's global)
        last_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, -1]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size]
        last_band_product = last_band_product * rsqrt_d

        # masking padded tokens (no need for inner_band_product since masking was performed in topK MMR)
        first_band_product += (1.0 - to_mask[:, :, :, :to_block_size].unsqueeze(3)) * attn_mask_penalty
        last_band_product += (1.0 - to_mask[:, :, :, -to_block_size:].unsqueeze(3)) * attn_mask_penalty
        rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * attn_mask_penalty

        # completing attention scores matrix for all q[-2:2]
        band_product = torch.cat(
            [first_band_product, inner_band_product, rand_band_product, last_band_product], dim=-1
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, (5+n_rand_blocks)*to_block_size]

        # safely doing softmax since attention matrix is completed
        attn_weights = nn.functional.softmax(
            band_product, dim=-1
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, (5+n_rand_blocks)*to_block_size]

        # contribution of sliding 
        # [bsz, n_heads, m//from_block_size-4, from_block_size, 3*to_block_size] x [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        #context_layer = self.torch_bmm_nd(
        #    attn_weights[:, :, :, :, to_block_size : 4 * to_block_size], exp_blocked_value_matrix, ndim=5
        #)
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]


        # Slice boundaries into the last dim of attn_weights, which after the cat is:
        # [ first_global | local_mmr | random | last_global ]
        # :to_block_size   local_start  local_end  rand_end:
        #                  local_end    rand_end
        # MMR reduced the local window from 3*to_block_size to k_to_select,
        # so offsets must be computed dynamically instead of hardcoded.
        local_start = to_block_size
        local_end   = to_block_size + k_to_select
        rand_end    = local_end + n_rand_blocks * to_block_size

        # local (MMR-selected) keys contribution
        context_layer = self.torch_bmm_nd(
            attn_weights[:, :, :, :, local_start : local_end],
            selected_local_values,   # ← must match what we scored against
            ndim=5,
        )


        # adding contribution of random keys
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, n_rand_blocks*to_block_size] x [bsz, n_heads, from_seq_len//from_block_size-4, n_rand_blocks*to_block_size, -1]
        #context_layer += self.torch_bmm_nd(
        #    attn_weights[:, :, :, :, 4 * to_block_size : -to_block_size], gathered_value[:, :, 1:-1], ndim=5
        #)
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]

        context_layer += self.torch_bmm_nd(
            attn_weights[:, :, :, :, local_end : rand_end], # adjust to local_end and rand_end
            gathered_value[:, :, 1:-1],
            ndim=5,
        )

        # adding contribution of global keys
        context_layer += torch.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, :to_block_size], blocked_value_matrix[:, :, 0]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]
        #context_layer += torch.einsum(
        #    "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, -to_block_size:], blocked_value_matrix[:, :, -1]
        #)  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]


        # adjust the slicing due to MMR selections, last global block (rand_end: because MMR shifted the offset)
        context_layer += torch.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, rand_end:], blocked_value_matrix[:, :, -1]
        )

        # 4th PART
        # last 2nd token attention scores
        # q[-2] x (sliding_keys, random_keys, global_keys)
        # sliding key blocks -> last 3 blocks
        # global key block -> 1st block
        # random key block -> based on indices stored in `randn_attn`

        second_last_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, -3],
                blocked_key_matrix[:, :, -2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, -1],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+n_random_blocks)*to_block_size, -1]
        second_last_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, -3],
                blocked_value_matrix[:, :, -2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, -1],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+r)*to_block_size, -1]

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]
        second_last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -2], second_last_key_mat, ndim=4)
        second_last_seq_pad = torch.cat(
            [
                to_mask[:, :, :, :to_block_size],
                to_mask[:, :, :, -3 * to_block_size :],
                to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size]),
            ],
            dim=3,
        )
        second_last_rand_pad = torch.cat(
            [
                rand_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]),
                rand_mask[:, :, -1],
            ],
            dim=3,
        )
        second_last_product = second_last_product * rsqrt_d
        second_last_product += (1.0 - torch.minimum(second_last_seq_pad, second_last_rand_pad)) * attn_mask_penalty
        second_last_attn_weights = nn.functional.softmax(
            second_last_product, dim=-1
        )  # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]

        # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, -1]
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

        # this is just for visualizing; forward pass doesn't depend on following code
        # TODO(PVP): need to verify if below code is correct
        attention_probs = torch.zeros(
            bsz, n_heads, from_seq_len, to_seq_len, dtype=context_layer.dtype, device=context_layer.device
        )

        # 1st query block
        # corresponding to `first_context_layer`
        attention_probs[:, :, :from_block_size, :] = first_attn_weights  # all keys global

        # 2nd query block
        # corresponding to `second_context_layer`
        attention_probs[:, :, from_block_size : 2 * from_block_size, : 3 * to_block_size] = second_attn_weights[
            :, :, :, : 3 * to_block_size
        ]  # 1st three key blocks (global + sliding)
        attention_probs[:, :, from_block_size : 2 * from_block_size, -to_block_size:] = second_attn_weights[
            :, :, :, 3 * to_block_size : 4 * to_block_size
        ]  # last key block (global)
        # random keys
        for p1, i1, w1 in zip(range(bsz), rand_attn, second_attn_weights):
            # p1, i1, w1 corresponds to batch_dim i.e. following operation is done for each sequence in batch
            for p2, i2, w2 in zip(range(n_heads), i1, w1):
                # p2, i2, w2 corresponds to head_dim i.e. following operation is done for each heads
                attn_probs_view = attention_probs.view(
                    bsz,
                    n_heads,
                    from_seq_len // from_block_size,
                    from_block_size,
                    to_seq_len // to_block_size,
                    to_block_size,
                )
                right_slice = w2[:, 4 * to_block_size :]
                attn_probs_view[p1, p2, 1, :, i2[0]] = right_slice.view(from_block_size, n_rand_blocks, to_block_size)

        # Middle query blocks
        # corresponding to `context_layer`
        # sliding keys
        # NOTE: MMR selects k_to_select tokens from a 3-block window, which no longer maps
        # cleanly back to a contiguous block range in attention_probs since k_to_select
        # is not guaranteed to be a multiple of to_block_size. Skipping middle-block
        # sliding key visualization — random and global key probs below are still correct.

        # for q_idx in range(from_seq_len // from_block_size - 4):
        #     attn_probs_view = attention_probs.view(
        #         bsz,
        #         n_heads,
        #         from_seq_len // from_block_size,
        #         from_block_size,
        #         to_seq_len // to_block_size,
        #         to_block_size,
        #     )[:, :, 2:-2, :, 1:-1, :]
        #     right_slice = attn_weights[:, :, q_idx, :, to_block_size : 4 * to_block_size]
        #     attn_probs_view[:, :, q_idx, :, q_idx : q_idx + 3, :] = right_slice.view(
        #         bsz, n_heads, from_block_size, 3, to_block_size
        #     )  # inner_band_product

        # global keys (corresponding to 1st key block)
        attention_probs[:, :, 2 * from_block_size : -2 * from_block_size, :to_block_size] = attn_weights[
            :, :, :, :, :to_block_size
        ].view(bsz, n_heads, -1, to_block_size)  # first_band_product

        # global keys (corresponding to last key block)
        # attention_probs[:, :, 2 * from_block_size : -2 * from_block_size, -to_block_size:] = attn_weights[
        #     :, :, :, :, -to_block_size:
        # ].view(bsz, n_heads, -1, to_block_size)  # last_band_product
        # NOTE: MMR shifted the offset of the last global block — it no longer starts at -to_block_size
        # from the end; use rand_end which accounts for the reduced local window size
        attention_probs[:, :, 2 * from_block_size : -2 * from_block_size, -to_block_size:] = attn_weights[
            :, :, :, :, rand_end:
        ].view(bsz, n_heads, -1, to_block_size)

        # random keys
        for p1, i1, w1 in zip(range(bsz), rand_attn, attn_weights):
            # p1, i1, w1 corresponds to batch_dim i.e. following operation is done for each sequence in batch
            for p2, i2, w2 in zip(range(n_heads), i1, w1):
                # p2, i2, w2 corresponds to head_dim i.e. following operation is done for each heads
                for q_idx in range(1, len(i2) - 1):
                    attn_probs_view = attention_probs.view(
                        bsz,
                        n_heads,
                        from_seq_len // from_block_size,
                        from_block_size,
                        to_seq_len // to_block_size,
                        to_block_size,
                    )
                    #right_slice = w2[q_idx - 1, :, 4 * to_block_size : -to_block_size]
                    # NOTE: MMR shifted where the random block starts in attn_weights — it no longer starts at
                    # 4*to_block_size; use dynamic local_end:rand_end to account for reduced local window size
                    right_slice = w2[q_idx - 1, :, local_end : rand_end]

                    attn_probs_view[p1, p2, q_idx + 1, :, i2[q_idx]] = right_slice.view(
                        from_block_size, n_rand_blocks, to_block_size
                    )

        # Second-last query block
        # corresponding to `second_last_context_layer`
        attention_probs[:, :, -2 * from_block_size : -from_block_size, :to_block_size] = second_last_attn_weights[
            :, :, :, :to_block_size
        ]  # 1st key block (global)
        attention_probs[:, :, -2 * from_block_size : -from_block_size, -3 * to_block_size :] = (
            second_last_attn_weights[:, :, :, to_block_size : 4 * to_block_size]
        )  # last three blocks (global + sliding)
        # random keys
        for p1, i1, w1 in zip(range(bsz), rand_attn, second_last_attn_weights):
            # p1, i1, w1 corresponds to batch_dim i.e. following operation is done for each sequence in batch
            for p2, i2, w2 in zip(range(n_heads), i1, w1):
                # p2, i2, w2 corresponds to head_dim i.e. following operation is done for each heads
                attn_probs_view = attention_probs.view(
                    bsz,
                    n_heads,
                    from_seq_len // from_block_size,
                    from_block_size,
                    to_seq_len // to_block_size,
                    to_block_size,
                )
                right_slice = w2[:, 4 * to_block_size :]
                attn_probs_view[p1, p2, -2, :, i2[-1]] = right_slice.view(
                    from_block_size, n_rand_blocks, to_block_size
                )

        # last query block
        # corresponding to `last_context_layer`
        attention_probs[:, :, -from_block_size:, :] = last_attn_weights  # all keys global

        return context_layer, attention_probs

    @staticmethod
    def F_normalize_safe(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=dim, eps=eps)

    def top_k_mmr(
        self,
        query: torch.Tensor,        # [bsz, n_heads, num_middle_blocks, block_size, d]
        keys: torch.Tensor,         # [bsz, n_heads, num_middle_blocks, Fw, d]  — candidate window
        k: int,                     # how many tokens to select per query block
        valid_mask: torch.Tensor | None,  # [bsz, n_heads, num_middle_blocks, Fw] — True = valid token
    ) -> torch.Tensor:              # returns [bsz, n_heads, num_middle_blocks, k] — indices into Fw
        """
        Two-phase blocked MMR over a sliding window of size Fw.
        Phase 1: topK prefilter reduces Fw candidates to Kc = ceil(mmr_prefilter_mult * k).
        Phase 2: mmr_diversity_steps diversity rounds on the candidate set, then greedy fill.
        All hyperparameters drawn from bigger_bird_config.
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

        # mask padding tokens to -inf before prefilter so they can never enter the candidate set
        if valid_mask is not None:
            relevance = relevance.masked_fill(~valid_mask.bool(), torch.finfo(relevance.dtype).min)

        # PHASE 1: topK prefilter
        # reduce from Fw to Kc = min(Fw, ceil(mmr_prefilter_mult * k)) candidates
        Kc = int(min(Fw, max(k, np.ceil(bigger_bird_config.mmr_prefilter_mult * k))))
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
        total_steps = min(k, 1 + int(bigger_bird_config.mmr_diversity_steps))
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
                remaining = remaining - bigger_bird_config.gamma_diversity * cos

                # mask the selected position so it cannot be picked again
                remaining.scatter_(-1, j.unsqueeze(-1), -1e9)

        # greedy fill for remaining k - total_steps slots
        # remaining scores already reflect diversity penalties from the rounds above
        if k > total_steps:
            _, fill_pos = torch.topk(remaining, k=k - total_steps, dim=-1)     # [bsz, n_heads, n_mid, k-total_steps]
            selected[..., total_steps:] = torch.gather(cand_idx, -1, fill_pos)

        return selected  # [bsz, n_heads, n_mid, k]

