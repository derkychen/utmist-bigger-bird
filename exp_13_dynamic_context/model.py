import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput

# Idea: Dynamic Context Window — Fixed Token Budget
# Accepts ANY input length by capping the post-early-layer attention to a fixed
# token budget.  After a few dense early layers, we score all tokens by hidden-state
# importance and keep exactly target_budget tokens regardless of original length.
#
# For very long inputs (> chunk_size), early layers are also run in independent
# chunks so that embedding + early attention never materialises the full O(n²)
# matrix at once.  The chunk outputs are scored, and a global top-k selects the
# budget tokens for the remaining (expensive) layers.


class DynamicContextEncoder(nn.Module):
    """BART encoder wrapper with fixed-token-budget dropping.

    Layers 0 .. drop_after_layer-1: full dense attention (or chunked for very long seqs).
    At depth = drop_after_layer: score tokens globally, keep exactly target_budget.
    Remaining layers run on the smaller sequence -> attention cost is O(budget²).
    """
    def __init__(
        self,
        encoder: nn.Module,
        drop_after_layer: int = 3,
        target_budget: int = 4096,
        chunk_size: int = 8192,
        use_local_early: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.drop_after_layer = drop_after_layer
        self.target_budget = target_budget
        self.chunk_size = chunk_size
        self.use_local_early = use_local_early

    def _embed(self, input_ids):
        """BART embedding (tokens + positions + LN + dropout)."""
        inputs_embeds = self.encoder.embed_tokens(input_ids)
        embed_pos = self.encoder.embed_positions(input_ids)
        embed_pos = embed_pos.to(inputs_embeds.device)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.encoder.layernorm_embedding(hidden_states)
        dropout_p = self.encoder.config.dropout
        return F.dropout(hidden_states, p=dropout_p, training=self.training)

    def _build_4d_mask(self, mask_2d, dtype):
        """Convert [B, T] binary mask to BART's [B, 1, 1, T] additive mask."""
        mask_f = mask_2d.float() if mask_2d.dtype != torch.float else mask_2d
        ext = (1.0 - mask_f) * torch.finfo(dtype).min
        return ext[:, None, None, :]

    def _run_layers(self, hidden_states, current_mask, start_layer, end_layer):
        """Run encoder layers [start_layer, end_layer) with current mask."""
        for i in range(start_layer, end_layer):
            layer = self.encoder.layers[i]
            ext_mask = self._build_4d_mask(current_mask, hidden_states.dtype)
            out = layer(hidden_states, attention_mask=ext_mask, layer_head_mask=None, output_attentions=False)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return hidden_states, current_mask

    def _score_and_budget(self, hidden_states, current_mask, budget):
        """Score tokens by L2 norm and keep top `budget`, preserving order."""
        norms = hidden_states.norm(dim=-1)  # [B, T]
        pad_mask = (current_mask == 0)
        norms = norms.masked_fill(pad_mask, -1e9)
        cur_len = hidden_states.size(1)
        keep_n = min(budget, cur_len)
        _, top_idx = torch.topk(norms, k=keep_n, dim=-1)  # [B, keep_n]
        top_idx, _ = torch.sort(top_idx, dim=-1)  # preserve relative order
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        hidden_states = torch.gather(hidden_states, 1, gather_idx)
        current_mask = torch.gather(current_mask, 1, top_idx)
        return hidden_states, current_mask, top_idx

    def forward(self, input_ids=None, attention_mask=None, return_dict=True, **kwargs):
        bsz, seq_len = input_ids.shape
        budget = self.target_budget

        if attention_mask is None:
            current_mask = torch.ones(bsz, seq_len, device=input_ids.device, dtype=torch.long)
        else:
            current_mask = attention_mask

        # ------------------------------------------------------------------
        # PATH A: short sequence — full early layers then budget drop
        # ------------------------------------------------------------------
        if seq_len <= self.chunk_size and not self.use_local_early:
            hidden_states = self._embed(input_ids)
            hidden_states, current_mask = self._run_layers(
                hidden_states, current_mask, 0, self.drop_after_layer
            )
            hidden_states, current_mask, kept_indices = self._score_and_budget(
                hidden_states, current_mask, budget
            )
            hidden_states, current_mask = self._run_layers(
                hidden_states, current_mask, self.drop_after_layer, len(self.encoder.layers)
            )
            return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=None, attentions=None), current_mask, kept_indices

        # ------------------------------------------------------------------
        # PATH B: very long sequence — chunked early layers
        # ------------------------------------------------------------------
        # We cannot run full self-attention over 100k+ tokens in early layers.
        # Instead:
        #   1. Embed the FULL sequence (O(n) memory — cheap).
        #   2. Split into non-overlapping chunks of size chunk_size.
        #   3. Run early layers on EACH chunk independently (local attention only).
        #   4. Concatenate chunk outputs and score globally.
        #   5. Keep top `budget` tokens across the whole input.
        #   6. Run remaining layers on the selected budget tokens.
        # ------------------------------------------------------------------
        hidden_states = self._embed(input_ids)  # [B, T, D]
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        chunk_outputs = []
        chunk_masks = []
        for c in range(num_chunks):
            start = c * self.chunk_size
            end = min(start + self.chunk_size, seq_len)
            chunk_h = hidden_states[:, start:end, :]          # [B, chunk, D]
            chunk_m = current_mask[:, start:end]                # [B, chunk]

            chunk_h, chunk_m = self._run_layers(
                chunk_h, chunk_m, 0, self.drop_after_layer
            )
            chunk_outputs.append(chunk_h)
            chunk_masks.append(chunk_m)

        # Concatenate all chunk outputs
        hidden_states = torch.cat(chunk_outputs, dim=1)   # [B, T, D]
        current_mask = torch.cat(chunk_masks, dim=1)      # [B, T]

        # Global scoring & budget selection
        hidden_states, current_mask, kept_indices = self._score_and_budget(
            hidden_states, current_mask, budget
        )

        # Remaining layers on the budget-sized sequence
        hidden_states, current_mask = self._run_layers(
            hidden_states, current_mask, self.drop_after_layer, len(self.encoder.layers)
        )

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=None, attentions=None), current_mask, kept_indices


class AttnPool(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, mask):
        h = torch.tanh(self.proj(x))
        s = self.score(h).squeeze(-1)
        s = s.masked_fill(~mask.bool(), torch.finfo(s.dtype).min)
        a = torch.softmax(s, dim=-1)
        return torch.bmm(a.unsqueeze(1), x).squeeze(1)


class PatchedModel(nn.Module):
    def __init__(
        self,
        base_model,
        drop_after_layer: int = 3,
        target_budget: int = 4096,
        chunk_size: int = 8192,
        use_local_early: bool = False,
    ):
        super().__init__()
        self.model = base_model
        encoder = base_model.model.encoder
        self.dynamic_encoder = DynamicContextEncoder(
            encoder,
            drop_after_layer=drop_after_layer,
            target_budget=target_budget,
            chunk_size=chunk_size,
            use_local_early=use_local_early,
        )
        hidden = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model")
        self.attn_pool = AttnPool(hidden)
        self.drop_after_layer = drop_after_layer
        self.target_budget = target_budget
        self.chunk_size = chunk_size
        self.use_local_early = use_local_early

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable()

    @property
    def supports_gradient_checkpointing(self):
        return getattr(self.model, "supports_gradient_checkpointing", True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        encoder_out, final_mask, _ = self.dynamic_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last = encoder_out.last_hidden_state
        if final_mask is None:
            final_mask = torch.ones(last.size()[:2], device=last.device, dtype=torch.long)
        pooled = self.attn_pool(last, final_mask)
        logits = self.model.classification_head(pooled)
        loss = None
        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)

    @property
    def config(self):
        return self.model.config
