"""In-place residual adds in the Llama decoder layers (exact memory saver for long prefill).

HF's LlamaDecoderLayer computes `hidden_states = residual + attn_out` and `... + mlp_out`, allocating a new
[n, hidden] tensor for each add while its input is still alive, and LlamaModel.forward keeps `inputs_embeds`
referenced for the whole layer loop. Adding in place into the layer's input instead makes the embedding
output the single residual-stream buffer for all layers. Measured on Nemotron-UltraLong (NF4, ~13K tokens):
the prefill peak drops by ~2 [n, 4096] BF16 tensors (~1 GB each at ~128K tokens, ~8 GB each at ~1M), with
identical logits and tokens for v3, v3 + eviction and exp 21. The adds are the same ops, so results are
bit-identical.

The layer's input tensor is modified: do not pass `inputs_embeds` you still need, or read
`output_hidden_states` (they would alias). Inference only.
"""
import types


def _forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False,
             position_embeddings=None, **kwargs):
    h = self.input_layernorm(hidden_states)
    h, _ = self.self_attn(hidden_states=h, attention_mask=attention_mask, position_ids=position_ids,
                          past_key_values=past_key_values, use_cache=use_cache,
                          position_embeddings=position_embeddings, **kwargs)
    hidden_states.add_(h)
    del h
    hidden_states.add_(self.mlp(self.post_attention_layernorm(hidden_states)))
    return hidden_states


def install_inplace_residual(model):
    """Patch every decoder layer's forward; idempotent."""
    for layer in model.model.layers:
        layer.forward = types.MethodType(_forward, layer)
    return model
