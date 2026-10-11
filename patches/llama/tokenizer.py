"""Tokenizer loading for R1-Distill-Llama-8B.

transformers 5.x maps this checkpoint's ``tokenizer_class: LlamaTokenizerFast``
to the SentencePiece-style ``LlamaTokenizer``, which mis-tokenizes the Llama-3
byte-level vocabulary (spaces and newlines are dropped, token ids change).
Loading ``tokenizer.json`` through ``PreTrainedTokenizerFast`` avoids the
remap; the BOS post-processor restores LlamaTokenizerFast's add_bos behaviour.
"""

from __future__ import annotations

from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


def load_llama_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    bos = tokenizer.bos_token
    if bos is not None:
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0",
            pair=f"{bos}:0 $A:0 {bos}:1 $B:1",
            special_tokens=[(bos, tokenizer.bos_token_id)],
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
