"""The R1-Llama tokenizer must keep Llama-3 byte-level tokenization.

transformers 5.x ``AutoTokenizer`` maps this checkpoint to a SentencePiece-style
``LlamaTokenizer`` that drops spaces, which silently corrupts every prompt.
"""

import os

from eval.ruler_llama.run_generative import load_tokenizer

MODEL = os.environ.get("R1_TOKENIZER_PATH", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")


def test_round_trip_keeps_spaces():
    tok = load_tokenizer(MODEL)
    text = "One of the special magic numbers is 2940341.\nWhat is it?"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert ids[:4] == [4054, 315, 279, 3361]  # "One", " of", " the", " special"
    assert tok.decode(ids) == text


def test_bos_and_padding():
    tok = load_tokenizer(MODEL)
    ids = tok("hello")["input_ids"]
    assert ids[0] == tok.bos_token_id
    assert tok.pad_token is not None
    assert tok.padding_side == "left"
