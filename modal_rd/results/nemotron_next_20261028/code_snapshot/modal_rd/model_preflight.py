"""CPU checkpoint/tokenizer gate, including R1 manifest regression and chat BOS parity."""
import json
from pathlib import Path
from transformers import AutoConfig
from modal_rd.model_config import MODEL_DIR, MAX_CONTEXT, PROMPT_FORMAT, model_metadata, tokenize_prompt
from modal_rd.niah import load_tokenizer, prepare


def main():
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    assert cfg.model_type == 'llama', cfg.model_type
    assert cfg.max_position_embeddings == MAX_CONTEXT
    tok = load_tokenizer()
    if PROMPT_FORMAT == 'chat':
        text = 'Tell me the number 1234567.'
        native = tok.apply_chat_template([{'role': 'user', 'content': text}], add_generation_prompt=True)
        rendered = tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)
        ours = tokenize_prompt(tok, rendered)['input_ids']
        if hasattr(native, 'keys'):
            native = native['input_ids']
        assert native == ours,  'Chat tokenization differs from native template'
        assert ours.count(tok.bos_token_id) == 1, 'Duplicated/missing BOS'
    from experiments.exp_20_sota_long_context.run import Prompter
    prompter = Prompter(tok, 512)
    probe_ids, _ = prompter.build({'id': 'preflight', 'prefix': 'Read this document: ',
                                  'context': 'The magic number is 1234567. ',
                                  'suffix': 'Choose the correct answer. A: 1234567 B: 7654321'})
    assert probe_ids.count(tok.bos_token_id) == 1, 'Exp20 duplicated/missing BOS'
    assert len(probe_ids) <= 512
    if PROMPT_FORMAT == 'chat':
        assert '<｜User｜>' not in tok.decode(probe_ids), 'R1 template leaked into native chat'
    rows = prepare(tok, 512000, 3, verify=True)
    report = {**model_metadata(), 'pass': True, 'model_type': cfg.model_type,
              'config_context': cfg.max_position_embeddings, 'rope_scaling': getattr(cfg, 'rope_scaling', None),
              'prompts': [{k: v for k, v in row.items() if k != 'text'} for row in rows]}
    path = Path('/vol/results') / f'preflight_{Path(MODEL_DIR).name}_{PROMPT_FORMAT}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
