"""CPU check: rebuilt prompts match the CCDB Round 4 manifests token-for-token."""
import sys

from modal_rd.niah import load_tokenizer, prepare

tok = load_tokenizer()
for nbytes in map(int, (sys.argv[1] if len(sys.argv) > 1 else "512000,261000").split(",")):
    rows = prepare(tok, nbytes, 6, verify=True)
    print(f"MATCH bytes={nbytes} tokens={[r['tokens'] for r in rows]} answers={[r['answer'] for r in rows]}", flush=True)
