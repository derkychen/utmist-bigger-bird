"""CPU check: token lengths, skipped prompts and gold answers per RULER NIAH variant."""
import sys

from modal_rd.niah import load_tokenizer, prepare_task

TASKS = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery"]
BYTES = list(map(int, sys.argv[2].split(","))) if len(sys.argv) > 2 else [512000]
tok = load_tokenizer()
for task in TASKS:
    for nbytes in BYTES:
        try:
            rows, skipped = prepare_task(tok, task, nbytes, 3)
            print(f"TASK {task} bytes={nbytes} tokens={[r['tokens'] for r in rows]} skipped={skipped} "
                  f"answers={rows[0]['answers']} tail={rows[0]['text'][-120:]!r}", flush=True)
        except Exception as e:
            print(f"TASK {task} bytes={nbytes} ERROR {e}", flush=True)
