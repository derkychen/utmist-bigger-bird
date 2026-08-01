#!/usr/bin/env python3
"""One-shot smoke test for the R1 vLLM server.

Sends a single prompt and prints the response. Use this to verify the server
is up and the model is producing sane output before launching the interactive
chat client.

    python r1/smoke_test.py
    python r1/smoke_test.py --url http://127.0.0.1:8000 --prompt "What is 27*13?"
"""
import argparse
import sys
from openai import OpenAI


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="r1-distill-llama-8b")
    p.add_argument("--prompt", default="What is 27 * 13? Think step by step.")
    p.add_argument("--max-tokens", type=int, default=2048)
    args = p.parse_args()

    client = OpenAI(base_url=args.url, api_key="EMPTY")
    print(f"smoke test → {args.url}  model={args.model}")
    print(f"prompt: {args.prompt}\n")
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            temperature=0.6,
            top_p=0.95,
            max_tokens=args.max_tokens,
        )
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    out = resp.choices[0].message.content
    print(out)
    usage = resp.usage
    print(f"\n--- tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens} ---")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
