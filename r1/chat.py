#!/usr/bin/env python3
"""
Interactive chat client for the R1-Distill vLLM server.

Talks to the OpenAI-compatible endpoint exposed by `r1/serve.sh`.
Streams tokens, renders R1's <think> reasoning in a dim colour and the final
answer in full colour so you can see the model's chain-of-thought vs. output.

Usage (from a node that can reach the server — typically the same GPU node):
    python r1/chat.py                          # default http://127.0.0.1:8000
    python r1/chat.py --url http://127.0.0.1:8000 --model r1-distill-llama-8b
    python r1/chat.py --system "You are a helpful math tutor."

Commands inside the session:
    /reset   clear conversation history
    /system  change the system prompt (then type the new prompt)
    /quit    exit
"""
from __future__ import annotations

import argparse
import sys
from openai import OpenAI

# ANSI colours — kept subtle so the terminal stays readable.
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def render_stream(stream) -> str:
    """Stream tokens, colouring <think>...</think> blocks dim cyan.

    Returns the full assembled text.
    """
    full = []
    in_think = False
    pending_open = THINK_OPEN  # we match the literal tag as it streams
    buf = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if not delta.content:
            continue
        tok = delta.content
        full.append(tok)
        buf += tok

        # Detect <think> open / close as they arrive (may span chunks).
        while buf:
            if not in_think and THINK_OPEN in buf:
                pre, _, buf = buf.partition(THINK_OPEN)
                if pre:
                    sys.stdout.write(pre)
                    sys.stdout.flush()
                sys.stdout.write(DIM + CYAN)
                sys.stdout.flush()
                in_think = True
                continue
            if in_think and THINK_CLOSE in buf:
                pre, _, buf = buf.partition(THINK_CLOSE)
                if pre:
                    sys.stdout.write(pre)
                    sys.stdout.flush()
                sys.stdout.write(RESET + "\n")
                sys.stdout.flush()
                in_think = False
                continue
            # Flush safe prefix, keep a tail that might be a partial tag.
            safe = len(buf)
            if not in_think:
                # keep last len(THINK_OPEN)-1 chars in case tag is split
                safe = max(0, len(buf) - (len(THINK_OPEN) - 1))
            else:
                safe = max(0, len(buf) - (len(THINK_CLOSE) - 1))
            if safe:
                sys.stdout.write(buf[:safe])
                sys.stdout.flush()
                buf = buf[safe:]
            break
    # Flush whatever remains.
    if buf:
        sys.stdout.write(buf)
        sys.stdout.flush()
    if in_think:
        sys.stdout.write(RESET)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(full)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI base URL (must include /v1)")
    p.add_argument("--model", default="r1-distill-llama-8b", help="served model name (see --served-model-name in serve.sh)")
    p.add_argument("--system", default=None, help="optional system prompt")
    p.add_argument("--temperature", type=float, default=0.6, help="R1 recommended 0.5-0.7")
    p.add_argument("--top-p", type=float, default=0.95, help="nucleus sampling")
    p.add_argument("--max-tokens", type=int, default=8192, help="max new tokens per turn")
    args = p.parse_args()

    client = OpenAI(base_url=args.url, api_key="EMPTY")

    # R1-Distill chat template expects plain user turns (no pre-wrapped <think>).
    system_prompt = args.system or (
        "You are DeepSeek-R1, a helpful reasoning assistant. "
        "Think step by step inside <think>...</think> before answering."
    )
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    print(f"{BOLD}R1 chat{RESET} — model: {GREEN}{args.model}{RESET}  endpoint: {args.url}")
    print(f"  /reset  /system  /quit   |  temp={args.temperature}  top_p={args.top_p}  max_tokens={args.max_tokens}")
    print()

    while True:
        try:
            user = input(f"{BOLD}{YELLOW}you › {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            messages = [{"role": "system", "content": system_prompt}]
            print(f"{DIM}(history cleared){RESET}\n")
            continue
        if user == "/system":
            new = input("new system prompt: ").strip()
            if new:
                system_prompt = new
                messages = [{"role": "system", "content": system_prompt}]
                print(f"{DIM}(system prompt updated, history cleared){RESET}\n")
            continue

        messages.append({"role": "user", "content": user})
        print(f"{BOLD}{GREEN}r1 › {RESET}", end="", flush=True)
        try:
            stream = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                stream=True,
            )
            reply = render_stream(stream)
        except Exception as e:
            print(f"\n{BOLD}error:{RESET} {e}", file=sys.stderr)
            # drop the failed user turn so history stays consistent
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
