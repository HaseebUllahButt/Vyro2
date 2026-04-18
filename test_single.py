"""
test_single.py
──────────────
Quick CLI tool for judges and developers to test any single prompt.

Usage:
    python test_single.py "What's the weather in London in Celsius?"
    python test_single.py "now do GBP instead" --history '100 USD to EUR'
    python test_single.py --interactive
"""

import sys
import json
import argparse
import time

sys.path.insert(0, ".")
from inference import run


def _print_result(prompt: str, history: list, result: str, latency_ms: float):
    import re
    m = re.search(r'<tool_call>(.*?)</tool_call>', result)
    print(f"\n{'─'*60}")
    print(f"Prompt : {prompt}")
    if history:
        print(f"History: {len(history)} turn(s)")
    print(f"Result : {result}")
    if m:
        try:
            data = json.loads(m.group(1).strip())
            print(f"Parsed : {json.dumps(data, indent=2)}")
        except Exception:
            pass
    print(f"Time   : {latency_ms:.1f}ms")
    print('─'*60)


def main():
    parser = argparse.ArgumentParser(description="Test inference.run() from the CLI")
    parser.add_argument("prompt", nargs="?", help="User prompt to test")
    parser.add_argument(
        "--history", "-H",
        default=None,
        help="Previous user message (creates a 2-turn context)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive REPL mode with persistent history",
    )
    args = parser.parse_args()

    if args.interactive:
        print("Pocket-Agent interactive mode. Type 'reset' to clear history, 'quit' to exit.\n")
        history = []
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if user_input.lower() == "reset":
                history = []
                print("[History cleared]")
                continue
            t0 = time.time()
            result = run(user_input, history)
            latency = (time.time() - t0) * 1000
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result})
            print(f"Agent ({latency:.0f}ms): {result}\n")
        return

    if not args.prompt:
        parser.print_help()
        sys.exit(1)

    history = []
    if args.history:
        # Simulate a single prior turn
        t0 = time.time()
        prior_result = run(args.history, [])
        history = [
            {"role": "user", "content": args.history},
            {"role": "assistant", "content": prior_result},
        ]
        print(f"Prior turn: {args.history!r} → {prior_result}")

    t0 = time.time()
    result = run(args.prompt, history)
    latency_ms = (time.time() - t0) * 1000

    _print_result(args.prompt, history, result, latency_ms)


if __name__ == "__main__":
    main()
