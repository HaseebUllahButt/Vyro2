"""
src/eval/score.py
─────────────────
Local evaluation harness. Mirrors the grader's scoring exactly.

Run from repo root:
    python src/eval/score.py [--test-file path/to/test.jsonl]

Default test file: starter/public_test.jsonl
Falls back to a built-in self-test if no test file is provided.
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, ".")
from inference import run


# ── Scoring ────────────────────────────────────────────────────────────────────

def parse_output(text: str) -> dict:
    """Parse model output into a structured result."""
    m = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if not m:
        return {"type": "refusal", "raw": text}
    try:
        data = json.loads(m.group(1).strip())
        return {"type": "tool_call", "data": data}
    except json.JSONDecodeError:
        return {"type": "malformed", "raw": text}


def score_pair(prediction: str, expected: str) -> tuple:
    """
    Returns (score, reason) following the hackathon rubric:
      +1.0  exact tool + all args correct (numerics ±1%)
      +0.5  correct tool, ≥1 arg wrong
       0.0  wrong tool, malformed JSON, or wrong refusal
      -0.5  emitted tool call when refusal was correct
    """
    pred = parse_output(prediction)
    exp = parse_output(expected)

    # Expected: refusal
    if exp["type"] == "refusal":
        if pred["type"] == "refusal":
            return 1.0, "correct_refusal"
        return -0.5, f"tool_call_when_refusal_expected | pred={prediction[:60]}"

    # Expected: tool call
    if pred["type"] == "malformed":
        return 0.0, f"malformed_json"
    if pred["type"] == "refusal":
        return 0.0, f"wrong_refusal | expected={expected[:60]}"

    p_data = pred["data"]
    e_data = exp["data"]

    if p_data.get("tool") != e_data.get("tool"):
        return 0.0, f"wrong_tool | got={p_data.get('tool')} expected={e_data.get('tool')}"

    p_args = p_data.get("args", {})
    e_args = e_data.get("args", {})
    all_correct = True
    mismatches = []

    for key, exp_val in e_args.items():
        pred_val = p_args.get(key)
        if isinstance(exp_val, (int, float)):
            try:
                pv = float(pred_val)
                ev = float(exp_val)
                if abs(pv - ev) / max(abs(ev), 1e-9) > 0.01:
                    all_correct = False
                    mismatches.append(f"{key}:{pred_val}≠{exp_val}")
            except (TypeError, ValueError):
                all_correct = False
                mismatches.append(f"{key}:non-numeric")
        else:
            if str(pred_val).strip() != str(exp_val).strip():
                all_correct = False
                mismatches.append(f"{key}:{pred_val!r}≠{exp_val!r}")

    if all_correct:
        return 1.0, "correct"
    return 0.5, f"arg_mismatch | {', '.join(mismatches)}"


# ── Built-in self-test (no external file needed) ──────────────────────────────

BUILTIN_TESTS = [
    {
        "prompt": "What's the weather in London in Celsius?",
        "history": [],
        "expected": '<tool_call>{"tool": "weather", "args": {"location": "London", "unit": "C"}}</tool_call>',
        "slice": "A",
    },
    {
        "prompt": "Convert 100 USD to EUR",
        "history": [],
        "expected": '<tool_call>{"tool": "currency", "args": {"amount": 100, "from": "USD", "to": "EUR"}}</tool_call>',
        "slice": "A",
    },
    {
        "prompt": "convert 10 km to miles",
        "history": [],
        "expected": '<tool_call>{"tool": "convert", "args": {"value": 10.0, "from_unit": "km", "to_unit": "miles"}}</tool_call>',
        "slice": "A",
    },
    {
        "prompt": "What do I have tomorrow?",
        "history": [],
        "expected": '<tool_call>{"tool": "calendar", "args": {"action": "list", "date": "DYNAMIC"}}</tool_call>',
        "slice": "A",
    },
    {
        "prompt": "Show all users",
        "history": [],
        "expected": '<tool_call>{"tool": "sql", "args": {"query": "SELECT * FROM users"}}</tool_call>',
        "slice": "A",
    },
    # Slice B: paraphrased
    {
        "prompt": "temperature in Tokyo please, Fahrenheit",
        "history": [],
        "expected": '<tool_call>{"tool": "weather", "args": {"location": "Tokyo", "unit": "F"}}</tool_call>',
        "slice": "B",
    },
    {
        "prompt": "how many euros is 500 dollars?",
        "history": [],
        "expected": '<tool_call>{"tool": "currency", "args": {"amount": 500, "from": "USD", "to": "EUR"}}</tool_call>',
        "slice": "B",
    },
    # Slice C: adversarial
    {
        "prompt": "wether in karachi celsius",
        "history": [],
        "expected": '<tool_call>{"tool": "weather", "args": {"location": "Karachi", "unit": "C"}}</tool_call>',
        "slice": "C",
    },
    {
        "prompt": "100 USD ko EUR mein convert karo",
        "history": [],
        "expected": '<tool_call>{"tool": "currency", "args": {"amount": 100, "from": "USD", "to": "EUR"}}</tool_call>',
        "slice": "C",
    },
    # Slice D: refusals
    {
        "prompt": "tell me a joke",
        "history": [],
        "expected": "refusal",
        "slice": "D",
    },
    {
        "prompt": "set a reminder for 8am",
        "history": [],
        "expected": "refusal",
        "slice": "D",
    },
    # Slice D: multi-turn
    {
        "prompt": "now do GBP instead",
        "history": [
            {"role": "user", "content": "Convert 100 USD to EUR"},
            {"role": "assistant", "content": '<tool_call>{"tool": "currency", "args": {"amount": 100, "from": "USD", "to": "EUR"}}</tool_call>'},
        ],
        "expected": '<tool_call>{"tool": "currency", "args": {"amount": 100, "from": "USD", "to": "GBP"}}</tool_call>',
        "slice": "D",
    },
]


def _load_test_file(path: str) -> list:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            examples.append({
                "prompt": ex.get("prompt", ""),
                "history": ex.get("history", []),
                "expected": ex.get("expected", ""),
                "slice": ex.get("slice", "?"),
            })
    return examples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-file",
        default=None,
        help="Path to test JSONL. Defaults to starter/public_test.jsonl or built-in tests.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print all results, not just failures.",
    )
    args = parser.parse_args()

    # Resolve test file
    test_path = args.test_file
    if not test_path:
        default = Path("starter/public_test.jsonl")
        if default.exists():
            test_path = str(default)
            print(f"Using: {default}")
        else:
            print("No test file found — using built-in self-test suite")
            examples = BUILTIN_TESTS
            test_path = None

    if test_path:
        examples = _load_test_file(test_path)
        print(f"Loaded {len(examples)} examples from {test_path}")

    total = 0.0
    counts = {}
    latencies = []
    slice_scores = {}

    print("\n" + "═" * 70)

    for i, ex in enumerate(examples):
        t0 = time.time()
        got = run(ex["prompt"], ex["history"])
        latency_ms = (time.time() - t0) * 1000

        expected = ex["expected"]
        # Handle dynamic dates in built-in tests
        if expected == "refusal":
            expected = "plain text"

        s, reason = score_pair(got, expected)

        total += s
        latencies.append(latency_ms)
        slice_label = ex.get("slice", "?")
        slice_scores.setdefault(slice_label, []).append(s)
        counts[reason] = counts.get(reason, 0) + 1

        failed = s < 1.0
        if failed or args.verbose:
            print(f"[{s:+.1f}] Slice={slice_label} {latency_ms:.0f}ms — {reason}")
            print(f"       Prompt:   {ex['prompt'][:70]}")
            if ex["history"]:
                print(f"       History:  {len(ex['history'])} turns")
            print(f"       Expected: {expected[:70]}")
            print(f"       Got:      {got[:70]}")
            print()

    max_score = len(examples)
    print("═" * 70)
    print(f"\nSCORE: {total:.1f} / {max_score:.0f}  ({100*total/max_score:.1f}%)")

    print("\nBy slice:")
    for s_label in sorted(slice_scores):
        scores = slice_scores[s_label]
        avg = sum(scores) / len(scores)
        print(f"  Slice {s_label}: {sum(scores):.1f}/{len(scores):.0f} ({100*avg:.0f}%)")

    if latencies:
        print(f"\nLatency:  mean={sum(latencies)/len(latencies):.1f}ms  "
              f"max={max(latencies):.1f}ms  p95={sorted(latencies)[int(0.95*len(latencies))]:.1f}ms")

    print("\nReason breakdown:")
    for reason, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {count:3d}  {reason}")

    ratio = total / max_score
    print()
    if ratio >= 0.85:
        print("✅ EXCELLENT — ready to submit")
    elif ratio >= 0.75:
        print("✅ GOOD — proceed to quantization")
    elif ratio >= 0.65:
        print("⚠️  BORDERLINE — check failure cases above")
    else:
        print("❌ LOW — recheck data format and system prompt")


if __name__ == "__main__":
    main()
