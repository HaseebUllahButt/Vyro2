"""
src/data/lint.py
────────────────
Validates all training examples and writes a clean file.

Run from repo root:
    python src/data/lint.py

Input:  data/train.jsonl
Output: data/train_clean.jsonl  (only valid examples)
"""

import json
import re
from datetime import datetime
from pathlib import Path

ISO3_VALID = {
    "USD", "EUR", "GBP", "JPY", "PKR", "INR", "AED", "SAR",
    "CNY", "CAD", "AUD", "NZD", "CHF", "SGD", "HKD",
}

KNOWN_TOOLS = {"weather", "calendar", "convert", "currency", "sql"}


def lint_example(ex: dict) -> list:
    errors = []
    messages = ex.get("messages", [])

    if not messages:
        return ["Empty messages list"]

    roles = [m["role"] for m in messages]
    if roles[0] != "system":
        errors.append("First message is not system")

    # Find the last assistant message
    assistant_content = None
    for m in reversed(messages):
        if m["role"] == "assistant":
            assistant_content = m.get("content", "")
            break

    if assistant_content is None:
        return ["No assistant message found"]

    # Plain text refusal — always valid
    if "<tool_call>" not in assistant_content:
        if len(assistant_content.strip()) < 3:
            errors.append("Refusal content too short")
        return errors

    # Has tool_call — validate
    if "<tool_call>" in assistant_content and "</tool_call>" not in assistant_content:
        errors.append("Unclosed <tool_call> tag")
        return errors

    m = re.search(r'<tool_call>(.*?)</tool_call>', assistant_content, re.DOTALL)
    if not m:
        errors.append("tool_call tag found but regex failed to extract content")
        return errors

    try:
        data = json.loads(m.group(1).strip())
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in tool_call: {e}")
        return errors

    tool = data.get("tool")
    args = data.get("args", {})

    if tool not in KNOWN_TOOLS:
        errors.append(f"Unknown tool: '{tool}'")

    if tool == "currency":
        fc = str(args.get("from", "")).upper()
        tc = str(args.get("to", "")).upper()
        if fc not in ISO3_VALID:
            errors.append(f"Invalid 'from' currency code: '{fc}'")
        if tc not in ISO3_VALID:
            errors.append(f"Invalid 'to' currency code: '{tc}'")
        if not isinstance(args.get("amount"), (int, float)):
            errors.append(f"'amount' is not a number: {args.get('amount')!r}")

    if tool == "weather":
        if args.get("unit") not in ("C", "F"):
            errors.append(f"'unit' must be C or F, got: {args.get('unit')!r}")
        if not str(args.get("location", "")).strip():
            errors.append("'location' is empty")

    if tool == "calendar":
        if args.get("action") not in ("list", "create"):
            errors.append(f"'action' must be list|create, got: {args.get('action')!r}")
        date_str = str(args.get("date", ""))
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            errors.append(f"'date' not in YYYY-MM-DD format: {date_str!r}")

    if tool == "convert":
        if not isinstance(args.get("value"), (int, float)):
            errors.append(f"'value' is not a number: {args.get('value')!r}")
        if not str(args.get("from_unit", "")).strip():
            errors.append("'from_unit' is empty")
        if not str(args.get("to_unit", "")).strip():
            errors.append("'to_unit' is empty")

    if tool == "sql":
        query = str(args.get("query", "")).strip()
        if not query:
            errors.append("'query' is empty")
        elif not re.match(
            r'^(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH)\b',
            query.upper()
        ):
            errors.append(f"'query' doesn't start with a valid SQL keyword: {query[:40]!r}")

    return errors


def main():
    src = Path("data/train.jsonl")
    if not src.exists():
        print("ERROR: data/train.jsonl not found. Run: python src/data/generate.py")
        return

    good, bad = [], []

    with open(src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                bad.append((i, [f"JSON parse error: {e}"]))
                continue
            errs = lint_example(ex)
            if errs:
                bad.append((i, errs))
            else:
                good.append(ex)

    print(f"Lint results: {len(good)} passed, {len(bad)} failed")

    if bad:
        print("\nFirst 10 failures:")
        for i, errs in bad[:10]:
            print(f"  Line {i:4d}: {errs}")

    dst = Path("data/train_clean.jsonl")
    with open(dst, "w", encoding="utf-8") as f:
        for ex in good:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nClean dataset: {len(good)} examples → {dst}")

    if len(bad) / max(len(good) + len(bad), 1) > 0.05:
        print("\n⚠️  WARNING: >5% of examples failed lint. Check generate.py.")
    else:
        print("✅ Lint passed — data quality looks good.")


if __name__ == "__main__":
    main()
