"""
demo/app.py
───────────
Gradio chatbot demo. Shows raw tool call JSON to make model behavior visible.

Usage:
    python demo/app.py          # local
    python demo/app.py --share  # public URL (Colab)
"""

import sys
import json
import re
import argparse

sys.path.insert(0, ".")
from inference import run

# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_response(raw: str) -> str:
    """Format the raw model output for display in the chat."""
    m = re.search(r'<tool_call>(.*?)</tool_call>', raw, re.DOTALL)
    if not m:
        return raw  # plain text refusal

    try:
        data = json.loads(m.group(1).strip())
        tool = data.get("tool", "?")
        args = data.get("args", {})
        args_str = "\n".join(f"  {k}: {v}" for k, v in args.items())
        return (
            f"🔧 **Tool call: `{tool}`**\n"
            f"```\n{args_str}\n```\n"
            f"<details><summary>Raw JSON</summary>\n\n"
            f"```json\n{json.dumps(data, indent=2)}\n```\n</details>"
        )
    except json.JSONDecodeError:
        return raw


# ── Chat state ────────────────────────────────────────────────────────────────

_history: list = []   # list of {"role": ..., "content": ...}


def chat(user_message: str, chat_display: list):
    global _history
    raw = run(user_message, _history)
    _history.append({"role": "user", "content": user_message})
    _history.append({"role": "assistant", "content": raw})
    formatted = _format_response(raw)
    chat_display.append((user_message, formatted))
    return "", chat_display


def reset():
    global _history
    _history = []
    return [], []


# ── UI ────────────────────────────────────────────────────────────────────────

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Run: pip install gradio>=4.31.0")
    sys.exit(1)

EXAMPLES = [
    "What's the weather in London in Celsius?",
    "Convert 100 USD to Pakistani rupees",
    "convert 10 km to miles",
    "What do I have tomorrow?",
    "Schedule a team meeting next Monday",
    "Show all users from the database",
    "100 AED to USD",
    "wether in karachi celsius",                # adversarial typo
    "100 dollars ko euros mein convert karo",   # code-switched
    "now do GBP instead",                       # multi-turn (needs prior context)
    "tell me a joke",                           # refusal
    "set a reminder for 8am",                   # refusal
]

DESCRIPTION = """
## 🤖 Pocket-Agent — Offline Mobile Assistant

Routes requests to one of **5 tools** via structured JSON:
`weather` · `calendar` · `convert` · `currency` · `sql`

Try the examples below or type your own. Multi-turn context is maintained within the session.

> **For judges:** You can also call `inference.run(prompt, history)` directly.
> See `test_single.py` for a CLI testing interface.
"""

with gr.Blocks(title="Pocket-Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    chatbot = gr.Chatbot(
        label="Conversation",
        height=450,
        show_copy_button=True,
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="e.g. 'What's the weather in Tokyo in Celsius?' or 'Convert 500 USD to EUR'",
            label="Your message",
            scale=5,
        )
        send = gr.Button("Send ↵", variant="primary", scale=1)

    with gr.Row():
        clear = gr.Button("🗑 Reset conversation")

    gr.Examples(
        examples=EXAMPLES,
        inputs=msg,
        label="Quick examples (click to load)",
    )

    gr.Markdown("""
---
### How to provide your own test cases

**Option 1 — CLI:**
```bash
python test_single.py "What's the weather in Dubai in Celsius?"
```

**Option 2 — Python:**
```python
from inference import run
print(run("Convert 100 USD to EUR", []))
```

**Option 3 — Batch test:**
```bash
python src/eval/score.py --test-file your_test.jsonl
```
Test file format (one JSON per line):
```json
{"prompt": "weather in London in C", "history": [], "expected": "<tool_call>..."}
```
""")

    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    send.click(chat, [msg, chatbot], [msg, chatbot])
    clear.click(reset, [], [chatbot, chatbot])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public Gradio URL")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo.launch(
        share=args.share,
        server_port=args.port,
        show_error=True,
    )
