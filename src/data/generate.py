"""
src/data/generate.py
────────────────────
Generates high-quality synthetic training data for all 5 tools.

Run from repo root:
    python src/data/generate.py

Outputs:
    data/train.jsonl   — full dataset
"""

import json
import random
import hashlib
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

# ── System prompt (MUST match inference.py exactly) ───────────────────────────
TODAY = date.today().isoformat()
SYSTEM = (
    f"Today is {TODAY}. You are an offline mobile assistant.\n"
    "Emit tool calls ONLY as: <tool_call>{\"tool\": \"name\", \"args\": {...}}</tool_call>\n"
    "Available tools:\n"
    "  weather:  {location: string, unit: C|F}\n"
    "  calendar: {action: list|create, date: YYYY-MM-DD, title?: string}\n"
    "  convert:  {value: number, from_unit: string, to_unit: string}\n"
    "  currency: {amount: number, from: ISO3, to: ISO3}\n"
    "  sql:      {query: string}\n"
    "Rules:\n"
    "- Currency codes: uppercase ISO3 only (USD, EUR, GBP, PKR, INR, JPY, AED, CNY, CAD, AUD)\n"
    "- Dates: YYYY-MM-DD format only\n"
    "- Weather unit: C or F only\n"
    "- For chitchat, unknown tools, or ambiguous requests: respond in plain text, NO tool_call tag"
)

REFUSAL_TEXT = (
    "I can help with weather, calendar, unit conversion, "
    "currency exchange, and SQL queries."
)

# ── Hash protection (skip if public_test.jsonl not present) ───────────────────
_banned: set = set()
_public_path = Path("starter/public_test.jsonl")
if _public_path.exists():
    with open(_public_path) as _f:
        _banned = {
            hashlib.sha256(json.loads(l)["prompt"].encode()).hexdigest()
            for l in _f if l.strip()
        }
    print(f"Loaded {len(_banned)} banned hashes from public_test.jsonl")
else:
    print("No public_test.jsonl found — skipping hash protection (safe for synthetic data)")


def _safe(prompt: str) -> bool:
    return hashlib.sha256(prompt.encode()).hexdigest() not in _banned


def _tc(tool: str, args: dict) -> str:
    return f'<tool_call>{json.dumps({"tool": tool, "args": args}, ensure_ascii=False)}</tool_call>'


def _ex(messages: list) -> dict:
    return {"messages": [{"role": "system", "content": SYSTEM}] + messages}


# ── 1. WEATHER ────────────────────────────────────────────────────────────────

CITIES = [
    "London", "Karachi", "Dubai", "New York", "Tokyo", "Paris",
    "Lahore", "Istanbul", "Cairo", "Mumbai", "Berlin", "Toronto",
    "Sydney", "Singapore", "Madrid", "Lagos", "Nairobi", "Bangkok",
    "Riyadh", "Jakarta", "Seoul", "Mexico City", "São Paulo", "Rome",
    "Amsterdam", "Copenhagen", "Stockholm", "Oslo", "Vienna", "Warsaw",
    "Beijing", "Shanghai", "Hong Kong", "Dhaka", "Colombo", "Kabul",
    "Islamabad", "Peshawar", "Quetta", "Multan", "Faisalabad",
]

WEATHER_CLEAN = [
    "What's the weather in {city} in {word}?",
    "What is the weather like in {city} in {word}?",
    "Tell me the weather in {city} in {word}",
    "weather {city} {word}",
    "current weather in {city} in {code}",
    "How hot is it in {city}? Give me {word}",
    "Is it cold in {city}? Show in {word}",
    "Temperature in {city} in {word} please",
    "What should I wear in {city} today? ({word})",
    "{city} weather in {word}",
    "Get me the {word} temperature for {city}",
    "weather forecast {city} {code}",
    "How's the weather in {city}? In {word}",
    "Check weather {city} {code}",
    "Give me {city} weather in {word}",
]

WEATHER_ADVERSARIAL = [
    # Typos
    "wether in {city} {word}",
    "wheather {city} {code}",
    "temprature {city} {word}",
    "whats the wheater in {city} in {word}",
    "waether {city} {code}",
    # Code-switched (Urdu/Hindi mix)
    "{city} mein mausam kaisa hai {word} mein",
    "{city} ka mausam {word} mein batao",
    "{city} mein aaj garmi hai? {word} mein bataiye",
    "aaj {city} mein kitni garmi hai? {code}",
    # Spanish mix
    "¿como esta el clima en {city}? en {word}",
    "clima en {city} {code}",
    # Abbreviated
    "temp {city} {code} plz",
    "wthr {city} {code}",
    # CAPS
    "WEATHER {city} {code}",
    "WHAT IS THE TEMPERATURE IN {city} IN {code}",
    # With units written out
    "show me {city} weather, I need it in {word}",
]


def gen_weather() -> list:
    examples = []
    for _ in range(30):
        city = random.choice(CITIES)
        code = random.choice(["C", "F"])
        word = "Celsius" if code == "C" else "Fahrenheit"
        tpl = random.choice(WEATHER_CLEAN)
        prompt = tpl.format(city=city, word=word, code=code)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("weather", {"location": city, "unit": code})},
            ]))

    for _ in range(20):
        city = random.choice(CITIES)
        code = random.choice(["C", "F"])
        word = "Celsius" if code == "C" else "Fahrenheit"
        tpl = random.choice(WEATHER_ADVERSARIAL)
        prompt = tpl.format(city=city, word=word, code=code)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("weather", {"location": city, "unit": code})},
            ]))
    return examples


# ── 2. CURRENCY ───────────────────────────────────────────────────────────────

CURRENCY_PAIRS = [
    ("USD", "EUR"), ("USD", "GBP"), ("USD", "PKR"), ("USD", "INR"),
    ("USD", "AED"), ("USD", "JPY"), ("USD", "CNY"), ("USD", "CAD"),
    ("EUR", "USD"), ("EUR", "GBP"), ("EUR", "PKR"), ("EUR", "INR"),
    ("GBP", "USD"), ("GBP", "EUR"), ("PKR", "USD"), ("INR", "USD"),
    ("AED", "USD"), ("AED", "PKR"), ("JPY", "USD"), ("CAD", "USD"),
    ("AUD", "USD"), ("AUD", "EUR"), ("CHF", "USD"), ("SAR", "USD"),
]

CURRENCY_NAMES = {
    "USD": ["USD", "dollars", "US dollars", "dollar", "american dollars"],
    "EUR": ["EUR", "euros", "euro"],
    "GBP": ["GBP", "pounds", "British pounds", "sterling", "pound"],
    "PKR": ["PKR", "Pakistani rupees", "rupees", "rupee"],
    "INR": ["INR", "Indian rupees", "Indian rupee"],
    "JPY": ["JPY", "yen", "Japanese yen"],
    "AED": ["AED", "dirhams", "dirham", "UAE dirhams"],
    "CNY": ["CNY", "yuan", "renminbi"],
    "CAD": ["CAD", "Canadian dollars", "Canadian dollar"],
    "AUD": ["AUD", "Australian dollars", "Australian dollar"],
    "CHF": ["CHF", "Swiss francs", "franc"],
    "SAR": ["SAR", "Saudi riyals", "riyal"],
}

AMOUNTS = [1, 5, 10, 20, 50, 100, 150, 200, 250, 500, 750,
           1000, 1500, 2000, 5000, 10000, 25000, 50000]

CURRENCY_CLEAN = [
    "Convert {amt} {fw} to {tw}",
    "How much is {amt} {fw} in {tw}?",
    "{amt} {fc} to {tc}",
    "exchange {amt} {fw} to {tw}",
    "I have {amt} {fw}, convert to {tw}",
    "currency conversion: {amt} {fw} → {tw}",
    "What is {amt} {fw} in {tw}?",
    "change {amt} {fw} to {tw}",
    "convert {amt} {fc} into {tc}",
    "{amt} {fw} = ? {tw}",
    "how many {tw} is {amt} {fw}?",
    "quick convert {amt} {fc} to {tc}",
    "I need to convert {amt} {fw} to {tw}",
    "Give me {amt} {fw} in {tw}",
    "{amt} {fc} in {tc}?",
]

CURRENCY_ADVERSARIAL = [
    # Urdu/Hindi code-switching
    "{amt} {fw} ko {tw} mein convert karo",
    "mujhe {amt} {fw} se {tw} chahiye",
    "{amt} {fw} se {tw} kitne bante hain",
    # Spanish mix
    "¿cuánto son {amt} {fw} en {tw}?",
    "convertir {amt} {fw} a {tw}",
    # Typos
    "conveert {amt} {fw} to {tw}",
    "convertt {amt} {fw} to {tw}",
    # Number formats
    "{amt_comma} {fw} to {tw}",
    # CAPS
    "CONVERT {amt} {fc} TO {tc}",
]


def gen_currency() -> list:
    examples = []
    for _ in range(30):
        fc, tc = random.choice(CURRENCY_PAIRS)
        amt = random.choice(AMOUNTS)
        fw = random.choice(CURRENCY_NAMES[fc])
        tw = random.choice(CURRENCY_NAMES.get(tc, [tc]))
        tpl = random.choice(CURRENCY_CLEAN)
        prompt = tpl.format(amt=amt, fw=fw, tw=tw, fc=fc, tc=tc)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("currency", {"amount": amt, "from": fc, "to": tc})},
            ]))

    for _ in range(20):
        fc, tc = random.choice(CURRENCY_PAIRS)
        amt = random.choice(AMOUNTS)
        fw = random.choice(CURRENCY_NAMES[fc])
        tw = random.choice(CURRENCY_NAMES.get(tc, [tc]))
        amt_comma = f"{amt:,}"
        tpl = random.choice(CURRENCY_ADVERSARIAL)
        prompt = tpl.format(amt=amt, fw=fw, tw=tw, fc=fc, tc=tc, amt_comma=amt_comma)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("currency", {"amount": amt, "from": fc, "to": tc})},
            ]))
    return examples


# ── 3. CONVERT ────────────────────────────────────────────────────────────────

UNIT_PAIRS = [
    # Length
    (1.0, "km", "miles"), (5.0, "km", "miles"), (10.0, "km", "miles"),
    (42.195, "km", "miles"), (1.0, "miles", "km"), (10.0, "miles", "km"),
    (1.0, "m", "ft"), (100.0, "m", "yards"), (1.0, "ft", "m"),
    (1.0, "inch", "cm"), (30.0, "cm", "inch"), (6.0, "ft", "m"),
    # Weight
    (1.0, "kg", "lb"), (70.0, "kg", "lb"), (0.5, "kg", "g"),
    (100.0, "g", "oz"), (1.0, "lb", "kg"), (500.0, "g", "kg"),
    (10.0, "kg", "lb"), (5.0, "lb", "kg"), (200.0, "g", "oz"),
    # Volume
    (1.0, "liter", "gallon"), (5.0, "liter", "gallon"),
    (1.0, "gallon", "liter"), (250.0, "ml", "liter"),
    # Area
    (1.0, "sqm", "sqft"), (100.0, "sqft", "sqm"), (1.0, "acre", "sqm"),
    # Misc
    (1.0, "ton", "kg"), (1000.0, "kg", "ton"),
]

CONVERT_CLEAN = [
    "Convert {v} {fu} to {tu}",
    "How many {tu} is {v} {fu}?",
    "{v} {fu} in {tu}",
    "change {v} {fu} to {tu}",
    "what is {v} {fu} in {tu}?",
    "{v} {fu} to {tu} please",
    "I need {v} {fu} in {tu}",
    "convert {v} {fu} into {tu}",
    "{v} {fu} = how many {tu}?",
    "how much is {v} {fu} in {tu}",
    "unit conversion: {v} {fu} to {tu}",
    "give me {v} {fu} in {tu}",
]

CONVERT_ADVERSARIAL = [
    # Typos
    "conveert {v} {fu} to {tu}",
    "converrt {v} {fu} into {tu}",
    # Urdu/Hindi
    "{v} {fu} ko {tu} mein convert karo",
    "{v} {fu} se {tu} kitne hote hain",
    # Arrows
    "{v} {fu} → {tu}",
    "{v} {fu} => {tu}",
    # CAPS
    "CONVERT {v} {fu} TO {tu}",
]


def gen_convert() -> list:
    examples = []
    for _ in range(25):
        v, fu, tu = random.choice(UNIT_PAIRS)
        tpl = random.choice(CONVERT_CLEAN)
        prompt = tpl.format(v=v, fu=fu, tu=tu)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("convert", {"value": v, "from_unit": fu, "to_unit": tu})},
            ]))

    for _ in range(15):
        v, fu, tu = random.choice(UNIT_PAIRS)
        tpl = random.choice(CONVERT_ADVERSARIAL)
        prompt = tpl.format(v=v, fu=fu, tu=tu)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("convert", {"value": v, "from_unit": fu, "to_unit": tu})},
            ]))
    return examples


# ── 4. CALENDAR ───────────────────────────────────────────────────────────────

def _future_date(offset_days: int) -> str:
    return (date.today() + timedelta(days=offset_days)).isoformat()


def _next_weekday(n: int) -> str:
    today = date.today()
    days = n - today.weekday()
    if days <= 0:
        days += 7
    return (today + timedelta(days=days)).isoformat()


DATE_POOL = [
    (date.today().isoformat(), ["today", "tonight"]),
    (_future_date(1), ["tomorrow"]),
    (_future_date(2), ["in two days", "day after tomorrow"]),
    (_future_date(7), ["next week", "in a week"]),
    (_next_weekday(0), ["next Monday", "coming Monday", "this Monday"]),
    (_next_weekday(1), ["next Tuesday", "coming Tuesday"]),
    (_next_weekday(2), ["next Wednesday", "coming Wednesday"]),
    (_next_weekday(3), ["next Thursday", "coming Thursday"]),
    (_next_weekday(4), ["next Friday", "coming Friday", "this Friday"]),
    (_next_weekday(5), ["next Saturday", "this Saturday"]),
    (_next_weekday(6), ["next Sunday", "this weekend", "this Sunday"]),
    (_future_date(30), ["next month", "in a month"]),
]

LIST_TEMPLATES = [
    "What do I have {expr}?",
    "What's on my calendar {expr}?",
    "Show my schedule for {expr}",
    "List my events for {expr}",
    "Any meetings {expr}?",
    "What are my appointments {expr}?",
    "Check my calendar {expr}",
    "What's scheduled for {expr}?",
    "Do I have anything {expr}?",
    "Show me my agenda for {expr}",
    "calendar {expr}",
    "events {expr}",
]

CREATE_TEMPLATES = [
    ("Schedule {title} for {expr}", "{title}"),
    ("Add '{title}' to my calendar on {expr}", "{title}"),
    ("Create event '{title}' on {expr}", "{title}"),
    ("Book a {title} for {expr}", "{title}"),
    ("Set up {title} on {expr}", "{title}"),
    ("Add {title} {expr}", "{title}"),
    ("Put {title} on my calendar for {expr}", "{title}"),
    ("Can you schedule {title} for {expr}?", "{title}"),
    ("I need to add {title} on {expr}", "{title}"),
    ("New event: {title} on {expr}", "{title}"),
]

EVENT_TITLES = [
    "team standup", "doctor appointment", "dentist", "lunch with Sarah",
    "project review", "call with client", "gym session", "interview",
    "birthday party", "dinner", "weekly sync", "code review",
    "meeting with manager", "presentation", "workshop", "training session",
    "flight check-in", "video call", "coffee with Ahmed", "hackathon",
]


def gen_calendar() -> list:
    examples = []
    for _ in range(20):
        iso, exprs = random.choice(DATE_POOL)
        expr = random.choice(exprs)
        tpl = random.choice(LIST_TEMPLATES)
        prompt = tpl.format(expr=expr)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("calendar", {"action": "list", "date": iso})},
            ]))

    for _ in range(20):
        iso, exprs = random.choice(DATE_POOL)
        expr = random.choice(exprs)
        title = random.choice(EVENT_TITLES)
        tpl, title_tpl = random.choice(CREATE_TEMPLATES)
        actual_title = title_tpl.format(title=title)
        prompt = tpl.format(title=title, expr=expr)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("calendar", {
                    "action": "create", "date": iso, "title": actual_title
                })},
            ]))
    return examples


# ── 5. SQL ────────────────────────────────────────────────────────────────────

SQL_EXAMPLES = [
    ("Show all users", "SELECT * FROM users"),
    ("How many users are there?", "SELECT COUNT(*) FROM users"),
    ("Get users who signed up this month",
     "SELECT * FROM users WHERE created_at >= date('now', 'start of month')"),
    ("List pending orders", "SELECT * FROM orders WHERE status = 'pending'"),
    ("How many orders are pending?",
     "SELECT COUNT(*) FROM orders WHERE status = 'pending'"),
    ("Show top 10 products by sales",
     "SELECT * FROM products ORDER BY sales DESC LIMIT 10"),
    ("Find customers from Karachi",
     "SELECT * FROM customers WHERE city = 'Karachi'"),
    ("Show revenue for last 30 days",
     "SELECT SUM(amount) FROM orders WHERE created_at >= date('now', '-30 days')"),
    ("Get all failed transactions",
     "SELECT * FROM transactions WHERE status = 'failed'"),
    ("Show active subscriptions",
     "SELECT * FROM subscriptions WHERE status = 'active'"),
    ("Count products in inventory", "SELECT COUNT(*) FROM inventory"),
    ("List employees in engineering",
     "SELECT * FROM employees WHERE department = 'engineering'"),
    ("Show me the last 5 logins",
     "SELECT * FROM logins ORDER BY created_at DESC LIMIT 5"),
    ("Get users with premium plans",
     "SELECT * FROM users WHERE plan = 'premium'"),
    ("run this: SELECT * FROM logs WHERE level = 'error'",
     "SELECT * FROM logs WHERE level = 'error'"),
    ("execute: SELECT name, email FROM users LIMIT 5",
     "SELECT name, email FROM users LIMIT 5"),
    ("SELECT * FROM sales WHERE month = '2024-01'",
     "SELECT * FROM sales WHERE month = '2024-01'"),
    ("get all products where price > 100",
     "SELECT * FROM products WHERE price > 100"),
    ("show monthly revenue",
     "SELECT strftime('%Y-%m', created_at) as month, SUM(amount) FROM orders GROUP BY month"),
    ("find duplicate emails",
     "SELECT email, COUNT(*) FROM users GROUP BY email HAVING COUNT(*) > 1"),
]


def gen_sql() -> list:
    examples = []
    for prompt, query in SQL_EXAMPLES:
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("sql", {"query": query})},
            ]))
    return examples


# ── 6. REFUSALS ───────────────────────────────────────────────────────────────

REFUSAL_PROMPTS = [
    # Chitchat
    "how are you", "what's up", "good morning", "hey there", "hello",
    "thanks", "thank you", "bye", "goodbye", "nice to meet you",
    "how's your day", "are you sentient", "what's your name",
    "do you have feelings", "tell me a joke", "make me laugh",
    "what do you think about AI", "are you a robot",

    # Non-existent tools
    "set a reminder for 8am", "call mom for me", "send an email to John",
    "search the web for news", "navigate to downtown",
    "play some music", "take a photo", "translate this to French",
    "open maps", "book me a flight to Dubai",
    "what's on Netflix", "search youtube", "send a WhatsApp",
    "book a restaurant", "check flight status PK301",
    "turn on the lights", "set thermostat to 22",

    # Hallucination bait (fake tools)
    "use the stocks tool to get AAPL price",
    "use the sensor tool to check temperature",
    "activate home automation",
    "use the news tool",
    "check my email using the email tool",

    # Knowledge questions (model should not tool-call these)
    "what is quantum computing",
    "who discovered gravity",
    "what is the capital of France",
    "explain machine learning",
    "how does GPS work",
    "what is blockchain",

    # Ambiguous (no history)
    "convert that",
    "do it again",
    "same thing",
    "what was the result",
    "change it to euros",
    "show me more",
    "repeat that",

    # Subtle traps — look like tool calls but aren't
    "what's the exchange rate",        # no amount → can't call currency
    "how's the weather",               # no location → can't call weather
    "check my calendar",               # no date → ambiguous
    "run a query",                     # no query body → can't call sql
    "show me the conversion",          # no value/units → can't call convert
]


def gen_refusals() -> list:
    examples = []
    for prompt in REFUSAL_PROMPTS:
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": REFUSAL_TEXT},
            ]))
    return examples


# ── 7. MULTI-TURN ─────────────────────────────────────────────────────────────

def gen_multiturn() -> list:
    today_iso = date.today().isoformat()
    tomorrow_iso = _future_date(1)
    examples = []

    convs = [
        # Currency reference resolution
        {
            "history": [
                ("Convert 100 USD to EUR",
                 _tc("currency", {"amount": 100, "from": "USD", "to": "EUR"})),
            ],
            "prompt": "now do GBP instead",
            "answer": _tc("currency", {"amount": 100, "from": "USD", "to": "GBP"}),
        },
        {
            "history": [
                ("How much is 500 dollars in rupees?",
                 _tc("currency", {"amount": 500, "from": "USD", "to": "PKR"})),
            ],
            "prompt": "what about in euros?",
            "answer": _tc("currency", {"amount": 500, "from": "USD", "to": "EUR"}),
        },
        {
            "history": [
                ("100 USD to AED",
                 _tc("currency", {"amount": 100, "from": "USD", "to": "AED"})),
            ],
            "prompt": "how about 200?",
            "answer": _tc("currency", {"amount": 200, "from": "USD", "to": "AED"}),
        },
        # Weather reference
        {
            "history": [
                ("weather in Karachi in Celsius",
                 _tc("weather", {"location": "Karachi", "unit": "C"})),
            ],
            "prompt": "what about Dubai?",
            "answer": _tc("weather", {"location": "Dubai", "unit": "C"}),
        },
        {
            "history": [
                ("weather in London in Fahrenheit",
                 _tc("weather", {"location": "London", "unit": "F"})),
            ],
            "prompt": "and Tokyo?",
            "answer": _tc("weather", {"location": "Tokyo", "unit": "F"}),
        },
        # Convert reference
        {
            "history": [
                ("convert 10 km to miles",
                 _tc("convert", {"value": 10, "from_unit": "km", "to_unit": "miles"})),
            ],
            "prompt": "now do 20 km",
            "answer": _tc("convert", {"value": 20, "from_unit": "km", "to_unit": "miles"}),
        },
        {
            "history": [
                ("5 kg to lb",
                 _tc("convert", {"value": 5, "from_unit": "kg", "to_unit": "lb"})),
            ],
            "prompt": "and 10 kg?",
            "answer": _tc("convert", {"value": 10, "from_unit": "kg", "to_unit": "lb"}),
        },
        # Calendar multi-step
        {
            "history": [
                ("What's on my calendar today?",
                 _tc("calendar", {"action": "list", "date": today_iso})),
                ("Looks free. Schedule a team meeting",
                 _tc("calendar", {"action": "create", "date": today_iso, "title": "team meeting"})),
            ],
            "prompt": "also add lunch for the same day",
            "answer": _tc("calendar", {"action": "create", "date": today_iso, "title": "lunch"}),
        },
        # 3-turn: weather → convert → currency
        {
            "history": [
                ("weather in Tokyo in Celsius",
                 _tc("weather", {"location": "Tokyo", "unit": "C"})),
                ("convert 5 km to miles",
                 _tc("convert", {"value": 5, "from_unit": "km", "to_unit": "miles"})),
            ],
            "prompt": "100 USD to JPY",
            "answer": _tc("currency", {"amount": 100, "from": "USD", "to": "JPY"}),
        },
        # Refusal after a valid turn (context switch)
        {
            "history": [
                ("weather in Tokyo in Celsius",
                 _tc("weather", {"location": "Tokyo", "unit": "C"})),
            ],
            "prompt": "tell me a joke now",
            "answer": REFUSAL_TEXT,
        },
        {
            "history": [
                ("100 USD to EUR",
                 _tc("currency", {"amount": 100, "from": "USD", "to": "EUR"})),
            ],
            "prompt": "play some music",
            "answer": REFUSAL_TEXT,
        },
        # Calendar then currency (different tools)
        {
            "history": [
                ("What meetings do I have tomorrow?",
                 _tc("calendar", {"action": "list", "date": tomorrow_iso})),
            ],
            "prompt": "also convert 200 USD to PKR",
            "answer": _tc("currency", {"amount": 200, "from": "USD", "to": "PKR"}),
        },
        # Ambiguous reference SHOULD work with history
        {
            "history": [
                ("Convert 1 liter to gallon",
                 _tc("convert", {"value": 1, "from_unit": "liter", "to_unit": "gallon"})),
            ],
            "prompt": "now do 5 liters",
            "answer": _tc("convert", {"value": 5, "from_unit": "liter", "to_unit": "gallon"}),
        },
        # SQL follow-up
        {
            "history": [
                ("show all users",
                 _tc("sql", {"query": "SELECT * FROM users"})),
            ],
            "prompt": "now filter by active status",
            "answer": _tc("sql", {"query": "SELECT * FROM users WHERE status = 'active'"}),
        },
    ]

    for conv in convs:
        messages = []
        for user_msg, asst_msg in conv["history"]:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": asst_msg})
        messages.append({"role": "user", "content": conv["prompt"]})
        messages.append({"role": "assistant", "content": conv["answer"]})
        examples.append(_ex(messages))

    return examples


# ── 8. MULTILINGUAL (Slice C target) ─────────────────────────────────────────
# Covers Urdu/Hindi, Arabic (romanised + script), Spanish, Turkish, Punjabi.
# These directly map to the adversarial bonus slice.

# Urdu/Hindi weather phrases: {city}, {unit_word}, {unit_code}
URDU_WEATHER = [
    "{city} mein aaj mausam kaisa hai? {unit_word} mein batao",
    "{city} ka mausam {unit_word} mein chahiye",
    "aaj {city} mein kitni garmi hai {unit_code} mein",
    "{city} mein sardi hai? {unit_word} mein bataiye",
    "{city} ka temperature {unit_word} mein kya hai",
    "mujhe {city} ka mausam {unit_code} mein chahiye",
    "{city} weather {unit_word} mein please",
    "aaj {city} mein mausam kya hoga {unit_word} mein",
    "{city} mein barishh hogi? {unit_word} mein batao",
    "yaar {city} ka mausam {unit_code} mein kya hai",
    # Punjabi mix
    "{city} wich aaj mausam ki haal a {unit_word} wich",
    # Urdu script
    "{city} میں موسم {unit_word} میں بتاؤ",
    "{city} کا درجہ حرارت {unit_code} میں کیا ہے",
]

# Arabic romanised + script weather: {city}, {unit_word}, {unit_code}
ARABIC_WEATHER = [
    "ma howa altaqs fi {city} bil {unit_word}",
    "kaifa altaqs fi {city} {unit_code}",
    "akhbirni an taqs {city} bil {unit_word}",
    "hal al jaw fi {city} harin? {unit_word}",
    "areed taqs {city} {unit_code} min fadlak",
    # Arabic script
    "ما هو الطقس في {city} بالـ{unit_word}؟",
    "كيف الطقس في {city} بالـ{unit_code}؟",
    "هل الجو حار في {city}؟ بالـ{unit_word}",
]

# Spanish weather: {city}, {unit_word}, {unit_code}
SPANISH_WEATHER = [
    "¿Cuál es el clima en {city} en {unit_word}?",
    "clima en {city} {unit_code}",
    "temperatura en {city} en {unit_word}",
    "¿qué temperatura hace en {city} en {unit_word}?",
    "dime el tiempo en {city} en {unit_code}",
    "¿cómo está el clima en {city} en {unit_word}?",
    "pronóstico del tiempo en {city} {unit_code}",
]

# Turkish weather: {city}, {unit_word}, {unit_code}
TURKISH_WEATHER = [
    "{city} hava durumu {unit_word} cinsinden",
    "{city}'da hava nasıl? {unit_code}",
    "{city} sıcaklık {unit_word} olarak",
]

# Urdu/Hindi currency: {amt}, {fc_word}, {tc_word}, {fc}, {tc}
URDU_CURRENCY = [
    "{amt} {fc_word} ko {tc_word} mein convert karo",
    "mujhe {amt} {fc_word} se {tc_word} chahiye",
    "{amt} {fc_word} se {tc_word} kitne bante hain",
    "{amt} {fc_word} ka {tc_word} mein kya rate hai",
    "bhai {amt} {fc_word} kitne {tc_word} hain",
    "{amt} {fc} ko {tc} mein badlo",
    "{amt} {fc_word} {tc_word} mein calculate karo",
    # Urdu script
    "{amt} {fc_word} کو {tc_word} میں کنورٹ کرو",
    "{amt} {fc_word} کتنے {tc_word} ہیں",
]

# Arabic romanised currency: {amt}, {fc_word}, {tc_word}, {fc}, {tc}
ARABIC_CURRENCY = [
    "km yusawi {amt} {fc_word} bil {tc_word}",
    "tahwil {amt} {fc_word} ila {tc_word}",
    "areed tahwil {amt} {fc} ila {tc}",
    # Arabic script
    "كم يساوي {amt} {fc_word} بالـ{tc_word}؟",
    "حوّل {amt} {fc_word} إلى {tc_word}",
]

# Spanish currency
SPANISH_CURRENCY = [
    "¿Cuántos {tc_word} son {amt} {fc_word}?",
    "convertir {amt} {fc_word} a {tc_word}",
    "¿cuánto son {amt} {fc_word} en {tc_word}?",
    "¿cuántas {tc_word} equivalen a {amt} {fc_word}?",
    "necesito cambiar {amt} {fc_word} a {tc_word}",
]

# Urdu/Hindi convert: {v}, {fu}, {tu}
URDU_CONVERT = [
    "{v} {fu} ko {tu} mein convert karo",
    "{v} {fu} se {tu} kitne hote hain",
    "{v} {fu} {tu} mein batao",
    "bhai {v} {fu} kitne {tu} hain",
    "{v} {fu} to {tu} mein ek dum",
    # Urdu script
    "{v} {fu} کو {tu} میں کنورٹ کرو",
]

# Spanish convert
SPANISH_CONVERT = [
    "convertir {v} {fu} a {tu}",
    "¿cuántas {tu} son {v} {fu}?",
    "¿cuántos {tu} tiene {v} {fu}?",
    "{v} {fu} en {tu}",
]

# Urdu calendar: date expressions and event titles
URDU_CALENDAR_LIST = [
    "kal ka schedule dikhao",
    "aaj kya kya hai mere calendar mein",
    "is hafte ke meetings dikhao",
    "aaj ka agenda kya hai",
    "kal koi meeting hai?",
    "اگلے ہفتے کا شیڈول دکھاؤ",
]

URDU_CALENDAR_CREATE = [
    # (prompt_template, title_value)
    ("Monday ko {title} ka meeting add karo", "meeting"),
    ("kal {title} schedule karo", "{title}"),
    ("is Friday ko {title} set karo", "{title}"),
    ("{title} ko calendar mein add karo kal ke liye", "{title}"),
    ("آج {title} کیلنڈر میں ڈالو", "{title}"),
]

# Multilingual refusals (model must NOT emit tool_call for these)
MULTILINGUAL_REFUSALS = [
    # Urdu/Hindi chitchat
    "shukriya", "shukria", "shukriya bhai", "theek hai",
    "kya haal hai", "aap kaise hain", "assalamu alaikum",
    "khuda hafiz", "alvida", "goodnight bhai",
    "joke sunao", "kuch funny batao",
    "mujhe navigate karo downtown tak",
    "music lagao", "gaana bajao",
    "WhatsApp pe message karo Ahmed ko",
    "reminder set karo 8 baje ka",
    "alarm lagao kal subah 7 baje",
    "yahan ka mausam",           # no city → ambiguous refusal
    "kuch bhi convert karo",     # no args → ambiguous refusal
    # Arabic chitchat
    "shukran", "shukran jazeelan", "marhaba", "ahlan",
    "kaifa haluka", "ma ismuka", "wada", "ma'a salama",
    "قل لي نكتة",
    "ابحث في الإنترنت",
    "شغّل موسيقى",
    # Spanish chitchat
    "¿Cómo estás?", "gracias", "hola", "adiós",
    "cuéntame un chiste",
    "busca en internet",
    "pon música",
    "¿cuál es la capital de Francia?",
    # Turkish chitchat
    "nasılsın", "teşekkürler", "merhaba", "güle güle",
]


CURRENCY_WORDS = {
    "USD": ["dollar", "dolar", "doları", "دولار", "dollars"],
    "EUR": ["euro", "euros", "يورو", "euros"],
    "GBP": ["pound", "sterlin", "جنيه", "pounds"],
    "PKR": ["rupee", "rupees", "روپے", "روبية"],
    "INR": ["Indian rupee", "rupay", "روپیہ"],
    "JPY": ["yen", "yen", "ين"],
    "AED": ["dirham", "درہم", "درهم", "dirhams"],
    "CNY": ["yuan", "yuan", "يوان"],
    "CAD": ["Canadian dollar", "Canadian dollars"],
    "AUD": ["Australian dollar", "Australian dollars"],
    "CHF": ["franc", "francs"],
    "SAR": ["riyal", "rial", "ريال"],
}

CONVERT_WORD_PAIRS = [
    # Urdu/Hindi natural language unit names
    (10.0, "kilometer", "miles"),
    (5.0, "km", "miles"),
    (1.0, "mile", "km"),
    (70.0, "kilo", "lb"),
    (1.0, "kilo", "g"),
    (100.0, "gram", "oz"),
    (5.0, "liter", "gallon"),
    (1.0, "gallon", "liter"),
    (1.0, "foot", "m"),
    (30.0, "cm", "inch"),
]


def gen_multilingual() -> list:
    examples = []
    ML_CITIES = [
        "Karachi", "Lahore", "Dubai", "Riyadh", "Istanbul", "Cairo",
        "London", "New York", "Tokyo", "Madrid", "Paris", "Mumbai",
        "Islamabad", "Berlin", "Singapore", "Abu Dhabi", "Ankara",
    ]
    ML_CURRENCY_PAIRS = [
        ("USD", "PKR"), ("USD", "AED"), ("USD", "EUR"), ("USD", "INR"),
        ("GBP", "PKR"), ("EUR", "PKR"), ("AED", "PKR"), ("SAR", "PKR"),
        ("USD", "SAR"), ("AED", "USD"), ("EUR", "AED"),
    ]
    ML_AMOUNTS = [50, 100, 200, 500, 1000, 2000, 5000]

    # ── Urdu/Hindi weather ────────────────────────────────────────────────────
    for tpl in URDU_WEATHER:
        city = random.choice(ML_CITIES)
        code = random.choice(["C", "F"])
        unit_word = "Celsius" if code == "C" else "Fahrenheit"
        prompt = tpl.format(city=city, unit_word=unit_word, unit_code=code)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("weather", {"location": city, "unit": code})},
            ]))

    # ── Arabic weather ────────────────────────────────────────────────────────
    for tpl in ARABIC_WEATHER:
        city = random.choice(ML_CITIES)
        code = random.choice(["C", "F"])
        unit_word = "celsius" if code == "C" else "fahrenheit"
        prompt = tpl.format(city=city, unit_word=unit_word, unit_code=code)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("weather", {"location": city, "unit": code})},
            ]))

    # ── Spanish weather ───────────────────────────────────────────────────────
    for tpl in SPANISH_WEATHER:
        city = random.choice(ML_CITIES)
        code = random.choice(["C", "F"])
        unit_word = "Celsius" if code == "C" else "Fahrenheit"
        prompt = tpl.format(city=city, unit_word=unit_word, unit_code=code)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("weather", {"location": city, "unit": code})},
            ]))

    # ── Turkish weather ───────────────────────────────────────────────────────
    for tpl in TURKISH_WEATHER:
        city = random.choice(ML_CITIES)
        code = random.choice(["C", "F"])
        unit_word = "Celsius" if code == "C" else "Fahrenheit"
        prompt = tpl.format(city=city, unit_word=unit_word, unit_code=code)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("weather", {"location": city, "unit": code})},
            ]))

    # ── Urdu/Hindi currency ───────────────────────────────────────────────────
    for tpl in URDU_CURRENCY:
        fc, tc = random.choice(ML_CURRENCY_PAIRS)
        amt = random.choice(ML_AMOUNTS)
        fc_word = random.choice(CURRENCY_WORDS.get(fc, [fc]))
        tc_word = random.choice(CURRENCY_WORDS.get(tc, [tc]))
        prompt = tpl.format(amt=amt, fc_word=fc_word, tc_word=tc_word, fc=fc, tc=tc)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("currency", {"amount": amt, "from": fc, "to": tc})},
            ]))

    # ── Arabic currency ───────────────────────────────────────────────────────
    for tpl in ARABIC_CURRENCY:
        fc, tc = random.choice(ML_CURRENCY_PAIRS)
        amt = random.choice(ML_AMOUNTS)
        fc_word = random.choice(CURRENCY_WORDS.get(fc, [fc]))
        tc_word = random.choice(CURRENCY_WORDS.get(tc, [tc]))
        prompt = tpl.format(amt=amt, fc_word=fc_word, tc_word=tc_word, fc=fc, tc=tc)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("currency", {"amount": amt, "from": fc, "to": tc})},
            ]))

    # ── Spanish currency ──────────────────────────────────────────────────────
    for tpl in SPANISH_CURRENCY:
        fc, tc = random.choice(ML_CURRENCY_PAIRS)
        amt = random.choice(ML_AMOUNTS)
        fc_word = random.choice(CURRENCY_WORDS.get(fc, [fc]))
        tc_word = random.choice(CURRENCY_WORDS.get(tc, [tc]))
        prompt = tpl.format(amt=amt, fc_word=fc_word, tc_word=tc_word, fc=fc, tc=tc)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("currency", {"amount": amt, "from": fc, "to": tc})},
            ]))

    # ── Urdu/Hindi unit convert ───────────────────────────────────────────────
    for tpl in URDU_CONVERT:
        v, fu, tu = random.choice(CONVERT_WORD_PAIRS)
        prompt = tpl.format(v=v, fu=fu, tu=tu)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("convert", {"value": v, "from_unit": fu, "to_unit": tu})},
            ]))

    # ── Spanish unit convert ──────────────────────────────────────────────────
    for tpl in SPANISH_CONVERT:
        v, fu, tu = random.choice(CONVERT_WORD_PAIRS)
        prompt = tpl.format(v=v, fu=fu, tu=tu)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("convert", {"value": v, "from_unit": fu, "to_unit": tu})},
            ]))

    # ── Urdu calendar ─────────────────────────────────────────────────────────
    today_iso = date.today().isoformat()
    tomorrow_iso = (date.today() + timedelta(days=1)).isoformat()
    next_mon_iso = _next_weekday(0)
    next_fri_iso = _next_weekday(4)

    for prompt in URDU_CALENDAR_LIST:
        iso = random.choice([today_iso, tomorrow_iso])
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("calendar", {"action": "list", "date": iso})},
            ]))

    for tpl, title_val in URDU_CALENDAR_CREATE:
        events = ["standup", "meeting", "lunch", "review", "call"]
        title = random.choice(events)
        iso = random.choice([next_mon_iso, next_fri_iso, tomorrow_iso])
        prompt = tpl.format(title=title)
        actual_title = title_val.format(title=title)
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _tc("calendar", {
                    "action": "create", "date": iso, "title": actual_title
                })},
            ]))

    # ── Multilingual refusals ─────────────────────────────────────────────────
    for prompt in MULTILINGUAL_REFUSALS:
        if _safe(prompt):
            examples.append(_ex([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": REFUSAL_TEXT},
            ]))

    return examples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path("data").mkdir(exist_ok=True)

    weather      = gen_weather()        # ~50
    currency     = gen_currency()       # ~50
    convert      = gen_convert()        # ~40
    calendar     = gen_calendar()       # ~40
    sql          = gen_sql()            # ~20
    refusals     = gen_refusals()       # ~58
    multiturn    = gen_multiturn()      # 14
    multilingual = gen_multilingual()   # ~120

    all_examples = (
        weather + currency + convert + calendar
        + sql + refusals + multiturn + multilingual
    )

    # Oversample adversarial (Slice C) and refusals (prevents -0.5)
    ADVERSARIAL_MARKERS = [
        "mein", "karo", "chahiye", "batao", "bataiye",  # Urdu/Hindi
        "hain", "kitne", "mausam",                       # Urdu/Hindi
        "¿", "cuánto", "convertir", "clima",            # Spanish
        "shukran", "marhaba", "tahwil", "taqs",         # Arabic
        "nasil", "teşekkür",                             # Turkish
        "conveert", "wheather", "wether", "temprature", # typos
        "→", "=>",                                       # symbols
        "CONVERT", "WEATHER",                            # CAPS
        "میں", "کو", "ما هو", "كيف",                   # scripts
    ]

    adversarial = [
        e for e in all_examples
        if any(
            marker in (e["messages"][-2]["content"] if len(e["messages"]) >= 2 else "")
            for marker in ADVERSARIAL_MARKERS
        )
    ]
    refusal_examples = [
        e for e in all_examples
        if e["messages"][-1]["content"] == REFUSAL_TEXT
    ]

    # 3x oversample multilingual adversarial, 2x refusals
    all_examples = all_examples + adversarial * 2 + refusal_examples

    random.shuffle(all_examples)

    out = Path("data/train.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    counts = {
        "weather": len(weather),
        "currency": len(currency),
        "convert": len(convert),
        "calendar": len(calendar),
        "sql": len(sql),
        "refusals": len(refusals),
        "multiturn": len(multiturn),
        "multilingual": len(multilingual),
        "adversarial (2x oversampled)": len(adversarial),
        "refusals (oversampled)": len(refusal_examples),
        "TOTAL after oversampling": len(all_examples),
    }

    print("Data generation complete:")
    for k, v in counts.items():
        print(f"  {k:35s}: {v}")
    print(f"\nSaved {len(all_examples)} examples → {out}")


if __name__ == "__main__":
    main()
