#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "anthropic",
# ]
# ///
#
# Run with:  uv run agent1-claude-proper.py   (uv installs deps into an isolated
# env). Also requires:  export ANTHROPIC_API_KEY=...   (or an `ant auth login`
# profile — see agents/agent1.py.md and the vault note on Claude auth).

# weather-agent — "proper" Claude variant: NATIVE tool calling (no text parsing)
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT CHANGED vs agent1-claude.py
# ─────────────────────────────────────────────────────────────────────────────
# agent1-claude.py is a 1:1 port of the hand-rolled TAO agent: the model is told
# to emit literal `Thought:` / `Action:` / `Args:` / `Final:` text, and Python
# recovers the tool call by STRING-SPLITTING that text and decides "done" by
# searching for the substring "Final:".
#
# This version does it the way the Anthropic API is designed to be used:
#
#   • Tools are declared as JSON SCHEMAS in `tools=[...]`. The model returns a
#     structured `tool_use` content block (name + already-parsed `input` dict) —
#     no prose to split, no `json.loads` on a hand-cut substring.
#   • Loop control is driven by `response.stop_reason`, a real signal from the
#     API, NOT by grepping the text:
#         stop_reason == "tool_use"  → the model wants a tool; run it, send back a
#                                      `tool_result`, loop again.
#         stop_reason == "end_turn"  → the model is done; its text IS the answer.
#         stop_reason == "refusal"   → declined for safety; stop.
#   • `strict: True` on the tool guarantees the `input` validates against the
#     schema (lat/lon are always present and numeric).
#
# Net effect: the same Thought→Action→Observation idea, but the boundaries are
# carried by the protocol instead of by fragile string parsing. Compare the
# run trace here with agent1-claude.py to see the difference.
#
# (The SDK also ships a higher-level `client.beta.messages.tool_runner()` that
#  drives this loop for you via an `@beta_tool` decorator; the MANUAL loop below
#  is written out on purpose so the tool-detection + stop-reason mechanics are
#  visible — which is the whole point of this file.)
# ─────────────────────────────────────────────────────────────────────────────

import json
import requests
import time
import anthropic

# ── 1. Open-Meteo weather-code lookup ──────────────────────────────────────
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


# ── 2. Tools (plain Python functions) ──────────────────────────────────────
def get_weather(lat: float, lon: float) -> dict:
    """
    Return today's forecast:
        { "high": °C, "low": °C, "conditions": str }
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=weathercode,temperature_2m_max,temperature_2m_min"
        "&forecast_days=1&timezone=auto"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            daily = r.json()["daily"]
            return {
                "high": daily["temperature_2m_max"][0],
                "low": daily["temperature_2m_min"][0],
                "conditions": WEATHER_CODES.get(daily["weathercode"][0], "Unknown"),
            }
        except (requests.Timeout, requests.ConnectionError):
            if attempt == max_retries - 1:
                raise
            print(f"  ⚠️  Retry {attempt + 1}/{max_retries - 1} after timeout...")
            time.sleep(2)

    # Unreachable: the final attempt always returns a dict or re-raises. The
    # explicit raise makes every code path return-or-raise so the type checker
    # (Pylance reportReturnType) doesn't see an implicit `return None`.
    raise RuntimeError("get_weather: retries exhausted without a result")


# Name → Python function, so the loop can dispatch a tool call by its name.
TOOLS = {
    "get_weather": get_weather,
}

# ── 3. Tool SCHEMAS (what the model sees) ──────────────────────────────────
# This is the structured contract — the model receives these JSON schemas and
# emits a matching `tool_use` block. `strict: True` (requires additionalProperties
# False + a full `required` list) guarantees the `input` validates exactly, so we
# never get a malformed/partial argument set.
TOOLS_SCHEMA = [
    {
        "name": "get_weather",
        "description": (
            "Get today's weather forecast (high °C, low °C, and a short conditions "
            "description) for a geographic coordinate."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {
                    "type": "number",
                    "description": "Latitude in decimal degrees, e.g. 51.5074",
                },
                "lon": {
                    "type": "number",
                    # The hint about sign mitigates the Q2 geocoding bug, where the
                    # model put Valencia at +0.38 (east) instead of -0.38 (west).
                    "description": "Longitude in decimal degrees; negative = West, e.g. -0.1278",
                },
            },
            "required": ["lat", "lon"],
            "additionalProperties": False,
        },
    }
]

# ── 4. LLM client + system prompt ──────────────────────────────────────────
client = anthropic.Anthropic()
CLAUDE_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024

# NOTE: no Thought/Action/Args/Final FORMAT contract here — that whole rigid
# protocol existed only to make text parsing possible. With native tool use the
# system prompt is just a plain instruction; the model figures out the coordinates
# (still from its own knowledge — see Q2) and calls the tool on its own.
SYSTEM = (
    "You are a helpful weather assistant. When asked about a location, work out "
    "its latitude and longitude and call the get_weather tool to fetch the forecast. "
    "Then reply in one friendly sentence stating the conditions, high, and low in °C."
)


# ── 5. Native tool-use loop ────────────────────────────────────────────────
def run(question: str) -> str:
    """Run the agent using native tool calling; completion is detected via
    response.stop_reason, never by parsing the model's text."""
    messages = [{"role": "user", "content": question}]

    print("\n--- native tool-use loop (stop_reason driven, no text parsing) ---\n")

    max_iterations = 5  # safety bound on tool-call rounds
    for _ in range(max_iterations):
        reply = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            tools=TOOLS_SCHEMA,  # ← declare the tools as schemas
            messages=messages,
        )

        # ── Trace ── show the structured blocks (this is the "Thought/Action"
        # made explicit: text blocks are the model's words, tool_use blocks are
        # the structured calls — both typed, neither parsed from prose).
        print(f"[stop_reason: {reply.stop_reason}]")
        for block in reply.content:
            if block.type == "text" and block.text.strip():
                print(f"  Text: {block.text.strip()}")
            elif block.type == "tool_use":
                print(f"  Tool call → {block.name}({json.dumps(block.input)})")
        print()

        # ── Safety stop ──
        if reply.stop_reason == "refusal":
            return "Sorry — the request was declined."

        # The assistant turn (text + tool_use blocks) must be echoed back into the
        # history verbatim before we answer any tool calls.
        messages.append({"role": "assistant", "content": reply.content})

        # ── Act ── The ONLY thing that drives another round is stop_reason ==
        # "tool_use" — a real API signal, not a substring match.
        if reply.stop_reason == "tool_use":
            tool_results = []
            for block in reply.content:
                if block.type != "tool_use":
                    continue
                func = TOOLS.get(block.name)
                if func is None:
                    # Unknown tool → report it back as an error tool_result so the
                    # model can recover, instead of crashing.
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Unknown tool: {block.name}",
                            "is_error": True,
                        }
                    )
                    continue
                try:
                    # block.input is ALREADY a parsed dict (schema-validated by
                    # strict mode) — splat it straight into the function.
                    observation = func(**block.input)
                    print(f"  Observation: {observation}\n")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,  # must match the tool_use block's id
                            "content": str(observation),
                        }
                    )
                except Exception as e:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error running {block.name}: {e}",
                            "is_error": True,
                        }
                    )

            # All tool_results for one assistant turn go back in a SINGLE user
            # message, then we loop so the model can use the observation.
            messages.append({"role": "user", "content": tool_results})
            continue

        # ── Done ── Any non-tool_use stop_reason (normally "end_turn") means the
        # model has finished; its text blocks ARE the final answer.
        return (
            "".join(b.text for b in reply.content if b.type == "text").strip()
            or "(no text answer)"
        )

    return "Sorry, I couldn't complete the task (hit the tool-round limit)."


# ── 6. Interactive loop ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Weather-forecast agent — Claude native tool use (type 'exit' to quit)\n")
    while True:
        loc = input("Location (or 'exit'): ").strip()
        if loc.lower() == "exit":
            print("Goodbye!")
            break

        query = f"What is the predicted weather today for {loc}?"

        try:
            answer = run(query)
            print(f"\n✓ {answer}\n")
        except Exception as e:
            print(f"⚠️  Error: {e}\n")
