# weather-agent with TAO — Anthropic (Claude Haiku) variant of agent1.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT CHANGED vs agent1.py
# ─────────────────────────────────────────────────────────────────────────────
# Same hand-rolled TAO (Thought → Action → Observation) agent as agent1.py, with
# the LLM backend swapped to Anthropic's Claude (model: Haiku 4.5):
#
#     agent1.py        → local Ollama, "llama3.2"        (langchain_ollama)
#     agent1-claude.py → Anthropic API, "claude-haiku-4-5" (anthropic SDK)
#
# Unlike Groq/OpenAI-compatible backends, the Anthropic Messages API shapes
# requests differently, so two things change beyond the import:
#
#   1. The system prompt is a SEPARATE `system=` parameter — it is NOT the first
#      entry of the messages list. So run() seeds `messages` with only the user
#      question and passes SYSTEM alongside it.
#   2. The reply text lives in a list of typed content blocks
#      (reply.content[i].text), not in `choices[0].message.content`.
#
# The TAO loop, the weather tool, the system-prompt contract, and the text
# parsing are otherwise unchanged — we deliberately keep the same hand-rolled
# text protocol so all three agents (Ollama / Groq / Claude) are comparable.
#
# NOTE (see agents/agent1.py.md, Q1): Claude also supports NATIVE tool calling
# (client.messages.create(..., tools=[...]) → stop_reason "tool_use" + structured
# tool_use blocks). We intentionally do NOT use it here, to keep this file a
# 1:1 port of agent1.py's text-protocol design rather than a rewrite.
#
# Setup:
#     uv add anthropic
#     export ANTHROPIC_API_KEY=...   # get a key at https://console.anthropic.com
# ─────────────────────────────────────────────────────────────────────────────

import json
import requests
import textwrap
import time
import anthropic  # official Anthropic SDK

# ── 1. Open-Meteo weather-code lookup ──────────────────────────────────────
# (identical to agent1.py)
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


# ── 2. Tools ───────────────────────────────────────────────────────────────
# (identical to agent1.py)
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
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == max_retries - 1:
                raise
            print(f"  ⚠️  Retry {attempt + 1}/{max_retries - 1} after timeout...")
            time.sleep(2)


# ── 3. Tool registry ────────────────────────────────────────────────────────
TOOLS = {
    "get_weather": get_weather,
}

# ── 4. LLM client ───────────────────────────────────────────────────────────
# Anthropic() reads the ANTHROPIC_API_KEY environment variable automatically.
# Haiku 4.5 is the small/fast/cheap Claude tier — the right pick for a simple,
# rigid-format agent loop like this one.
client = anthropic.Anthropic()
CLAUDE_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024  # each turn is short (a Thought + Action/Args, or a Final)

# ── 5. System prompt ────────────────────────────────────────────────────────
# (identical text to agent1.py — but on Claude this is passed as the separate
#  `system=` parameter, not as a message; see run() below)
SYSTEM = textwrap.dedent("""
You are a weather agent with one tool:

get_weather(lat:float, lon:float)
    → {"high": float, "low": float, "conditions": str}
    Returns today's weather forecast with temperatures in Celsius

You MUST follow this exact format. Do NOT add extra text or explanations.

To use the tool, output EXACTLY this format:
Thought: <your reasoning>
Action: get_weather
Args: {"lat": <latitude>, "lon": <longitude>}

Example:
Thought: I need to get weather for London at coordinates 51.5074, -0.1278
Action: get_weather
Args: {"lat": 51.5074, "lon": -0.1278}

When you have the information needed to answer, output:
Thought: <your reasoning>
Final: <complete natural language answer - NO Thought/Action/Args format here>

Example of Final:
Thought: I now have the weather data for London
Final: Today in London will be Slight rain showers with a high of 12.7°C and a low of 8.6°C.

CRITICAL RULES:
1. Follow the format EXACTLY - every response must start with "Thought:"
2. NEVER make up or hallucinate tool results
3. After outputting Action/Args, STOP and wait for Observation
4. Only proceed after you receive the actual Observation
5. After "Final:" output ONLY plain text - do NOT use Thought/Action/Args format
""").strip()


# ── 6. TAO run helper ───────────────────────────────────────────────────────
def run(question: str) -> str:
    """Execute the TAO loop, letting the AI decide which tools to call."""
    # Anthropic difference #1: the messages list holds ONLY user/assistant turns.
    # The system prompt is NOT a message here — it is passed via system=SYSTEM on
    # each create() call below.
    messages = [
        {"role": "user", "content": question},
    ]

    print("\n--- Thought → Action → Observation loop ---\n")

    max_iterations = 5
    for i in range(max_iterations):
        # ── Reason ── Call the Anthropic Messages API. system= carries the
        # contract; max_tokens is required by this API.
        reply = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=messages,
        )
        # Anthropic difference #2: the answer is a list of content blocks, not a
        # single string. Concatenate the text blocks to recover the reply text.
        response = "".join(
            block.text for block in reply.content if block.type == "text"
        ).strip()
        print(response + "\n")

        # ── Done? ── (identical parsing logic to agent1.py from here on)
        if "Final:" in response:
            final = response.split("Final:")[1].strip()
            return final

        # ── Act ──
        if "Action:" in response and "Args:" in response:
            try:
                action_line = response.split("Action:")[1].split("\n")[0].strip()
                args_text = response.split("Args:")[1].split("\n")[0].strip()

                tool_name = action_line
                tool_func = TOOLS.get(tool_name)

                if tool_func is None:
                    print(f"⚠️  Unknown tool: '{tool_name}'\n")
                    print(f"Available tools: {list(TOOLS.keys())}\n")
                    break

                args = json.loads(args_text)
                observation = tool_func(**args)
                print(f"Observation: {observation}\n")

                # Feed the action + observation back. Claude accepts plain strings
                # for user/assistant content; consecutive user turns are allowed
                # (the API merges them), so "Observation: ..." as a user turn is fine.
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {"role": "user", "content": f"Observation: {observation}"}
                )
            except json.JSONDecodeError as e:
                print(f"⚠️  Failed to parse Args as JSON: {e}\n")
                print(f"Args text was: {args_text}\n")
                break
            except Exception as e:
                print(f"⚠️  Error executing tool: {e}\n")
                break
        else:
            print("⚠️  AI response missing Action/Args format\n")
            print(
                f"Expected format:\nThought: ...\nAction: <tool_name>\nArgs: <json>\n"
            )
            print(f"Got:\n{response[:200]}...\n")
            break

    return "Sorry, I couldn't complete the task."


# ── 7. Interactive loop ────────────────────────────────────────────────────
# (identical to agent1.py)
if __name__ == "__main__":
    print("Weather-forecast agent — Claude (Haiku) backend (type 'exit' to quit)\n")
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
