#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "groq",
# ]
# ///
#
# Run with:  uv run agent1-groq.py   (uv reads the metadata above and installs
# deps into an isolated env). Also requires:  export GROQ_API_KEY=...

# weather-agent with TAO — Groq variant of agent1.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT CHANGED vs agent1.py
# ─────────────────────────────────────────────────────────────────────────────
# This is the SAME hand-rolled TAO (Thought → Action → Observation) agent as
# agent1.py. The only thing swapped is the LLM backend:
#
#     agent1.py       → local Ollama server, model "llama3.2"  (langchain_ollama)
#     agent1-groq.py  → Groq cloud inference, a Llama model    (groq SDK)
#
# Groq is a hosted inference provider (very fast, runs open models like Llama on
# custom LPU hardware). Its API is OpenAI-compatible, so the message format
# (system/user/assistant dicts) is IDENTICAL to what agent1.py already builds —
# which is why the run() loop below is unchanged from agent1.py except for the
# single line that calls the model. Everything else (the weather tool, the
# system-prompt contract, the text parsing) is copied verbatim.
#
# Setup:
#     uv add groq            # or: pip install groq
#     export GROQ_API_KEY=...  # get a key at https://console.groq.com
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import requests
import textwrap
import time
from groq import Groq  # OpenAI-compatible client for Groq's cloud API

# ── 1. Open-Meteo weather-code lookup ──────────────────────────────────────
# (identical to agent1.py — maps Open-Meteo's numeric WMO codes to text)
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
# (identical to agent1.py — the tool itself does not care which LLM drives it)
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


# ── 3. Tool registry ────────────────────────────────────────────────────────
TOOLS = {
    "get_weather": get_weather,
}

# ── 4. LLM client ───────────────────────────────────────────────────────────
# Groq() reads the GROQ_API_KEY environment variable automatically.
# Model choice: agent1.py uses local "llama3.2"; the nearest hosted production
# Llama on Groq is "llama-3.3-70b-versatile" — larger, so it follows the strict
# Thought/Action/Args format more reliably. For a faster/cheaper run, swap in
# "llama-3.1-8b-instant".
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── 5. System prompt ────────────────────────────────────────────────────────
# (identical to agent1.py — the format contract the text parser depends on)
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
    # Groq is OpenAI-compatible, so the system prompt is just the first message
    # in the list (role "system") — exactly like agent1.py. No restructuring
    # needed (contrast with agent1-claude.py, where the system prompt is a
    # separate parameter).
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    print("\n--- Thought → Action → Observation loop ---\n")

    max_iterations = 5
    for i in range(max_iterations):
        # ── Reason ── The ONLY backend-specific line: call Groq's chat endpoint.
        # temperature=0.0 keeps the output deterministic so the strict format
        # holds. The reply shape (choices[0].message.content) is the OpenAI shape.
        reply = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.0,
        )
        # message.content is typed Optional[str]; `or ""` guards None so the
        # type checker is satisfied and .strip() can't crash on null content.
        response = (reply.choices[0].message.content or "").strip()
        print(response + "\n")

        # ── Done? ──
        if "Final:" in response:
            final = response.split("Final:")[1].strip()
            return final

        # ── Act ── (identical parsing logic to agent1.py)
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
            print("Expected format:\nThought: ...\nAction: <tool_name>\nArgs: <json>\n")
            print(f"Got:\n{response[:200]}...\n")
            break

    return "Sorry, I couldn't complete the task."


# ── 7. Interactive loop ────────────────────────────────────────────────────
# (identical to agent1.py)
if __name__ == "__main__":
    print("Weather-forecast agent — Groq backend (type 'exit' to quit)\n")
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
