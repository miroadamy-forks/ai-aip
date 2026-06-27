# weather-agent with TAO – AI-driven tool selection + interactive loop + full tracing
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THIS PROGRAM IS
# ─────────────────────────────────────────────────────────────────────────────
# A minimal, hand-rolled "ReAct"-style agent. ReAct (Reason + Act) is more
# commonly described here with the TAO acronym:
#
#     Thought      – the LLM reasons about what to do next
#     Action       – the LLM names a tool and its arguments
#     Observation  – our code runs that tool and feeds the result back
#
# The loop repeats until the LLM emits a "Final:" answer instead of an action.
#
# The key idea: the Large Language Model never fetches weather data itself and
# is explicitly told not to invent it. Instead it DECIDES which tool to call and
# with what arguments; our Python code actually executes the tool and returns the
# real result. This separation is what makes the agent trustworthy — the facts
# come from a real API (Open-Meteo), the LLM only orchestrates.
#
# The whole "agent framework" here is intentionally written by hand (no
# LangChain agents/executors) so every step is visible and easy to trace.
# ─────────────────────────────────────────────────────────────────────────────

import json  # parse the LLM-produced Args (a JSON object) into a Python dict
import requests  # HTTP client used to call the Open-Meteo weather API
import textwrap  # dedent() the multi-line system prompt so indentation is clean
import time  # sleep between retry attempts on network failure
from langchain_ollama import ChatOllama  # thin wrapper to talk to a local Ollama LLM

# ── 1. Open-Meteo weather-code lookup ──────────────────────────────────────
# Open-Meteo reports conditions as a numeric "weathercode" (WMO code) rather
# than human text. This table maps each code to a readable description. We look
# the code up after fetching the forecast (see get_weather below). Any code not
# present here falls back to "Unknown" via dict.get().
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
# A "tool" is just a normal Python function the agent is allowed to call. This
# is the single tool in this example: given a latitude/longitude it returns
# today's high, low, and conditions. The LLM is told this signature in the
# system prompt and is responsible for supplying sensible coordinates for the
# location the user asked about.
def get_weather(lat: float, lon: float) -> dict:
    """
    Return today's forecast:
        { "high": °C, "low": °C, "conditions": str }
    """
    # Build the Open-Meteo request URL.
    #   daily=...            → ask for the daily weathercode, max and min temps
    #   forecast_days=1      → only today
    #   timezone=auto        → interpret "today" in the location's local timezone
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=weathercode,temperature_2m_max,temperature_2m_min"
        "&forecast_days=1&timezone=auto"
    )

    # Retry up to 3 times
    # Networks are flaky, so wrap the request in a small retry loop. Only
    # transient connection problems (Timeout / ConnectionError) are retried;
    # an HTTP error status (4xx/5xx) raised by raise_for_status() is NOT caught
    # here and will propagate up to the caller.
    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=15)  # 15s network timeout per attempt
            r.raise_for_status()  # turn HTTP 4xx/5xx into an exception
            daily = r.json()["daily"]  # pull out the "daily" forecast block
            # Each daily field is a list (one entry per forecast day); we asked
            # for a single day so index [0] is today's value.
            return {
                "high": daily["temperature_2m_max"][0],
                "low": daily["temperature_2m_min"][0],
                # Translate the numeric weathercode into readable text.
                "conditions": WEATHER_CODES.get(daily["weathercode"][0], "Unknown"),
            }
        except (requests.Timeout, requests.ConnectionError) as e:
            # On the LAST allowed attempt, give up and re-raise so the caller
            # can report the failure.
            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt
            # Otherwise log a retry notice and wait before trying again.
            # Note the user-facing counter shows "1/2" and "2/2" because it
            # prints attempt+1 out of (max_retries-1) remaining retries.
            print(f"  ⚠️  Retry {attempt + 1}/{max_retries - 1} after timeout...")
            time.sleep(2)  # Wait 2 seconds before retrying


# ── 3. Tool registry ────────────────────────────────────────────────────────
# Maps the tool NAME (the string the LLM will emit after "Action:") to the
# actual Python function. The run loop looks the function up here by name. To
# give the agent more abilities you would add more entries (and describe them in
# the system prompt below).
TOOLS = {
    "get_weather": get_weather,
}

# ── 4. LLM client ───────────────────────────────────────────────────────────
# Connect to a locally running Ollama server using the "llama3.2" model.
# temperature=0.0 makes output as deterministic as possible — important for an
# agent that must follow a rigid output format and not get "creative".
llm = ChatOllama(model="llama3.2", temperature=0.0)

# ── 5. System prompt ────────────────────────────────────────────────────────
# This is the contract we impose on the model. It teaches the LLM:
#   • which tool exists and its exact signature,
#   • the EXACT text format to use for a tool call (Thought/Action/Args),
#   • the EXACT text format for a final answer (Thought/Final),
#   • and hard rules (don't hallucinate results, stop after an Action and wait
#     for the Observation, etc.).
# Our run() loop parses the model's output by string-matching these keywords, so
# the format discipline enforced here is what makes the simple parser work.
# textwrap.dedent removes the leading indentation of the triple-quoted block;
# .strip() trims the surrounding blank lines.
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
    # The running conversation transcript. We seed it with the system prompt
    # (the rules/contract) and the user's question. As the loop proceeds we
    # append the assistant's actions and the resulting observations so the model
    # always sees the full history on the next turn.
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    print("\n--- Thought → Action → Observation loop ---\n")

    # Hard cap on loop iterations so a confused model can't spin forever.
    max_iterations = 5  # Safety limit
    for i in range(max_iterations):
        # ── Reason ── Ask the LLM for its next step given the transcript so far.
        reply = llm.invoke(messages)
        response = reply.content.strip()
        print(response + "\n")  # trace: show the model's raw output

        # ── Done? ── If the model emitted a Final answer, extract the text after
        # "Final:" and return it — the loop is complete.
        if "Final:" in response:
            # Extract and return the final answer
            final = response.split("Final:")[1].strip()
            return final

        # ── Act ── Otherwise expect a tool call in Action/Args form and run it.
        if "Action:" in response and "Args:" in response:
            try:
                # Extract action and args
                # Grab the tool name: everything after "Action:" up to end of
                # that line. Grab the args: the first line after "Args:". This
                # naive parsing relies on the strict format from the system
                # prompt (one tool name per line, JSON on a single line).
                action_line = response.split("Action:")[1].split("\n")[0].strip()
                args_text = response.split("Args:")[1].split("\n")[0].strip()

                # Get the tool function
                # Look the requested tool up in the registry by name.
                tool_name = action_line
                tool_func = TOOLS.get(tool_name)

                # Guard: the model asked for a tool we don't have. Report and
                # stop rather than crash.
                if tool_func is None:
                    print(f"⚠️  Unknown tool: '{tool_name}'\n")
                    print(f"Available tools: {list(TOOLS.keys())}\n")
                    break

                # Parse arguments and call the tool
                # The Args text is JSON like {"lat": 51.5, "lon": -0.1}; turn it
                # into a dict and splat it into the function as keyword args.
                args = json.loads(args_text)
                observation = tool_func(**args)
                print(f"Observation: {observation}\n")  # trace: real tool result

                # Add to conversation history
                # Feed both the model's action and the real observation back into
                # the transcript so the next iteration can produce the Final
                # answer (or another action).
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {"role": "user", "content": f"Observation: {observation}"}
                )
            except json.JSONDecodeError as e:
                # The model produced malformed JSON in Args — report and stop.
                print(f"⚠️  Failed to parse Args as JSON: {e}\n")
                print(f"Args text was: {args_text}\n")
                break
            except Exception as e:
                # Any other failure while running the tool (e.g. network error
                # re-raised by get_weather after exhausting retries, bad args).
                print(f"⚠️  Error executing tool: {e}\n")
                break
        else:
            # The model's response was neither a Final answer nor a valid
            # Action/Args call — it broke the format contract. Show what we got
            # (truncated) and stop.
            print("⚠️  AI response missing Action/Args format\n")
            print(
                f"Expected format:\nThought: ...\nAction: <tool_name>\nArgs: <json>\n"
            )
            print(f"Got:\n{response[:200]}...\n")
            break

    # Reached only if we hit max_iterations or broke out of the loop above
    # without returning a Final answer.
    return "Sorry, I couldn't complete the task."


# ── 7. Interactive loop ────────────────────────────────────────────────────
# Simple REPL: repeatedly ask the user for a location, wrap it into a question,
# run the TAO loop, and print the answer. Type "exit" to quit.
if __name__ == "__main__":
    print("Weather-forecast agent (type 'exit' to quit)\n")
    while True:
        loc = input("Location (or 'exit'): ").strip()
        if loc.lower() == "exit":
            print("Goodbye!")
            break

        # Build the question for the agent
        # Phrase the user's raw location as a full natural-language question for
        # the LLM. The model must infer coordinates for this location itself.
        query = f"What is the predicted weather today for {loc}?"

        try:
            answer = run(query)
            print(f"\n✓ {answer}\n")
        except Exception as e:
            # Catch-all so one failed lookup doesn't kill the whole session.
            print(f"⚠️  Error: {e}\n")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE RUN LOG
# ─────────────────────────────────────────────────────────────────────────────
# The transcript below is a real sample session. Note the second query
# (Bratislava) shows the network-retry path in get_weather: the first attempt
# timed out, "Retry 1/2" was printed, the second attempt succeeded, and the loop
# continued normally.
#
# (py_env) @miroadamy ➜ /workspaces/ai-aip/agents (main) $ python agent1.py
# Weather-forecast agent (type 'exit' to quit)
#
# Location (or 'exit'): Valencia, Spain
#
# --- Thought → Action → Observation loop ---
#
# Thought: I need to get weather for Valencia at coordinates 39.4673, 0.3832
# Action: get_weather
# Args: {"lat": 39.4673, "lon": 0.3832}
#
# Observation: {'high': 26.9, 'low': 25.6, 'conditions': 'Partly cloudy'}
#
# Thought: I now have the weather data for Valencia
# Final: Today in Valencia will be Partly cloudy with a high of 26.9°C and a low of 25.6°C.
#
#
# ✓ Today in Valencia will be Partly cloudy with a high of 26.9°C and a low of 25.6°C.
#
# Location (or 'exit'): Bratislava, Slovak Republic
#
# --- Thought → Action → Observation loop ---
#
# Thought: I need to get weather for Bratislava at coordinates 48.1423, 17.1081
# Action: get_weather
# Args: {"lat": 48.1423, "lon": 17.1081}
#
#   ⚠️  Retry 1/2 after timeout...
# Observation: {'high': 37.5, 'low': 25.7, 'conditions': 'Partly cloudy'}
#
# Thought: I now have the weather data for Bratislava
# Final: Today in Bratislava will be Partly cloudy with a high of 37.5°C and a low of 25.7°C.
#
#
# ✓ Today in Bratislava will be Partly cloudy with a high of 37.5°C and a low of 25.7°C.
# ─────────────────────────────────────────────────────────────────────────────
