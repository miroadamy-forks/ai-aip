# Questions & Answers — `agent1.py`

A running notebook of questions about the weather agent and their answers.

---

## Q1. Why does the code parse the LLM text for `Action`/`Args`/`Final`? Doesn't the Ollama API give an explicit stop reason and structured tool-use info like Anthropic does?

**Short answer:** The text-parsing is a *deliberate teaching choice*, **not** a limitation of Ollama.
Ollama's chat API **does** provide both (a) an explicit stop/finish reason and (b) structured
tool-call objects — very similar in spirit to Anthropic. `agent1.py` simply chooses to ignore
those and implement its own "ReAct/TAO" text protocol so every step is visible and the loop works
with *any* model, even ones not fine-tuned for tool calling.

Confidence: **HIGH** for the API facts (verified against the official Ollama docs via Context7,
2026-06-27). **MEDIUM** for the LangChain `response_metadata` detail (not freshly re-verified).

---

### What `agent1.py` actually does

It treats the LLM as a pure **text in → text out** function and enforces a format via the system
prompt:

```
Thought: <reasoning>
Action: get_weather
Args: {"lat": 39.46, "lon": 0.38}
```

Then `run()` recovers the tool call with plain string splitting:

```python
action_line = response.split("Action:")[1].split("\n")[0].strip()
args_text   = response.split("Args:")[1].split("\n")[0].strip()
args        = json.loads(args_text)
```

and decides "am I done?" by string-matching `"Final:"`. So the agent's control flow is driven
**entirely by parsing the assistant's prose**. The structured signals the model layer can provide
are never consulted.

---

### Yes — Ollama provides an explicit stop reason

Every Ollama `/api/chat` response includes a `done_reason` field:

```json
{
  "model": "llama3.2",
  "message": { "role": "assistant", "content": "Hello! How can I help you today?" },
  "done": true,
  "done_reason": "stop"
}
```

Typical values on the native endpoint: `"stop"` (model stopped naturally), `"length"`
(hit token limit), `"load"`. Through LangChain, this is surfaced on the returned message —
`reply.response_metadata.get("done_reason")` — but `agent1.py` only reads `reply.content`.

> Nuance vs. Anthropic: on the **native** `/api/chat` endpoint there is **no** dedicated
> `done_reason: "tool_use"` value. You detect a tool call by checking whether
> `message.tool_calls` is present, not by reading the stop reason. If you instead use Ollama's
> **OpenAI-compatible** endpoint (`/v1/chat/completions`), you *do* get
> `finish_reason: "tool_calls"`, which matches the OpenAI/Anthropic style closely.

---

### Yes — Ollama provides structured tool-call info

If you pass a `tools` array, a tool-capable model returns the call as **structured data**, not
prose. Native `/api/chat`:

```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": { "lat": 39.46, "lon": 0.38 }
        }
      }
    ]
  },
  "done": true,
  "done_reason": "stop"
}
```

Note `arguments` is already a **JSON object** on the native endpoint (no fragile string parsing
needed). `llama3.2` — the model this script uses — *is* tool-calling capable, so `agent1.py`
could have used this directly.

---

### Side-by-side: Anthropic vs. Ollama

| Concept | Anthropic Messages API | Ollama native `/api/chat` | Ollama OpenAI-compat `/v1/chat/completions` |
| --- | --- | --- | --- |
| Stop-reason field | `stop_reason` | `done_reason` | `finish_reason` |
| Value that signals a tool call | `"tool_use"` | *(none — detect via `message.tool_calls`)* | `"tool_calls"` |
| Tool-call payload | `content` block `type:"tool_use"` → `{name, input, id}` | `message.tool_calls[].function` → `{name, arguments}` | `choices[].message.tool_calls[].function` → `{name, arguments}` |
| Args type | `input` is a JSON object (dict) | `arguments` is a JSON object (dict) | `arguments` is a JSON **string** (OpenAI style) |

So conceptually they're close. The biggest difference is the native Ollama endpoint folds tool
calls under a generic `done_reason: "stop"` and expects you to inspect `tool_calls`, whereas
Anthropic gives the explicit `stop_reason: "tool_use"`.

---

### How the same agent looks using native tool calling (LangChain)

LangChain's `ChatOllama` exposes this via `.bind_tools()`. It infers the JSON schema from the
Python function's type hints + docstring, and returns parsed calls on `AIMessage.tool_calls`:

```python
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0.0)
llm_with_tools = llm.bind_tools([get_weather])   # schema auto-derived from get_weather

messages = [HumanMessage("What's the weather today in Valencia, Spain?")]
ai = llm_with_tools.invoke(messages)
messages.append(ai)

# ai.tool_calls -> [{'name': 'get_weather',
#                    'args': {'lat': 39.46, 'lon': 0.38},
#                    'id': 'call_abc', 'type': 'tool_call'}]
for call in ai.tool_calls:
    result = TOOLS[call["name"]](**call["args"])
    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

final = llm_with_tools.invoke(messages)   # model now writes the natural-language answer
print(final.content)
```

No `split("Action:")`, no `json.loads` on hand-cut substrings, no `"Final:"` sentinel — the
boundaries are carried by the protocol, not by the prose.

---

### So why hand-roll the text protocol at all?

Legitimate reasons (this is course/teaching code in `ai-aip`):

1. **Pedagogy / transparency.** The TAO loop is the lesson. Printing `Thought:` / `Action:` /
   `Observation:` makes the agent's reasoning literally readable in the terminal; native
   tool-calling hides those mechanics inside the message object.
2. **Model-agnostic.** The text protocol works with *any* chat model, including small/older ones
   with no tool-calling fine-tune. `bind_tools` only works on tool-capable models.
3. **No schema plumbing.** You skip constructing/registering JSON tool schemas.
4. **Full control of the loop and tracing.**

What you give up by hand-rolling (i.e. what the structured API buys you):

- **Robustness.** String-splitting breaks the instant the model deviates from the format
  (extra prose, markdown fences, multi-line JSON). `agent1.py` mitigates this only with strict
  prompt rules ("every response must start with `Thought:`").
- **A real stop signal.** The loop trusts the model to "STOP and wait for Observation". Nothing
  enforces it — the model could hallucinate an `Observation:` in the same turn. A structured
  `tool_calls` response ends the turn deterministically.
- **Parallel / multiple tool calls**, tool-call IDs, and clean `tool` role messages — all native
  to the structured API, all absent here.

**Bottom line:** parsing prose is the *intended pedagogical design* of `agent1.py`, not a
workaround for a missing Ollama feature. The structured stop-reason + tool-call API exists and,
for production code, is the more reliable path.

---

#### Sources
- Ollama API — `POST /api/chat` (`tools`, `message.tool_calls`, `done`, `done_reason`):
  https://docs.ollama.com/api/chat
- Ollama tool calling guide: https://docs.ollama.com/capabilities/tool-calling
- Ollama Anthropic-compatible endpoint (`stop_reason: "tool_use"`):
  https://docs.ollama.com/api/anthropic-compatibility
- Verified via Context7 on 2026-06-27.

---

## Q2. I type a city + country. Who converts that to latitude/longitude before it's sent to the tool?

**Short answer:** The **LLM does it, from memory.** There is **no geocoding step** — no API, no lookup table, no library. The model recalls the coordinates from its training data and writes them straight into the `Args` line. `get_weather` only ever receives numbers.

Confidence: **HIGH** (this is visible directly in the code and the run log).

---

### The trace, step by step

1. You type `Valencia, Spain`.
2. The REPL wraps it: `query = "What is the predicted weather today for Valencia, Spain?"` — still just text, no coordinates.
3. That text goes to the LLM. The model **invents the coordinates itself** in its reasoning step. From the run log:

   ```
   Thought: I need to get weather for Valencia at coordinates 39.4673, 0.3832
   Action: get_weather
   Args: {"lat": 39.4673, "lon": 0.3832}
   ```

   The "39.4673, 0.3832" came out of the model's weights — nothing in the program computed it.
4. Only **now** does Python get involved: `run()` parses that `Args` JSON and calls `get_weather(lat=39.4673, lon=0.3832)`.

So the geocoding "tool" is the language model's own world knowledge. The tool signature makes this explicit — it accepts coordinates, never a name:

```python
def get_weather(lat: float, lon: float) -> dict:   # numbers in, never "Valencia"
```

This is the same division of labour as Q1: the LLM **decides** (here, *which coordinates*), Python **executes** (the HTTP call). The difference is that for the weather data we deliberately don't trust the model (we fetch it live), but for the **coordinates we do** — silently.

---

### Why this matters — it's a real reliability hole

Using the LLM as a geocoder means the coordinates are a **plausible guess**, not a lookup. They can be subtly or badly wrong, and the program has no way to notice:

- **Concrete example from the run log:** Valencia, Spain is actually at about **39.47° N, 0.38° _W_** → longitude **−0.38**. The model emitted `"lon": 0.3832` — **positive** (east), which is a point in the Mediterranean Sea offshore, not the city. Open-Meteo still returned a plausible-looking forecast (26.9 / 25.6, Partly cloudy) because the wrong point is close enough that the weather is similar — so the bug is **invisible**: no error, just a quietly-wrong location.
- Ambiguous names ("Springfield", "Tripoli", duplicate city names across countries) are exactly where a from-memory guess is most likely to land in the wrong country.
- Smaller towns and non-English place names degrade further.

The model is acting as a confident geocoder with no error bars — and CRITICAL RULE #2 ("NEVER hallucinate tool results") doesn't help, because the coordinates are part of the **tool _input_**, not the tool _output_.

---

### The fix: make geocoding a real tool (or a real step)

**Option A — add a geocoding tool** and let the agent call it first (a genuine two-step TAO: geocode → get_weather). Open-Meteo offers a free geocoding endpoint:

```python
def geocode(name: str) -> dict:
    """Resolve a place name to coordinates via Open-Meteo's geocoding API."""
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": name, "count": 1},
        timeout=15,
    )
    r.raise_for_status()
    hit = r.json()["results"][0]
    return {"lat": hit["latitude"], "lon": hit["longitude"], "name": hit["name"]}

TOOLS = {"geocode": geocode, "get_weather": get_weather}
```

Then teach the model in the system prompt to call `geocode` before `get_weather`. Now the coordinates come from a real database, and the agent loop shows two Action/Observation rounds instead of one.

**Option B — resolve in Python before the agent runs.** Call `geocode(loc)` in the interactive loop and hand the coordinates to the agent, so the LLM never guesses them at all. Simpler, but the agent no longer "decides" the location — fine if you only care about reliability.

**Bottom line:** today, *the model* is your geocoder, invisibly, on every query. For anything beyond a demo, replace that guess with a real geocoding lookup (Option A keeps it agentic; Option B keeps it simple).

#### Sources
- Open-Meteo Geocoding API: https://open-meteo.com/en/docs/geocoding-api
- Behaviour read directly from `agent1.py` and its run log (no external verification needed).
