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
