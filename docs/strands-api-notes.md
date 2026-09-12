# Strands API notes

What was verified, by installing `strands-agents` and running it — not from memory
and not from a tutorial. Every claim below was checked against the installed
package or produced by a script that ran.

```
strands-agents 1.54.0   (import name: strands)
python 3.11.15
```

Anything a later version changes should be re-checked here first; the agent code
in `src/first_reader/agent/` leans on several of these behaviours.

---

## Corrections to what we assumed going in

### 1. Tool results are not flattened with `str(result)`

The assumption was that a tool's return value reaches the model as
`str(result)`, and that `agent.state` is the way around it. The conclusion was
right; the mechanism is not, and the difference changes the fix.

`DecoratedFunctionTool._wrap_tool_result` (`strands/tools/decorator.py:678`)
tries, in order:

| return value | what the model sees |
|---|---|
| `dict` with `status` and `content` keys | passed through as a `ToolResult` |
| `str` | the string itself |
| pydantic `BaseModel` | `model_dump_json()` |
| anything else | `json.dumps(result, ensure_ascii=False)` |
| the above raising `TypeError`/`ValueError` | `str(result)` |

So a dict or a list arrives as real JSON. Only values `json.dumps` refuses fall
back to `str()` — and a **dataclass is exactly such a value**. Verified:

```
tool returning Thing(a=1, b='x')  ->  {'status': 'success', 'content': [{'text': "Thing(a=1, b='x')"}]}
```

Since `models.py` is all dataclasses, every tool in `tools.py` returns a plain
dict and keeps the rich object elsewhere. The workaround stands; "return a dict"
is the whole of it.

### 2. `agent.state` cannot hold the objects either

`agent.state` is a `JSONSerializableDict` (`strands/types/json_dict.py`). `set()`
validates the value with `json.dumps` and deep-copies it, so a `DraftHistory`
raises there too. The carrier has to be serialised on the way in and rebuilt on
the way out; that is what `agent/_serde.py` exists for.

State is also per-process: it is not persistence, only a carrier within a run.

### 3. A name is not an interface (our bug, not the SDK's)

Not an SDK finding, but it cost the most time here. `tools.py` first resolved the
sibling module by attribute name:

```python
fn = getattr(module, "fetch_draft_history", None)   # wrong
```

`first_reader.wiki` does export that name — with a different signature:

```python
async def fetch_draft_history(client: WikiClient, title: str, *,
                              with_revisions: bool = True,
                              revision_limit: int | None = None
                              ) -> tuple[DraftHistory | None, ParseStats]
```

The mismatch surfaced four tool calls later as `TypeError: missing 1 required
positional argument`, inside a tool result, with nothing pointing at the cause.
`tools.py` now binds that signature explicitly and checks arity before adopting
any function it discovers by name.

---

## Confirmed as described

### Intervention actions

`strands/interventions/actions.py` defines exactly `Proceed`, `Deny`, `Guide`,
`Confirm`, `Transform`, with `InterventionAction` as their union. The
compatibility matrix in that file's docstring:

| action | before_invocation | before_tool_call | before_model_call | after_tool_call | after_model_call |
|---|---|---|---|---|---|
| `Proceed` | — | — | — | — | — |
| `Deny` | cancel | cancel | cancel | — | — |
| `Guide` | cancel+ | cancel+ | inject | — | inject + retry |
| `Confirm` | — | confirm | — | — | — |
| `Transform` | apply | apply | apply | apply | apply |

`—` warns at runtime and does nothing.

**`Guide`** attaches feedback rather than blocking outright. On
`before_tool_call` it cancels *with* the feedback attached, so the model sees why.
On `before_model_call` it injects the feedback as a user message. On
`after_model_call` it discards the response and retries — and the framework
imposes **no retry cap**, so a handler using it there has to converge on its own.
Guide-injected messages bypass session management and are not tracked by a
session manager.

**`Transform`** takes an `apply(event) -> None` callable that mutates the event
in place before execution continues. Later handlers see the mutation. It is the
only action that is a no-op on nothing — it applies at all five events.

First Reader uses `Deny`, `Proceed` and **`Transform`**. `Guide` on
`after_model_call` with no retry cap is a loop against a metered provider;
`Confirm` holds a process open (below).

`Transform` is the one worth dwelling on, because it is the only action that
lets a person change what the agent *does* rather than approve or reject what it
already did. `core.ReviewerCorrections` uses it like this:

```python
def before_tool_call(self, event):
    if event.tool_use["name"] != "interpret_reasons":
        return Proceed()
    corrections = self._policy.corrections_for(event.tool_use["input"]["title"])
    if not corrections:
        return Proceed()

    def apply(evt):
        evt.tool_use.setdefault("input", {})["known_readings"] = json.dumps(...)

    return Transform(apply=apply, reason="a reviewer already settled these")
```

Verified: `apply(event)` mutates `event.tool_use["input"]` in place, the tool
receives the injected argument, and handlers registered after this one see the
mutated event. So a reviewer correcting an estimate does not leave a note for
someone to read later — their answer becomes an argument of the agent's next
tool call, and the model is not asked the question again.

Two details that matter in practice. `Transform` is the only action that is
live on all five lifecycle events. And the mutation has to go through
`event.tool_use`; `BeforeToolCallEvent._can_write` permits `cancel_tool`,
`selected_tool` and `tool_use`, so replacing the whole `input` dict works and
assigning to some other attribute does not.

### Handlers

`InterventionHandler` is an ABC with a required `name` property, an optional
`on_error` (`"throw"` default / `"proceed"` fail-open / `"deny"` fail-closed), and
five lifecycle methods. Two things that are easy to get wrong:

- Overrides must be at **class level**. The framework detects which methods are
  overridden and registers callbacks only for those; assigning
  `handler.before_tool_call = fn` on an instance is not detected.
- Handlers run in registration order. `Deny` short-circuits the rest;
  `Guide` feedback accumulates across handlers.

`Agent(interventions=[...])` takes `list[InterventionHandler]`.

### Interrupt and resume

Verified with a running agent, not read:

```python
result = agent("go")
result.stop_reason        # "interrupt"
result.interrupts         # [Interrupt(id='v1:before_tool_call:<toolUseId>:<uuid5>', name='gate', reason='post it?', ...)]

responses = [{"interruptResponse": {"interruptId": i.id, "response": "y"}}
             for i in result.interrupts]
result = agent(responses)
result.stop_reason        # "end_turn" -- the tool then ran
```

Notes worth having:

- `Confirm(prompt=...)` with no `response` breaks out of the loop. The `prompt`
  arrives on the interrupt as `reason`; the interrupt's `name` is the handler's.
- With `Confirm(response=...)` supplied preemptively, the agent never pauses.
- Resuming requires a `list` of `interruptResponse` blocks. Any other prompt type
  raises `TypeError`, and an unknown interrupt id raises `KeyError`.
- A tool or hook can raise its own interrupt via `event.interrupt(name, reason)`;
  the second call to it returns the human's response instead of raising.

**We do not use this.** Escalation writes a card to a durable queue and the run
ends. Issue #3651 — a turn cut while a tool is in flight leaves a `toolUse` with
no matching `toolResult` and the session cannot be reinvoked — is only reachable
if something is holding a run open across human latency. Nothing here does.

### `Deny` in practice

The tool never executes, and the model receives:

```json
{"toolUseId": "...", "status": "error",
 "content": [{"text": "DENIED: <reason>"}]}
```

`core.ReversibilityGate` depends on this: refusing `post_reviewer_decision`
leaves the model an explanation rather than a silent failure.

### Tool annotations

`ToolSpec` has an optional `annotations: dict[str, object]`. Three properties
were checked, all of which the reversibility scheme needs:

1. It survives on the spec and can be set after decoration —
   `my_tool.tool_spec["annotations"] = {...}` works.
2. It is **not** forwarded to model provider APIs, so grades cost no tokens.
3. It is reachable from a handler via `event.selected_tool.tool_spec`.

The SDK's own docstring is worth quoting: annotations are *"untrusted hints from
the tool provider, not guarantees; consumers such as permission layers must not
treat them as a security boundary."* That is right for third-party tools. Here
every tool and the gate are in the same repository under the same review, so the
annotation is a declaration by the author to the author. The property that
matters is that an **undeclared** tool is refused, which `ReversibilityGate`
enforces.

### Structured output, without a second agent

`Model.structured_output(schema, prompt, system_prompt=...)` is on the `Model`
ABC, so it can be called on the agent's own model without constructing anything.
It is an async generator; the last event is `{"output": <pydantic instance>}`:

```python
async for event in model.structured_output(FamilyReading, prompt, system_prompt=...):
    last = event
reading = last["output"]          # a FamilyReading, or absent on failure
```

This is how the interpretation layer reads free-text decline reasons. Doing it
this way rather than as a nested `Agent` avoids the agent-as-tool shape that
issue #3076 reports as broken, and there is nothing about classifying one
paragraph that needs a loop.

Confirmed working against OpenRouter through `OpenAIModel(client_args={...
"base_url": "https://openrouter.ai/api/v1"})` with `openai/gpt-4o-mini`.

**Two costs that are easy to miss.** A model provider owns an HTTP client, so
constructing one per call leaks a client per call; their response generators are
then finalised outside any loop and the run ends with
`RuntimeError: generator didn't stop after athrow()` on stderr despite
succeeding. Reuse `agent.model`. And a *sync* `@tool` that drives async work on
its own loop in its own thread produces the same error for the same reason —
`@tool` supports `async def`, so a tool that calls a model should be one.

### `@tool`

```python
def tool(func=None, description=None, inputSchema=None, name=None, context=False)
```

- The input schema is generated from type hints and a Google-style docstring;
  `Args:` entries become parameter descriptions.
- `context=True` injects a `ToolContext` as `tool_context` (or a name of your
  choosing if `context` is a string). It carries `.tool_use`, `.agent`,
  `.invocation_state`, `.cancel_signal`. Verified: the context parameter is
  excluded from the generated `inputSchema`, so the model never sees it.
- Sync, `async def` and async-generator tools are all supported. Prefer
  `async def` for any tool that performs I/O: it runs on the agent's loop, and
  the alternative is a second loop on a second thread with the teardown problem
  described above.

### Per-invocation limits

```python
agent(prompt, limits={"turns": 10, "total_tokens": 120_000, "output_tokens": 4_000})
```

- Caps bound one invocation; counters do not accumulate across calls.
- Reaching one stops the loop **gracefully** at a turn boundary with
  `stop_reason` of `"limit_turns"` / `"limit_total_tokens"` /
  `"limit_output_tokens"`. No exception. Verified: `limits={"turns": 2}` returns
  `stop_reason == "limit_turns"` with `agent.messages` still reinvokable.
- Token caps are soft — one oversized response can overshoot by a turn.
- A non-positive or non-int value raises `TypeError`.

### Model providers

`strands.models` exports `BedrockModel` eagerly and lazily resolves
`AnthropicModel`, `GeminiModel`, `LiteLLMModel`, `LlamaAPIModel`,
`LlamaCppModel`, `MistralModel`, `OllamaModel`, `OpenAIModel`,
`OpenAIResponsesModel`, `SageMakerAIModel`, `WriterModel`. Bedrock is the default
only when `model=None`.

`OpenAIModel(client_args={"api_key": ..., "base_url": ...}, model_id=...)` points
at any OpenAI-compatible gateway, which is the cheap development path.

**`Model` is subclassable, and that is the cheapest path of all.** Implement
`update_config`, `get_config`, `structured_output` and `stream`, and yield the
Bedrock-converse event shape:

```python
{"messageStart": {"role": "assistant"}}
{"contentBlockStart": {"contentBlockIndex": 0,
                       "start": {"toolUse": {"toolUseId": ..., "name": ...}}}}
{"contentBlockDelta": {"contentBlockIndex": 0,
                       "delta": {"toolUse": {"input": "<json string>"}}}}
{"contentBlockStop": {"contentBlockIndex": 0}}
{"messageStop": {"stopReason": "tool_use"}}      # or "end_turn" with a text delta
{"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
              "metrics": {"latencyMs": 0}}}
```

`agent/_model.py` does this. It drives the real event loop, the real tools and
the real intervention layer with no network and no bill, which is what makes the
test suite and the demo free to run.

### `HumanInTheLoop` (vended, unused here)

`strands.vended_interventions.hitl` exists as described: `_TRUST_RESPONSES =
{"t", "trust"}`, `_TRUSTED_TOOLS_KEY = "hitl:trusted_tools"`, trust stored in
`agent.state`. `allowed_tools` supports `"*"` and `"!tool"` negation, and
`classifier=True` enables an LLM risk classifier.

Not used, for two reasons. Its default mode pauses via interrupt, which means a
live process per pending approval. And its `name` is a fixed class attribute, so
one agent can register at most one `HumanInTheLoop` — layering two policies means
subclassing to rename. Two named handlers of our own compose without that.

---

## Multi-agent

Not used. Issue #3076 (P1, bug-validated) reports nested agent-as-tool failing to
resume after an interrupt, and a reviewer agent behind a tool is that exact
shape. `strands.multiagent` and `agent.as_tool()` both exist and were left alone.

---

## Reproducing any of this

The probes are throwaway; the two that mattered are reproduced by the test
suite. `tests/test_agent.py` runs the real event loop against
`OfflinePlannerModel` and covers `Deny` on an irreversible tool, `Deny` on an
undeclared tool, the graceful `limit_turns` stop, and the full tool sequence with
its results. `pytest tests/test_agent.py` needs no credentials and touches no
network — `tests/_agent_support.py` fails any test that opens a socket.
