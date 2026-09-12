"""Model providers, and an offline planner that stands in for one.

Inference is the running cost of this project, so the default provider is not a
provider. ``OfflinePlannerModel`` implements the ``strands.models.Model``
interface and emits a deterministic sequence of tool calls. It drives the real
event loop, the real tools and the real intervention layer, at zero cost and
with no network. Tests use it; so does the demo run.

It is a stand-in, not a language model. It does not read the evidence and it
does not write prose. Everything it decides, the deterministic core had already
decided. Set ``FIRST_READER_MODEL`` to use a real provider.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import AsyncIterable
from typing import Any

from strands.models.model import Model
from strands.types.streaming import StreamEvent

logger = logging.getLogger(__name__)

#: Providers already reported as credentialed-but-uninstalled, so a batch run
#: says it once instead of once per draft.
_WARNED_UNAVAILABLE: set[str] = set()

TITLE_MARKER = re.compile(r"<title>(.*?)</title>", re.DOTALL)

#: The fixed order the offline planner walks. Mirrors what the system prompt
#: asks a real model to do, so a run under either provider takes the same shape.
PLAN = (
    "get_draft_history",
    "analyze_blockers",
    "interpret_reasons",
    "prepare_comment",
    "judge_escalation",
)

#: Cheap by default. Interpretation is one short structured answer per draft;
#: paying frontier prices for it would be the cost failure this project cannot
#: afford, and the task does not need frontier reasoning.
DEFAULT_CHEAP_MODEL = "openai/gpt-4o-mini"


class OfflinePlannerModel(Model):
    """Deterministic stand-in for a model provider.

    Walks :data:`PLAN`, then calls ``escalate_to_reviewer`` if and only if the
    ``judge_escalation`` result said to, then ends the turn.
    """

    #: Read by ``interpret.Interpreter``: this model can drive the loop but
    #: cannot read prose, so interpretation is skipped rather than attempted.
    is_offline_stub = True

    def __init__(self, *, model_id: str = "first-reader-offline-planner") -> None:
        self.config: dict[str, Any] = {
            "model_id": model_id,
            "context_window_limit": 200_000,
        }
        self.calls = 0

    # -- Model interface --

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> dict[str, Any]:
        return self.config

    def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("OfflinePlannerModel does not produce structured output")

    async def stream(
        self,
        messages: Any,
        tool_specs: Any = None,
        system_prompt: Any = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Emit one turn: either a tool call or a closing sentence."""
        self.calls += 1
        title = self._title(messages)
        done = self._completed_tools(messages)

        step = next((name for name in PLAN if name not in done), None)
        if step is not None:
            async for event in self._tool_use(step, {"title": title}):
                yield event
            return

        verdict = self._verdict(messages)
        if verdict is not None and verdict.get("escalate") and "escalate_to_reviewer" not in done:
            async for event in self._tool_use(
                "escalate_to_reviewer",
                {"title": title, "why": verdict.get("reason", "A reviewer should look at this.")},
            ):
                yield event
            return

        async for event in self._text(self._closing(title, verdict, done)):
            yield event

    # -- turn shapes --

    async def _tool_use(self, name: str, arguments: dict[str, Any]) -> AsyncIterable[StreamEvent]:
        tool_use_id = f"offline-{self.calls}-{name}"
        yield {"messageStart": {"role": "assistant"}}
        yield {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": tool_use_id, "name": name}},
            }
        }
        yield {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": json.dumps(arguments)}},
            }
        }
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "tool_use"}}
        yield self._usage()

    async def _text(self, text: str) -> AsyncIterable[StreamEvent]:
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
        yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}}
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield self._usage()

    @staticmethod
    def _usage() -> StreamEvent:
        return {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            }
        }

    # -- reading the conversation back --

    @staticmethod
    def _title(messages: Any) -> str:
        for message in messages:
            for block in message.get("content", []):
                text = block.get("text") if isinstance(block, dict) else None
                if text:
                    match = TITLE_MARKER.search(text)
                    if match:
                        return match.group(1).strip()
        return ""

    @staticmethod
    def _tool_name_by_id(messages: Any) -> dict[str, str]:
        names: dict[str, str] = {}
        for message in messages:
            for block in message.get("content", []):
                if isinstance(block, dict) and "toolUse" in block:
                    names[block["toolUse"]["toolUseId"]] = block["toolUse"]["name"]
        return names

    @classmethod
    def _completed_tools(cls, messages: Any) -> set[str]:
        """Tool names that already have a result, successful or denied."""
        names = cls._tool_name_by_id(messages)
        done: set[str] = set()
        for message in messages:
            for block in message.get("content", []):
                if isinstance(block, dict) and "toolResult" in block:
                    name = names.get(block["toolResult"].get("toolUseId", ""))
                    if name:
                        done.add(name)
        return done

    @classmethod
    def _verdict(cls, messages: Any) -> dict[str, Any] | None:
        """The most recent ``judge_escalation`` result, parsed."""
        names = cls._tool_name_by_id(messages)
        found: dict[str, Any] | None = None
        for message in messages:
            for block in message.get("content", []):
                if not (isinstance(block, dict) and "toolResult" in block):
                    continue
                result = block["toolResult"]
                if names.get(result.get("toolUseId", "")) != "judge_escalation":
                    continue
                for item in result.get("content", []):
                    text = item.get("text") if isinstance(item, dict) else None
                    if not text:
                        continue
                    try:
                        found = json.loads(text)
                    except json.JSONDecodeError:
                        continue
        return found

    @staticmethod
    def _closing(title: str, verdict: dict[str, Any] | None, done: set[str]) -> str:
        if verdict is None:
            return f"Could not judge {title}: no escalation verdict was returned."
        if verdict.get("escalate"):
            state = "queued for a reviewer" if "escalate_to_reviewer" in done else "not queued"
            return f"{title}: {state}. {verdict.get('reason', '')}"
        return f"{title}: handled without a reviewer. {verdict.get('reason', '')}"


def resolve_model(provider: str | None = None) -> Model:
    """Return the model provider named by ``FIRST_READER_MODEL``.

    Defaults to ``auto``: the first provider whose credentials are present, and
    :class:`OfflinePlannerModel` when none are. The offline planner is the
    fallback rather than the default, because with it the agent can walk the
    plan but cannot read a word of what a reviewer wrote, and reading that prose
    is most of what the model is here for.

    Bedrock is never chosen by ``auto``. It is the one option that bills per run
    on ambient credentials, and a test loop should not reach it by accident.

    Supported values:
        ``auto``       -- openrouter, then openai, then anthropic, then offline.
        ``openrouter`` -- ``OPENROUTER_API_KEY``. Cheap, and the development path.
        ``openai``     -- ``OPENAI_API_KEY``, optional ``OPENAI_BASE_URL``.
        ``anthropic``  -- ``ANTHROPIC_API_KEY``.
        ``bedrock``    -- ambient AWS credentials.
        ``ollama``     -- ``OLLAMA_HOST``, default ``http://localhost:11434``.
        ``offline``    -- :class:`OfflinePlannerModel`, no network, no cost.

    ``FIRST_READER_MODEL_ID`` overrides the model id for any real provider.

    Raises:
        ValueError: On an unknown provider name. Falling back to a default here
            would mean a typo silently changes which model reads the evidence.
    """
    name = (provider or os.getenv("FIRST_READER_MODEL", "auto")).strip().lower()

    if name == "auto":
        for candidate, key in (("openrouter", "OPENROUTER_API_KEY"),
                               ("openai", "OPENAI_API_KEY"),
                               ("anthropic", "ANTHROPIC_API_KEY")):
            if not os.getenv(key):
                continue
            try:
                return resolve_model(candidate)
            except ImportError as exc:
                # A key in the environment is not the same as a usable provider.
                # Anyone with OPENAI_API_KEY exported and the extra not installed
                # would otherwise get ModuleNotFoundError on their first command,
                # which is a bad way to meet a project. Say so and keep going.
                #
                # Once per process: a fresh agent is built per draft, so a batch
                # would otherwise repeat the same warning once a draft.
                if candidate not in _WARNED_UNAVAILABLE:
                    _WARNED_UNAVAILABLE.add(candidate)
                    logger.warning(
                        "%s has credentials but its client is not installed (%s); "
                        "trying the next provider. Install it with "
                        'pip install "first-reader[%s]".',
                        candidate, exc, candidate,
                    )
        return OfflinePlannerModel()

    if name in {"offline", "scripted", "none"}:
        return OfflinePlannerModel()

    model_id = os.getenv("FIRST_READER_MODEL_ID")

    if name == "openrouter":
        from strands.models.openai import OpenAIModel

        return OpenAIModel(
            client_args={
                "api_key": os.environ["OPENROUTER_API_KEY"],
                "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            },
            model_id=model_id or DEFAULT_CHEAP_MODEL,
            params={"max_tokens": 1024, "temperature": 0},
        )

    if name == "openai":
        from strands.models.openai import OpenAIModel

        client_args: dict[str, Any] = {"api_key": os.environ["OPENAI_API_KEY"]}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            client_args["base_url"] = base_url
        return OpenAIModel(
            client_args=client_args,
            model_id=model_id or "gpt-4o-mini",
            params={"max_tokens": 1024, "temperature": 0},
        )

    if name == "anthropic":
        from strands.models.anthropic import AnthropicModel

        return AnthropicModel(
            client_args={"api_key": os.environ["ANTHROPIC_API_KEY"]},
            model_id=model_id or "claude-haiku-4-5",
            max_tokens=1024,
            params={"temperature": 0},
        )

    if name == "bedrock":
        from strands.models.bedrock import BedrockModel

        return BedrockModel(model_id=model_id or "us.anthropic.claude-haiku-4-5-20251001-v1:0")

    if name == "ollama":
        from strands.models.ollama import OllamaModel

        return OllamaModel(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            model_id=model_id or "qwen2.5:7b",
        )

    raise ValueError(
        f"unknown FIRST_READER_MODEL {name!r}; expected one of: "
        "auto, openrouter, openai, anthropic, bedrock, ollama, offline"
    )
