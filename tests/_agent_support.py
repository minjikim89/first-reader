"""Scaffolding for the agent tests.

Two things the agent tests depend on: no network, and no inference. Both are
enforced here rather than trusted to each test.

Deliberately not a ``conftest.py``. The autouse fixtures below would otherwise
apply to every test in the directory, including the live-API integration tests
that other modules own. Each agent test module imports what it needs, so the
fixtures are scoped to the modules that asked for them::

    from _agent_support import ScriptedModel, no_network, offline  # noqa: F401
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable
from typing import Any

import pytest
from strands.models.model import Model
from strands.types.streaming import StreamEvent


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the stub fixtures and the offline planner for every test.

    The production default is ``auto`` -- the local dataset, then the live API.
    Unit tests opt into ``stub`` here explicitly and nowhere else; nothing falls
    back to it. See ``tests/test_agent_integration.py`` for the real-data runs.
    """
    monkeypatch.setenv("FIRST_READER_SOURCE", "stub")
    monkeypatch.setenv("FIRST_READER_MODEL", "offline")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any outbound HTTP call an immediate, obvious failure."""
    import socket

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)


class ScriptedModel(Model):
    """A model that replays a fixed list of turns.

    Used where a test needs the agent to attempt something the offline planner
    would never attempt, such as calling a tool it is not allowed to call.
    """

    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self.turns = list(turns)
        self.index = 0
        self.config: dict[str, Any] = {
            "model_id": "scripted",
            "context_window_limit": 200_000,
        }

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> dict[str, Any]:
        return self.config

    def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, **kwargs: Any) -> AsyncIterable[StreamEvent]:
        turn = self.turns[min(self.index, len(self.turns) - 1)]
        self.index += 1
        yield {"messageStart": {"role": "assistant"}}
        if turn["type"] == "tool":
            use_id = f"scripted-{self.index}"
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": use_id, "name": turn["name"]}},
                }
            }
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": json.dumps(turn.get("input", {}))}},
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": turn["text"]}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            }
        }


