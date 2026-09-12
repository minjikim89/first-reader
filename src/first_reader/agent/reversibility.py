"""Reversibility grades declared on every agent tool.

The grade is the sole input the intervention layer uses to decide whether a
call may run. It is declared next to the tool, not inside the handler, so the
mapping from "what this tool can undo" to "what the agent is allowed to do"
stays readable in one place.

Grades:
    REVERSIBLE   -- carrying decisions forward, aligning, preparing a summary.
                    Nothing outside this process changes. Runs freely.
    ESCALATING   -- writes a review card to the reviewer queue. Undone by
                    deleting the queue entry; no external side effect. Runs
                    freely, but is audited.
    IRREVERSIBLE -- posting to Wikipedia, or recording a verdict on a draft.
                    First Reader does not do these. Denied structurally.

Grades are stored in ``ToolSpec["annotations"]``. Verified against
strands-agents 1.54.0: that field is carried on the spec, is not forwarded to
model provider APIs, and is reachable from a ``BeforeToolCallEvent`` via
``event.selected_tool.tool_spec``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

ANNOTATION_KEY = "firstReader:reversibility"
NOTE_KEY = "firstReader:note"


class Reversibility(str, Enum):
    """How far a tool call can be walked back after it has run."""

    REVERSIBLE = "reversible"
    ESCALATING = "escalating"
    IRREVERSIBLE = "irreversible"


# MCP-standard behaviour hints, kept alongside our own grade so that a generic
# permission layer reading only the standard keys still sees the same picture.
_MCP_HINTS: dict[Reversibility, dict[str, object]] = {
    Reversibility.REVERSIBLE: {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
    Reversibility.ESCALATING: {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
    },
    Reversibility.IRREVERSIBLE: {
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
    },
}


def declare(tool_obj: Any, grade: Reversibility, note: str) -> Any:
    """Stamp a decorated tool with its reversibility grade.

    Args:
        tool_obj: A function decorated with ``@strands.tool``.
        grade: The grade to declare.
        note: One plain sentence saying what would have to be undone.

    Returns:
        The same tool object, so this can be used inline.
    """
    spec = tool_obj.tool_spec
    annotations = dict(spec.get("annotations") or {})
    annotations[ANNOTATION_KEY] = grade.value
    annotations[NOTE_KEY] = note
    annotations.update(_MCP_HINTS[grade])
    spec["annotations"] = annotations
    return tool_obj


def grade_of(tool_spec: dict[str, Any] | None) -> Reversibility | None:
    """Read the declared grade off a tool spec, or None if undeclared.

    An undeclared tool is not assumed safe. ``core.ReversibilityGate`` treats
    None as "unknown", which is handled the same as IRREVERSIBLE.
    """
    if not tool_spec:
        return None
    raw = (tool_spec.get("annotations") or {}).get(ANNOTATION_KEY)
    if raw is None:
        return None
    try:
        return Reversibility(raw)
    except ValueError:
        return None


def note_of(tool_spec: dict[str, Any] | None) -> str:
    """Read the plain-words note off a tool spec."""
    if not tool_spec:
        return ""
    return str((tool_spec.get("annotations") or {}).get(NOTE_KEY, ""))
