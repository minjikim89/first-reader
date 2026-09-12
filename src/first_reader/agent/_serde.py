"""JSON conversion for the shared model types.

Two consumers need this and neither can use the dataclasses directly:

1. ``agent.state`` is a ``JSONSerializableDict``. It validates every value with
   ``json.dumps`` on assignment and deep-copies it, so a dataclass cannot be
   stored there. Verified against strands-agents 1.54.0.
2. The reviewer queue on disk.

There is a third reason to have this: a ``@tool`` that returns a dataclass has
its result rendered to the model as ``str(result)`` -- the dataclass repr --
because ``json.dumps`` fails on it and the SDK falls back. Tools here return
plain dicts so the model receives real JSON.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..models import (
    REASON_FAMILY,
    AttemptSignal,
    Blocker,
    Decline,
    DraftHistory,
    ReasonFamily,
    ReviewCard,
    Revision,
)


def _dt(value: datetime) -> str:
    return value.isoformat()


def _undt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def decline_to_dict(d: Decline) -> dict[str, Any]:
    return {
        "code": d.code,
        "secondary_code": d.secondary_code,
        "decliner": d.decliner,
        "submitted_at": _dt(d.submitted_at),
        "declined_at": _dt(d.declined_at),
        "raw": d.raw,
        "family": d.family.value,
    }


def decline_from_dict(data: dict[str, Any]) -> Decline:
    return Decline(
        code=data["code"],
        secondary_code=data.get("secondary_code"),
        decliner=data["decliner"],
        submitted_at=_undt(data["submitted_at"]),
        declined_at=_undt(data["declined_at"]),
        raw=data.get("raw", ""),
    )


def revision_to_dict(r: Revision) -> dict[str, Any]:
    return {
        "revid": r.revid,
        "parentid": r.parentid,
        "user": r.user,
        "timestamp": _dt(r.timestamp),
        "comment": r.comment,
        "size": r.size,
    }


def revision_from_dict(data: dict[str, Any]) -> Revision:
    return Revision(
        revid=data["revid"],
        parentid=data["parentid"],
        user=data["user"],
        timestamp=_undt(data["timestamp"]),
        comment=data["comment"],
        size=data["size"],
    )


def history_to_dict(h: DraftHistory) -> dict[str, Any]:
    return {
        "title": h.title,
        "pageid": h.pageid,
        "declines": [decline_to_dict(d) for d in h.declines],
        "revisions": [revision_to_dict(r) for r in h.revisions],
        "is_pending": h.is_pending,
    }


def history_from_dict(data: dict[str, Any]) -> DraftHistory:
    return DraftHistory(
        title=data["title"],
        pageid=data["pageid"],
        declines=[decline_from_dict(d) for d in data.get("declines", [])],
        revisions=[revision_from_dict(r) for r in data.get("revisions", [])],
        is_pending=data.get("is_pending", False),
    )


def attempt_to_dict(a: AttemptSignal | None) -> dict[str, Any] | None:
    if a is None:
        return None
    return {
        "refs_before": a.refs_before,
        "refs_after": a.refs_after,
        "new_domains": list(a.new_domains),
        "bytes_changed": a.bytes_changed,
        "similarity": a.similarity,
        "touched_target_area": a.touched_target_area,
        "editor_comments": list(a.editor_comments),
        "added_sources": a.added_sources,
        "substantive_rewrite": a.substantive_rewrite,
    }


def attempt_from_dict(data: dict[str, Any] | None) -> AttemptSignal | None:
    if data is None:
        return None
    return AttemptSignal(
        refs_before=data["refs_before"],
        refs_after=data["refs_after"],
        new_domains=tuple(data.get("new_domains", ())),
        bytes_changed=data["bytes_changed"],
        similarity=data["similarity"],
        touched_target_area=data["touched_target_area"],
        editor_comments=tuple(data.get("editor_comments", ())),
    )


def blocker_to_dict(b: Blocker) -> dict[str, Any]:
    return {
        "code": b.code,
        "family": b.family.value,
        "first_seen": _dt(b.first_seen),
        "repeat_count": b.repeat_count,
        "attempt": attempt_to_dict(b.attempt),
        "carried_from": list(b.carried_from),
        "fixable_by_rewrite": b.fixable_by_rewrite,
    }


def blocker_from_dict(data: dict[str, Any]) -> Blocker:
    family = data.get("family")
    return Blocker(
        code=data["code"],
        family=ReasonFamily(family) if family else REASON_FAMILY.get(data["code"], ReasonFamily.UNKNOWN),
        first_seen=_undt(data["first_seen"]),
        repeat_count=data["repeat_count"],
        attempt=attempt_from_dict(data.get("attempt")),
        carried_from=list(data.get("carried_from", [])),
    )


def card_to_dict(card: ReviewCard, interpretation: Any | None = None) -> dict[str, Any]:
    """Serialise a card, optionally with the readings made about it.

    ``interpretation`` is a sibling key, never merged into ``blockers``. A
    reading is an estimate about a fact, and a reader of this file has to be
    able to tell which is which without knowing the code.
    """
    out = {
        "draft": history_to_dict(card.draft),
        "blockers": [blocker_to_dict(b) for b in card.blockers],
        "draft_comment": card.draft_comment,
        "needs_human": card.needs_human,
        "escalation_reason": card.escalation_reason,
        "wasted_rewrite": card.wasted_rewrite,
    }
    if interpretation is not None:
        out["interpretation"] = (
            interpretation.to_dict() if hasattr(interpretation, "to_dict") else interpretation
        )
    return out


def card_from_dict(data: dict[str, Any]) -> ReviewCard:
    return ReviewCard(
        draft=history_from_dict(data["draft"]),
        blockers=[blocker_from_dict(b) for b in data.get("blockers", [])],
        draft_comment=data.get("draft_comment", ""),
        needs_human=data.get("needs_human", False),
        escalation_reason=data.get("escalation_reason", ""),
    )
