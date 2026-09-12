"""Turn a draft's decline history into an ordered list of what is blocking it.

Two rules shape this module.

Nothing is ever dropped for going quiet
---------------------------------------
A reason raised in decline #1 and not repeated in decline #2 is still a blocker.
Reviewers write short notices; a second reviewer who leads with `v` has not
thereby cleared `nn`. Draft:0->1 Doctrine for AI is the case that makes this
concrete: `nn` was a secondary reason on the first decline, went unmentioned on
the second, and came back as the *primary* reason on the third. A tool that
treated silence as resolution would have told that contributor they were fine.

Order is the product
--------------------
A reviewer opening a card should see, first, the thing actually standing in the
way right now:

  1. the primary reason of the most recent decline
  2. then whatever has been raised most often
  3. then whatever was raised earliest
  4. then alphabetically, so the order is stable across runs
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from ..models import AttemptSignal, Blocker, Decline, DraftHistory
from .reasons import canonical_code, family_for


@dataclass
class _Occurrences:
    """Working accumulator for one code across the whole decline history."""

    code: str
    first_seen: datetime
    decliners: list[str]
    count: int


def _codes_of(decline: Decline) -> list[str]:
    """Canonical codes raised by one decline, primary first, deduplicated.

    A decline that names the same code as both primary and secondary reason
    counts once: it is one reviewer saying one thing.
    """
    out: list[str] = []
    for raw in (decline.code, decline.secondary_code):
        code = canonical_code(raw)
        if code and code not in out:
            out.append(code)
    return out


def build_blockers(
    history: DraftHistory,
    attempts: Mapping[str, AttemptSignal] | None = None,
) -> list[Blocker]:
    """Every reason ever raised on this draft, ordered by what blocks it now.

    `attempts` maps a decline code to the `AttemptSignal` measured for it.
    `DraftHistory` carries no wikitext, so attempt signals are computed
    elsewhere (see `analysis.attempt.detect_attempt`) and injected here. A code
    with no entry gets `attempt=None`, which means "not measured", not
    "no attempt was made".
    """
    declines = sorted(history.declines, key=lambda d: (d.declined_at, d.submitted_at))
    if not declines:
        return []

    lookup = {canonical_code(k): v for k, v in (attempts or {}).items()}

    seen: dict[str, _Occurrences] = {}
    for decline in declines:
        for code in _codes_of(decline):
            entry = seen.get(code)
            if entry is None:
                seen[code] = _Occurrences(
                    code=code,
                    first_seen=decline.declined_at,
                    decliners=[decline.decliner] if decline.decliner else [],
                    count=1,
                )
                continue
            entry.count += 1
            if decline.decliner and decline.decliner not in entry.decliners:
                entry.decliners.append(decline.decliner)

    latest = declines[-1]
    latest_primary = canonical_code(latest.code)

    blockers = [
        Blocker(
            code=entry.code,
            family=family_for(entry.code),
            first_seen=entry.first_seen,
            repeat_count=entry.count,
            attempt=lookup.get(entry.code),
            # Every decliner who raised this code, oldest first. For a repeated
            # code the earlier names are the ones it was carried from; for a
            # code raised once the single name is the reviewer who raised it.
            carried_from=list(entry.decliners),
        )
        for entry in seen.values()
    ]

    blockers.sort(
        key=lambda b: (
            0 if b.code == latest_primary else 1,  # what is blocking right now
            -b.repeat_count,  # then what keeps coming back
            b.first_seen,  # then what has been waiting longest
            b.code,  # then something deterministic
        )
    )
    return blockers
