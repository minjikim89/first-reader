"""Stand-ins for ``first_reader.wiki`` and ``first_reader.analysis``.

Those two modules are being written separately. Until they export the functions
this package expects, these fallbacks let the agent run end to end on fixture
drafts. ``tools.py`` prefers the real modules and only falls back to these.

The fixtures are not a test corpus. They are one draft per evidence state in the
escalation table, so that a run exercises every branch of the judgment.

Provenance. Every draft here is invented -- titles, histories, timestamps and
all -- and any resemblance to a real draft is coincidence. Every account name is
a neutral label (``Reviewer A``, ``Contributor``), never a real handle. AfC
reviewers are volunteers, and none of them should turn up in a repository
attached to a review they did not make. The one card in the demo UI that does
come from the public record lives in ``web/fixtures/cards.json``, which carries
its own ``provenance`` block.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..models import AttemptSignal, Decline, DraftHistory, Revision

_T0 = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


def _decline(code, decliner, day, secondary=None):
    return Decline(
        code=code,
        secondary_code=secondary,
        decliner=decliner,
        submitted_at=_T0 + timedelta(days=day - 7),
        declined_at=_T0 + timedelta(days=day),
        raw=f"{{{{AfC submission|d|{code}|u=Contributor|ns=118|decliner={decliner}}}}}",
    )


def _rev(revid, user, day, size, comment):
    return Revision(
        revid=revid,
        parentid=revid - 1,
        user=user,
        timestamp=_T0 + timedelta(days=day),
        comment=comment,
        size=size,
    )


#: One draft per row of the escalation table.
FIXTURES: dict[str, DraftHistory] = {
    # WASTED_REWRITE -- rewriting against a reason rewriting cannot fix.
    "Draft:Corner Bakery Collective": DraftHistory(
        title="Draft:Corner Bakery Collective",
        pageid=70000001,
        declines=[_decline("nn", "Reviewer A", 10), _decline("nn", "Reviewer A", 40)],
        revisions=[
            _rev(101, "Contributor", 12, 4200, "expanding the history section"),
            _rev(102, "Contributor", 20, 5900, "rewrote lead, trimmed promotional tone"),
            _rev(103, "Contributor", 33, 6100, "copyedit"),
        ],
        is_pending=True,
    ),
    # UNKNOWN_FAMILY -- a code we refuse to guess about.
    #
    # The code below must stay absent from models.REASON_FAMILY. If it is ever
    # mapped, this fixture stops exercising the UNKNOWN branch and
    # test_unknown_reason_family_goes_to_a_reviewer fails; pick another
    # unmapped code here rather than weakening that test. `cv` (copyright
    # violation) is deliberately unclassified in analysis/reasons.py: the
    # template says "rewrite it entirely in your own words" (WRITING) while AFCH
    # simultaneously files a WP:G12 speedy deletion (STRUCTURAL).
    "Draft:Helvetia Signals": DraftHistory(
        title="Draft:Helvetia Signals",
        pageid=70000002,
        declines=[_decline("cv", "Reviewer B", 15)],
        revisions=[_rev(201, "Contributor", 16, 3100, "responding to review")],
        is_pending=True,
    ),
    # INCONSISTENT_REVIEWERS -- three reviewers, three different reasons.
    "Draft:Mira Okonkwo": DraftHistory(
        title="Draft:Mira Okonkwo",
        pageid=70000003,
        declines=[
            _decline("ai", "Reviewer C", 8),
            _decline("nn", "Reviewer D", 22),
            _decline("adv", "Reviewer E", 45),
        ],
        revisions=[
            _rev(301, "Contributor", 10, 2800, "rewrite per review"),
            _rev(302, "Contributor", 25, 3300, "added two references"),
        ],
        is_pending=True,
    ),
    # NO_PRECEDENT at cold start, ROUTINE once the pattern has been seen.
    "Draft:Tuvan Throat Singing in Film": DraftHistory(
        title="Draft:Tuvan Throat Singing in Film",
        pageid=70000004,
        declines=[_decline("ai", "Reviewer F", 18)],
        revisions=[_rev(401, "Contributor", 19, 5200, "rewrote in my own words")],
        is_pending=True,
    ),
    # A reviewer who wrote the reason as free text instead of picking a code.
    # `reason` is the AfC code for "see the prose"; the prose is the third
    # positional parameter of the template, and nothing deterministic can read it.
    "Draft:Ravensworth Signal Box": DraftHistory(
        title="Draft:Ravensworth Signal Box",
        pageid=70000006,
        declines=[
            Decline(
                code="reason",
                secondary_code=None,
                decliner="Reviewer H",
                submitted_at=_T0 + timedelta(days=4),
                declined_at=_T0 + timedelta(days=11),
                raw=(
                    "{{AfC submission|d|reason|Everything here comes from the preservation "
                    "society's own newsletter. Until somebody outside the society has "
                    "written about the box at length, there is nothing to build on."
                    "|u=Contributor|ns=118|decliner=Reviewer H}}"
                ),
            )
        ],
        revisions=[_rev(601, "Contributor", 12, 3400, "added more detail from the newsletter")],
        is_pending=True,
    ),
    # REPEATED_REASON_NO_ATTEMPT -- the same reason, three times, nothing moved.
    "Draft:Blue Ridge Analytics": DraftHistory(
        title="Draft:Blue Ridge Analytics",
        pageid=70000005,
        declines=[
            _decline("nn", "Reviewer G", 5),
            _decline("nn", "Reviewer G", 25),
            _decline("nn", "Reviewer G", 50),
        ],
        revisions=[_rev(501, "Contributor", 6, 2100, "resubmitting")],
        is_pending=True,
    ),
}


#: Attempt signals keyed by (title, code).
#:
#: The real ``analysis`` module derives these from wikitext diffs. This stub
#: cannot -- ``Revision`` carries sizes, not text -- so the fixtures state them.
#: Everything here stays mechanically derivable in principle: reference counts,
#: added domains, byte deltas, similarity, and whether the edit touched the area
#: the reason pointed at. None of it is a judgment about whether the reason was
#: resolved; making that call is the review itself.
ATTEMPTS: dict[tuple[str, str], AttemptSignal] = {
    ("Draft:Corner Bakery Collective", "nn"): AttemptSignal(
        refs_before=4,
        refs_after=4,
        new_domains=(),
        bytes_changed=1900,
        similarity=0.41,
        touched_target_area=False,
        editor_comments=("rewrote lead, trimmed promotional tone", "copyedit"),
    ),
    ("Draft:Helvetia Signals", "cv"): AttemptSignal(
        refs_before=2,
        refs_after=3,
        new_domains=("swissinfo.ch",),
        bytes_changed=420,
        similarity=0.88,
        touched_target_area=True,
        editor_comments=("responding to review",),
    ),
    ("Draft:Mira Okonkwo", "adv"): AttemptSignal(
        refs_before=3,
        refs_after=5,
        new_domains=("thenationonlineng.net", "premiumtimesng.com"),
        bytes_changed=500,
        similarity=0.82,
        touched_target_area=True,
        editor_comments=("added two references",),
    ),
    # Measured, and it measured nothing: the resubmission is byte-identical
    # apart from the template. This is what "no attempt" looks like when it has
    # actually been looked at -- distinct from a code with no entry at all,
    # which means nobody looked.
    ("Draft:Blue Ridge Analytics", "nn"): AttemptSignal(
        refs_before=1,
        refs_after=1,
        new_domains=(),
        bytes_changed=0,
        similarity=1.0,
        touched_target_area=False,
        editor_comments=("resubmitting",),
    ),
    ("Draft:Tuvan Throat Singing in Film", "ai"): AttemptSignal(
        refs_before=6,
        refs_after=6,
        new_domains=(),
        bytes_changed=1400,
        similarity=0.55,
        touched_target_area=True,
        editor_comments=("rewrote in my own words",),
    ),
}


def fetch_draft_history(title: str) -> DraftHistory:
    """Return the fixture history for ``title``.

    Raises:
        KeyError: If the title is not one of the fixtures. The stub never
            invents a history; an unknown draft is an unknown draft.
    """
    try:
        return FIXTURES[title]
    except KeyError as exc:
        known = ", ".join(sorted(FIXTURES))
        raise KeyError(f"no fixture for {title!r}; fixtures are: {known}") from exc


def attempts_for(history: DraftHistory) -> dict[str, AttemptSignal]:
    """Attempt signals for one fixture draft, keyed by decline code.

    Mirrors what ``analysis.detect_attempt`` measures from wikitext, which this
    stub has none of. A code absent from :data:`ATTEMPTS` is *unmeasured*, and
    the escalation judgment treats that as a reason to involve a human.
    """
    return {
        code: signal
        for (title, code), signal in ATTEMPTS.items()
        if title == history.title
    }


def titles() -> list[str]:
    """Fixture titles, for the demo entry point."""
    return sorted(FIXTURES)
