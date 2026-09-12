"""Parser for the ``{{AfC submission}}`` template stack.

A declined draft carries one template per decline, newest at the top of the
page. This module turns that stack into ``models.Decline`` records, oldest
first, and reports what it could not parse instead of swallowing it.

Template shape, verified against ``Template:AfC submission/doc`` templatedata
and against 1,200+ live templates sampled from the pending and declined
categories:

* positional 1 -- status. ``d`` declined, ``r`` reviewing, ``t``/``T``
  unsubmitted draft, empty pending. A ``d`` template with a truthy ``reject``
  parameter renders as *rejected* rather than *declined*.
* positional 2 -- decline reason code (``nn``, ``ai``, ``v``, ...).
* positional 3 -- free-text detail for that code. It is NOT a second reason
  code; the second reason code is the named ``reason2`` parameter.
* named -- ``u``, ``ns``, ``ts``, ``declinets``, ``decliner``, ``reason2``,
  ``details``, ``details2``, ``small``, ``demo``, ``reject``, ``type``,
  ``reviewer``, ``reviewts``.

Order is not fixed: positional arguments are interleaved with named ones in
real pages, so positions are counted over the non-named arguments in order of
appearance, exactly as MediaWiki does.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from first_reader.models import REASON_FAMILY, Decline, DraftHistory

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from first_reader.wiki.client import WikiClient

__all__ = [
    "SubmissionStatus",
    "AfCTemplate",
    "ParseStats",
    "ParseResult",
    "iter_afc_templates",
    "parse_afc_template",
    "parse_declines",
    "parse_timestamp",
    "fetch_draft_history",
    "fetch_draft_histories",
]

# `{{AfC submission`, `{{AFC submission`, `{{afc_submission`, with optional
# leading colon/whitespace. Subpage transclusions such as
# `{{AfC submission/draft}}` are a different template and are not submissions.
_TEMPLATE_NAME = re.compile(r"^\{\{\s*:?\s*[Aa][Ff][Cc][ _]submission\s*(?=[|}])")

# Regions whose contents MediaWiki does not expand. A template inside one of
# these is documentation, not a real submission.
_INERT_REGION = re.compile(
    r"<!--.*?-->|<nowiki>.*?</nowiki>|<pre[^>]*>.*?</pre>|<syntaxhighlight[^>]*>.*?</syntaxhighlight>",
    re.DOTALL | re.IGNORECASE,
)

_TS_RE = re.compile(r"^\d{14}$")
# Two malformed shapes that occur in the wild and decode unambiguously.
# `ts=202603311800` -- AFCH wrote the timestamp without seconds.
_TS_NO_SECONDS_RE = re.compile(r"^\d{12}$")
# `ts=16:59, 14 August 2026 (UTC)` -- a reviewer pasted a signature timestamp
# instead of letting AFCH fill the field.
_TS_SIGNATURE_RE = re.compile(
    r"^(?P<h>\d{1,2}):(?P<mi>\d{2}),\s*(?P<d>\d{1,2})\s+(?P<mon>[A-Za-z]+)\s+(?P<y>\d{4})"
    r"(?:\s*\(UTC\))?$"
)
_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}


class SubmissionStatus(str, Enum):
    """Value of positional argument 1."""

    PENDING = "pending"  # empty first argument
    DECLINED = "declined"  # d
    REJECTED = "rejected"  # d + reject=<truthy>
    REVIEWING = "reviewing"  # r
    DRAFT = "draft"  # t / T -- never submitted
    UNKNOWN = "unknown"  # anything else; never guessed at


_STATUS_BY_ARG = {
    "": SubmissionStatus.PENDING,
    "d": SubmissionStatus.DECLINED,
    "r": SubmissionStatus.REVIEWING,
    "t": SubmissionStatus.DRAFT,
}


@dataclass(frozen=True)
class AfCTemplate:
    """One raw ``{{AfC submission}}`` occurrence, before validation."""

    raw: str
    status: SubmissionStatus
    positional: tuple[str, ...]  # 1-indexed args in order, index 0 == arg 1
    named: dict[str, str]

    def arg(self, index: int) -> str | None:
        """1-indexed positional argument, honouring explicit ``|2=`` form."""
        explicit = self.named.get(str(index))
        if explicit is not None:
            return explicit
        if 1 <= index <= len(self.positional):
            return self.positional[index - 1]
        return None


@dataclass
class ParseStats:
    """What the parse saw, including what it refused to turn into a Decline.

    Skips are counted, never silent. ``skip_reasons`` keys are stable strings
    so a collection run can aggregate them.
    """

    templates_found: int = 0
    declines_found: int = 0  # templates whose status is declined/rejected
    declines_parsed: int = 0
    declines_skipped: int = 0
    rejected_found: int = 0
    empty_code: int = 0
    skip_reasons: Counter[str] = field(default_factory=Counter)
    status_counts: Counter[str] = field(default_factory=Counter)
    unmapped_codes: Counter[str] = field(default_factory=Counter)

    @property
    def had_failure(self) -> bool:
        return self.declines_skipped > 0

    @property
    def skip_rate(self) -> float:
        return self.declines_skipped / self.declines_found if self.declines_found else 0.0

    def merge(self, other: ParseStats) -> None:
        self.templates_found += other.templates_found
        self.declines_found += other.declines_found
        self.declines_parsed += other.declines_parsed
        self.declines_skipped += other.declines_skipped
        self.rejected_found += other.rejected_found
        self.empty_code += other.empty_code
        self.skip_reasons.update(other.skip_reasons)
        self.status_counts.update(other.status_counts)
        self.unmapped_codes.update(other.unmapped_codes)

    def to_dict(self) -> dict[str, object]:
        return {
            "templates_found": self.templates_found,
            "declines_found": self.declines_found,
            "declines_parsed": self.declines_parsed,
            "declines_skipped": self.declines_skipped,
            "rejected_found": self.rejected_found,
            "empty_code": self.empty_code,
            "skip_reasons": dict(self.skip_reasons),
            "status_counts": dict(self.status_counts),
            "unmapped_codes": dict(self.unmapped_codes),
        }


@dataclass(frozen=True)
class ParseResult:
    declines: list[Decline]  # oldest first
    stats: ParseStats
    is_pending: bool = False


# --------------------------------------------------------------------- lexing


def _mask_inert_regions(text: str) -> str:
    """Blank out comments/nowiki/pre while preserving offsets."""

    def blank(match: re.Match[str]) -> str:
        return " " * (match.end() - match.start())

    return _INERT_REGION.sub(blank, text)


def _iter_balanced_templates(text: str):
    """Yield ``(start, end)`` spans of brace-balanced ``{{...}}`` regions.

    Scans left to right and skips past a template's whole body, so nested
    templates inside a parameter do not produce spurious top-level matches.
    """
    i = 0
    length = len(text)
    while True:
        start = text.find("{{", i)
        if start < 0:
            return
        depth = 0
        j = start
        while j < length - 1:
            pair = text[j : j + 2]
            if pair == "{{":
                depth += 1
                j += 2
                continue
            if pair == "}}":
                depth -= 1
                j += 2
                if depth == 0:
                    break
                continue
            j += 1
        else:
            return  # unbalanced tail: nothing more to find
        if depth != 0:
            return
        yield start, j
        i = start + 2  # allow finding a template nested inside this one


def _split_parameters(body: str) -> list[str]:
    """Split a template body on top-level ``|``.

    Depth is tracked for ``{{}}``, ``[[]]`` and ``{|...|}`` wikitables, since
    all three legally appear inside a decline's free-text detail parameter.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    i = 0
    length = len(body)
    while i < length:
        pair = body[i : i + 2]
        if pair in ("{{", "[["):
            depth += 1
            current.append(pair)
            i += 2
            continue
        if pair in ("}}", "]]"):
            depth -= 1
            current.append(pair)
            i += 2
            continue
        if pair == "{|":
            depth += 1
            current.append(pair)
            i += 2
            continue
        if pair == "|}":
            depth -= 1
            current.append(pair)
            i += 2
            continue
        if body[i] == "|" and depth <= 0:
            parts.append("".join(current))
            current = []
            i += 1
            continue
        current.append(body[i])
        i += 1
    parts.append("".join(current))
    return parts


def _split_named(part: str) -> tuple[str | None, str]:
    """Return ``(name, value)``; name is None for a positional argument.

    A parameter is named when it has an ``=`` outside any nested link or
    template -- the same rule MediaWiki applies.
    """
    depth = 0
    i = 0
    length = len(part)
    while i < length:
        pair = part[i : i + 2]
        if pair in ("{{", "[["):
            depth += 1
            i += 2
            continue
        if pair in ("}}", "]]"):
            depth -= 1
            i += 2
            continue
        if part[i] == "=" and depth <= 0:
            name = part[:i].strip()
            if name and "\n" not in name:
                return name, part[i + 1 :].strip()
            return None, part.strip()
        i += 1
    return None, part.strip()


def iter_afc_templates(wikitext: str) -> list[AfCTemplate]:
    """Every ``{{AfC submission}}`` in the page, in document order.

    Templates inside comments, ``<nowiki>`` or ``<pre>`` are ignored: they are
    documentation, not submissions.
    """
    masked = _mask_inert_regions(wikitext)
    found: list[AfCTemplate] = []
    for start, end in _iter_balanced_templates(masked):
        span = masked[start:end]
        if not _TEMPLATE_NAME.match(span):
            continue
        raw = wikitext[start:end]  # unmasked text for the record
        parts = _split_parameters(raw[2:-2])
        positional: list[str] = []
        named: dict[str, str] = {}
        for part in parts[1:]:
            name, value = _split_named(part)
            if name is None:
                positional.append(value)
            else:
                named[name.lower()] = value

        arg1 = (named.get("1") if named.get("1") is not None else (positional[0] if positional else "")) or ""
        status = _STATUS_BY_ARG.get(arg1.strip().lower(), SubmissionStatus.UNKNOWN)
        if status is SubmissionStatus.DECLINED and _is_truthy(named.get("reject")):
            status = SubmissionStatus.REJECTED
        found.append(
            AfCTemplate(raw=raw, status=status, positional=tuple(positional), named=named)
        )
    return found


def _is_truthy(value: str | None) -> bool:
    """AfC treats any non-empty value as set; ``no``/``false`` are not used."""
    return bool(value and value.strip() and value.strip().lower() not in {"no", "false", "0"})


def parse_timestamp(value: str | None) -> datetime | None:
    """``YYYYMMDDHHMMSS`` in UTC. Returns None rather than raising.

    Two degraded forms are also accepted, because both decode to exactly one
    instant and both were cross-checked against the sibling ``declinets`` on
    the same template when they were found live:

    * 12 digits -- ``YYYYMMDDHHMM``, seconds omitted; read as ``:00``.
    * ``16:59, 14 August 2026 (UTC)`` -- a pasted signature timestamp.

    Anything else returns None, which makes the caller skip and count the
    template rather than invent a time.
    """
    if not value:
        return None
    text = value.strip()

    if _TS_RE.match(text):
        fmt = "%Y%m%d%H%M%S"
    elif _TS_NO_SECONDS_RE.match(text):
        fmt = "%Y%m%d%H%M"
    else:
        match = _TS_SIGNATURE_RE.match(text)
        if match is None:
            return None
        month = _MONTHS.get(match.group("mon").lower())
        if month is None:
            return None
        try:
            return datetime(
                int(match.group("y")),
                month,
                int(match.group("d")),
                int(match.group("h")),
                int(match.group("mi")),
                tzinfo=timezone.utc,
            )
        except ValueError:
            return None

    try:
        return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# -------------------------------------------------------------------- parsing


def parse_afc_template(template: AfCTemplate) -> tuple[Decline | None, str | None]:
    """Turn one declined-status template into a ``Decline``.

    Returns ``(decline, None)`` on success or ``(None, skip_reason)`` when a
    field the data contract requires is missing or malformed. Never raises.
    """
    if template.status not in (SubmissionStatus.DECLINED, SubmissionStatus.REJECTED):
        return None, "not_a_decline"

    code = (template.arg(2) or "").strip()

    decliner = (template.named.get("decliner") or "").strip()
    if not decliner:
        return None, "missing_decliner"

    submitted_at = parse_timestamp(template.named.get("ts"))
    if submitted_at is None:
        return None, "bad_ts"

    declined_at = parse_timestamp(template.named.get("declinets"))
    if declined_at is None:
        return None, "bad_declinets"

    secondary = (template.named.get("reason2") or "").strip() or None

    return (
        Decline(
            code=code,
            secondary_code=secondary,
            decliner=decliner,
            submitted_at=submitted_at,
            declined_at=declined_at,
            raw=template.raw,
        ),
        None,
    )


def parse_declines(wikitext: str) -> ParseResult:
    """Parse the whole template stack of one page.

    Declines come back oldest first, ordered by ``declinets``. Anything that
    could not be parsed is counted in ``stats``, not raised and not dropped
    silently.
    """
    stats = ParseStats()
    declines: list[Decline] = []
    is_pending = False

    if not wikitext:
        return ParseResult(declines=[], stats=stats, is_pending=False)

    templates = iter_afc_templates(wikitext)
    stats.templates_found = len(templates)

    for template in templates:
        stats.status_counts[template.status.value] += 1
        if template.status in (SubmissionStatus.PENDING, SubmissionStatus.REVIEWING):
            is_pending = True
        if template.status not in (SubmissionStatus.DECLINED, SubmissionStatus.REJECTED):
            continue

        stats.declines_found += 1
        if template.status is SubmissionStatus.REJECTED:
            stats.rejected_found += 1

        decline, skip_reason = parse_afc_template(template)
        if decline is None:
            stats.declines_skipped += 1
            stats.skip_reasons[skip_reason or "unknown"] += 1
            continue

        if not decline.code:
            stats.empty_code += 1
        elif decline.code not in REASON_FAMILY:
            stats.unmapped_codes[decline.code] += 1

        declines.append(decline)
        stats.declines_parsed += 1

    declines.sort(key=lambda d: (d.declined_at, d.submitted_at))
    return ParseResult(declines=declines, stats=stats, is_pending=is_pending)


# ----------------------------------------------------------------- assembling


async def fetch_draft_history(
    client: WikiClient,
    title: str,
    *,
    with_revisions: bool = True,
    revision_limit: int | None = None,
) -> tuple[DraftHistory | None, ParseStats]:
    """Wikitext + revisions for one draft, assembled into a ``DraftHistory``.

    Returns ``(None, stats)`` when the page does not exist.
    """
    histories, stats = await fetch_draft_histories(
        client, [title], with_revisions=with_revisions, revision_limit=revision_limit
    )
    return (histories[0] if histories else None), stats


async def fetch_draft_histories(
    client: WikiClient,
    titles: Sequence[str],
    *,
    with_revisions: bool = True,
    revision_limit: int | None = None,
) -> tuple[list[DraftHistory], ParseStats]:
    """Batch version of :func:`fetch_draft_history`.

    Wikitext is fetched 50 titles per request. Revision history costs one
    request per title (the API cannot batch a full ``rvlimit`` listing across
    pages), so pass ``with_revisions=False`` for a cheap screening pass.
    """
    stats = ParseStats()
    texts = await client.get_wikitext_batch(list(titles))
    info = await client.get_page_info(list(titles))

    histories: list[DraftHistory] = []
    for title in titles:
        wikitext = texts.get(title)
        if wikitext is None:
            stats.skip_reasons["page_missing"] += 1
            continue
        page = info.get(title, {})
        result = parse_declines(wikitext)
        stats.merge(result.stats)
        history = DraftHistory(
            title=page.get("title", title),
            pageid=page.get("pageid", 0),
            declines=result.declines,
            revisions=[],
            is_pending=result.is_pending,
        )
        if with_revisions:
            history.revisions = await client.get_revisions(title, limit=revision_limit)
        histories.append(history)
    return histories, stats
