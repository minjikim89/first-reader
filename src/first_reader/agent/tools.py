"""The tools the agent may call, and the pure functions underneath them.

Each tool is declared with a reversibility grade (see ``reversibility.py``).
That declaration is what ``core.ReversibilityGate`` reads; the gate contains no
tool names of its own, so adding a tool without a grade fails closed rather
than silently gaining permission.

Two things about the tool boundary are shaped by verified SDK behaviour rather
than preference:

*Tools take a title, not an object.* A ``@tool`` result that is not
JSON-serialisable is rendered to the model as ``str(result)``. A ``DraftHistory``
would arrive as a dataclass repr -- large, and lossy in a way nothing downstream
can detect. So the rich objects live in ``agent.state`` keyed by title, and each
tool returns a small dict that the SDK serialises as real JSON.

*Nothing here waits for a human.* ``escalate_to_reviewer`` writes a card to a
durable queue and returns. A run finishes; a reviewer looks later. Holding a
process open across an approval would put a live ``toolUse`` without its
``toolResult`` at the mercy of anything that ends the turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from strands import tool
from strands.types.tools import ToolContext

from ..models import AttemptSignal, Blocker, DraftHistory, ReasonFamily, ReviewCard
from . import _serde
from .escalation import (
    DEFAULT_WEIGHTS,
    Weights,
    apply_rule,
    judge,
    pattern_key,
    signature,
)
from .interpret import Interpretation, Interpreter, Reading, free_text_reason
from .policy import TIGHTEN, PolicyStore
from .reversibility import Reversibility, declare

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_QUEUE_DIR = Path("data/queue")

HISTORY_KEY = "first_reader:history"
BLOCKERS_KEY = "first_reader:blockers"
CARD_KEY = "first_reader:card"
JUDGMENT_KEY = "first_reader:judgment"
INTERPRETATION_KEY = "first_reader:interpretation"
QUEUED_KEY = "first_reader:queued"
AUDIT_KEY = "first_reader:audit"
QUEUE_DIR_KEY = "first_reader:queue_dir"


# --- where the data comes from ----------------------------------------------


class Source(str, Enum):
    """Where ``fetch_history`` looks for a draft.

    ``AUTO`` is the default and the only one anything in production uses: the
    local dataset first because it is free and instant, the live API when the
    draft is not in it.

    ``STUB`` exists for tests. It is never reached by fallback -- an earlier
    version of this module fell back to fixtures when the real modules were
    missing, and the result was an agent that answered confidently about five
    invented drafts and raised ``KeyError`` on every real one, while 183 tests
    passed. A stub that can be reached by accident is worse than no stub.
    """

    AUTO = "auto"
    DATASET = "dataset"
    WIKIPEDIA = "wikipedia"
    STUB = "stub"


_SOURCE_ALIASES = {
    "auto": Source.AUTO,
    "dataset": Source.DATASET,
    "cache": Source.DATASET,
    "local": Source.DATASET,
    "wikipedia": Source.WIKIPEDIA,
    "wiki": Source.WIKIPEDIA,
    "live": Source.WIKIPEDIA,
    "stub": Source.STUB,
    "fixtures": Source.STUB,
}

#: Datasets, in the order they are looked for. The raw collection is not
#: distributed -- it carries real reviewer and contributor handles -- so a fresh
#: clone finds the anonymised file instead. The two are structurally identical;
#: only the account labels differ. See ``scripts/anonymize.py``.
DATASET_CANDIDATES = (
    Path("data/dataset.jsonl"),
    Path("data/dataset.anonymized.jsonl"),
)


def default_dataset() -> Path:
    """The dataset to read, resolved when it is needed rather than at import."""
    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    return DATASET_CANDIDATES[-1]


class HistoryUnavailable(RuntimeError):
    """A draft's history could not be obtained, and we will not invent one."""


def current_source() -> Source:
    """The source named by ``FIRST_READER_SOURCE``, defaulting to ``auto``."""
    raw = os.getenv("FIRST_READER_SOURCE", Source.AUTO.value).strip().lower()
    try:
        return _SOURCE_ALIASES[raw]
    except KeyError:
        raise ValueError(
            f"unknown FIRST_READER_SOURCE {raw!r}; expected one of: "
            + ", ".join(sorted({s.value for s in Source}))
        ) from None


@lru_cache(maxsize=4)
def _dataset_index(path: str) -> dict[str, DraftHistory]:
    """Title -> history for the collected dataset. Cached; it is read once."""
    from first_reader.wiki.collect import load_dataset

    file = Path(path)
    if not file.exists():
        return {}
    return {h.title: h for h in load_dataset(file)}


def dataset_titles(path: Path | str | None = None) -> list[str]:
    """Every draft title in the local dataset."""
    return sorted(_dataset_index(str(path or default_dataset())))


def _run_sync(factory: Callable[[], Awaitable[T]]) -> T:
    """Run a coroutine from sync code, including from inside a running loop.

    Tools here are synchronous and ``first_reader.wiki`` is async. Calling
    ``asyncio.run`` from inside the SDK's event loop would raise, so the
    coroutine gets its own loop on its own thread.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(factory())).result()


def _fetch_from_wikipedia(title: str) -> DraftHistory:
    """One draft's history from the live API.

    Bound to the real signature -- ``fetch_draft_history(client, title, ...)``
    returning ``(DraftHistory | None, ParseStats)``.
    """
    from first_reader.wiki import WikiClient, fetch_draft_history

    async def go() -> DraftHistory | None:
        async with WikiClient() as client:
            history, _stats = await fetch_draft_history(client, title)
            return history

    history = _run_sync(go)
    if history is None:
        raise HistoryUnavailable(f"{title!r} does not exist on Wikipedia")
    return history


def fetch_history(
    title: str,
    *,
    source: Source | str | None = None,
    dataset: Path | str | None = None,
) -> DraftHistory:
    """Full observable history of one draft.

    Raises:
        HistoryUnavailable: When the draft cannot be found. Never falls back to
            fixture data: an agent that quietly answers about the wrong draft is
            the failure this whole project is supposed to be the opposite of.
    """
    chosen = current_source() if source is None else _SOURCE_ALIASES.get(str(source), source)
    dataset = Path(dataset) if dataset is not None else default_dataset()

    if chosen is Source.STUB:
        from . import _stubs

        return _stubs.fetch_draft_history(title)

    if chosen in (Source.AUTO, Source.DATASET):
        found = _dataset_index(str(dataset)).get(title)
        if found is not None:
            return found
        if chosen is Source.DATASET:
            raise HistoryUnavailable(
                f"{title!r} is not in {dataset}. Collect it with "
                f"`python -m first_reader.wiki.collect`, or use "
                f"FIRST_READER_SOURCE=wikipedia to fetch it live."
            )

    try:
        return _fetch_from_wikipedia(title)
    except HistoryUnavailable:
        raise
    except Exception as exc:  # network down, API error, parse failure
        raise HistoryUnavailable(
            f"{title!r} is not in {dataset} and the live API call failed: {exc}"
        ) from exc


def compute_blockers(
    history: DraftHistory,
    attempts: Mapping[str, AttemptSignal] | None = None,
) -> list[Blocker]:
    """Reasons still standing, ordered by what is blocking the draft now.

    Delegates to ``first_reader.analysis.build_blockers``, bound by import
    rather than discovered by name. A same-named function with a different
    signature binds cleanly and fails several tool calls later, which is how
    the first version of this module broke.

    ``attempts`` maps a decline code to a measured ``AttemptSignal``. A code
    with no entry gets ``attempt=None``, which means *not measured* -- never
    "no attempt was made". ``escalation`` keeps that distinction.
    """
    from first_reader.analysis import build_blockers

    return build_blockers(history, attempts)


def measure_attempts(
    history: DraftHistory,
    *,
    codes: Sequence[str] | None = None,
) -> dict[str, AttemptSignal]:
    """What the contributor did between the previous decline and the last submission.

    The window is deliberately not "since the most recent decline". The
    interesting question on a card is what the contributor did in the round that
    has just been reviewed -- the work the standing decline was made against.
    So it runs from the *previous* decline to the submission the latest decline
    responded to.

    Needs wikitext, which ``DraftHistory`` does not carry, so this reaches the
    API. Revision content is cached under ``data/cache``. Returns ``{}`` when
    the history is too short to have a window, or when the fetch fails: an
    unmeasured attempt is reported as unmeasured, never as an absent one.
    """
    from first_reader.analysis import detect_attempt

    declines = sorted(history.declines, key=lambda d: (d.declined_at, d.submitted_at))
    revisions = sorted(history.revisions, key=lambda r: (r.timestamp, r.revid))
    if not declines or not revisions:
        return {}

    latest = declines[-1]
    before_ts = declines[-2].declined_at if len(declines) >= 2 else revisions[0].timestamp
    after_ts = latest.submitted_at

    before_rev = max(
        (r for r in revisions if r.timestamp <= before_ts),
        key=lambda r: r.timestamp,
        default=revisions[0],
    )
    after_rev = max(
        (r for r in revisions if r.timestamp <= after_ts),
        key=lambda r: r.timestamp,
        default=revisions[-1],
    )
    if before_rev.revid == after_rev.revid:
        return {}

    between = [r for r in revisions if before_rev.timestamp < r.timestamp <= after_rev.timestamp]

    from first_reader.wiki import WikiClient

    async def go() -> dict[int, str]:
        async with WikiClient() as client:
            return await client.get_revision_content([before_rev.revid, after_rev.revid])

    try:
        texts = _run_sync(go)
    except Exception as exc:
        logger.warning("%s: could not fetch revision content (%s)", history.title, exc)
        return {}

    missing = [r.revid for r in (before_rev, after_rev) if r.revid not in texts]
    if missing:
        # Deleted or suppressed revisions come back absent rather than empty.
        # Reporting the attempt as unmeasured sends the draft to a human, which
        # is the right answer when we cannot see what the contributor did.
        logger.warning(
            "%s: revision content unavailable for %s; attempt left unmeasured",
            history.title,
            ", ".join(str(r) for r in missing),
        )
        return {}

    before_text, after_text = texts[before_rev.revid], texts[after_rev.revid]

    wanted = list(codes) if codes is not None else sorted(
        {c for d in declines for c in (d.code, d.secondary_code) if c}
    )
    return {
        code: detect_attempt(before_text, after_text, between, code) for code in wanted
    }


# --- comment preparation ----------------------------------------------------

_FAMILY_SENTENCE = {
    ReasonFamily.SOURCE_EXISTENCE: (
        "Rewriting cannot clear this reason. It depends on whether suitable sources exist, "
        "not on how the draft is written."
    ),
    ReasonFamily.WRITING: "This reason is about how the draft is written, so editing it is the right response.",
    ReasonFamily.STRUCTURAL: "This reason calls for a different action than editing this draft.",
    ReasonFamily.UNKNOWN: (
        "This decline code is not in our map, so we are not saying whether rewriting would help."
    ),
}


def _attempt_sentence(blocker: Blocker) -> str:
    attempt = blocker.attempt
    if attempt is None:
        return "We could not observe any edit between the last decline and this resubmission."

    parts: list[str] = []
    if attempt.added_sources:
        domains = ", ".join(attempt.new_domains) if attempt.new_domains else "no new domains"
        plural = "" if attempt.added_sources == 1 else "s"
        parts.append(f"{attempt.added_sources} reference{plural} added ({domains})")
    else:
        parts.append("no references added")

    parts.append(f"{attempt.bytes_changed:+,} bytes")
    parts.append(f"about {round((1 - attempt.similarity) * 100)}% of the text rewritten")
    parts.append(
        "the edit touched the part the reason pointed at"
        if attempt.touched_target_area
        else "the edit did not touch the part the reason pointed at"
    )
    return "Since the last decline: " + "; ".join(parts) + "."


def compose_comment(
    card: ReviewCard,
    *,
    interpretation: Interpretation | None = None,
    interpreter: Interpreter | None = None,
) -> str:
    """Draft text a reviewer may post under their own account.

    Written by the model when one is available, from facts that are handed over
    already settled -- every code, family, count and percentage below was
    decided deterministically and the model may not change one. It chooses
    words. This note often has to tell someone that the month they spent
    rewriting could not have worked, and :func:`template_comment` says that
    accurately but badly.

    Falls back to the template whenever the model is unavailable, fails, or
    returns something that does not survive :func:`_comment_is_safe`.

    Carries forward what reviewers already decided and reports what changed. It
    contains no assessment of the draft, because assessing the draft is the
    review, and First Reader does not review.

    This is the ``prepare_comment(card)`` entry point.
    """
    # No interpreter, no model call. Bulk offline passes and every unit test
    # go through here, and none of them should reach a paid endpoint by default.
    if interpreter is None or not interpreter.available or card.top_blocker is None:
        return template_comment(card, interpretation=interpretation)
    return _run_sync(
        lambda: compose_comment_async(
            card, interpretation=interpretation, interpreter=interpreter
        )
    )


async def compose_comment_async(
    card: ReviewCard,
    *,
    interpretation: Interpretation | None = None,
    interpreter: Interpreter | None = None,
) -> str:
    """The note, written by the model on the caller's loop. See :func:`compose_comment`."""
    reader = interpreter
    if reader is not None and reader.available and card.top_blocker is not None:
        written = await reader.write_comment_async(_comment_facts(card, interpretation))
        if written and _comment_is_safe(written, card):
            return _with_footer(written, interpretation)

    return template_comment(card, interpretation=interpretation)


def _with_footer(body: str, interpretation: Interpretation | None) -> str:
    """Attach the estimate disclosure and the provenance line, in that order."""
    parts = [body.strip()]
    estimates = _estimate_block(interpretation)
    if estimates:
        parts.append(estimates)
    parts.append(PREPARED_BY)
    return "\n\n".join(parts)


def _estimate_block(interpretation: Interpretation | None) -> str:
    """Say, in fixed words, that a reading was estimated and what it was read from.

    Written here rather than by the model on purpose. The disclosure that a
    machine guessed at something is the last thing that should depend on the
    machine remembering to mention it, so it is assembled deterministically and
    is identical every time.
    """
    if not interpretation or not interpretation.families:
        return ""

    lines: list[str] = []
    for reading in interpretation.families.values():
        if reading.family is None and not reading.corrected_by_reviewer:
            continue
        family = {
            "source_existence": "whether suitable sources exist",
            "writing": "how the draft is written",
            "structural": "something other than editing this draft",
        }.get(reading.value, reading.value.replace("_", " "))

        if reading.corrected_by_reviewer:
            lines.append(
                f"A reviewer has recorded that the `{reading.subject}` notice is about "
                f"{family}. That is their reading, not ours."
            )
        else:
            lines.append(
                f"The `{reading.subject}` notice was written as free text, so there is no "
                f"decline code to carry forward. First Reader reads it as being about "
                f"{family} \u2014 an estimate from the reviewer's own words, not a decision, "
                f"and it has changed nothing above. What it read: \u201c{reading.evidence}\u201d"
            )
    return "\n\n".join(lines)


#: Attached to every note, however it was written. A reviewer posting under
#: their own name has to know which part of it a machine produced.
PREPARED_BY = (
    "Prepared by First Reader. It carries prior review decisions forward and reports "
    "what changed; it does not review drafts and did not assess this one. Please check "
    "before posting."
)

#: Phrases that would make the note a review. The model is told not to write
#: these; this is what happens when it does anyway.
#:
#: The second group is the one that caught a real failure. Asked to describe an
#: edit, the model wrote "the new source from beverungen.de does not appear to
#: resolve the existing concerns" -- a verdict on a source, which is the review
#: itself. Whether an edit resolved a reason is precisely what First Reader is
#: built not to say, and saying it in a note a reviewer posts under their own
#: name is the worst place to say it.
_FORBIDDEN = (
    # A verdict on this draft or this subject. Only a reviewer may write these.
    "should be accepted",
    "should be declined",
    "i have accepted",
    "i have declined",
    "is notable",
    "is not notable",
    "not notable enough",
    "in my opinion",
    # A verdict on a source. Naming a domain and grading it is the review.
    "source is reliable",
    "source is unreliable",
    "sources are reliable",
    "sources are unreliable",
    "unreliable source",
    "not a reliable",
    "not independent",
    # An assertion that the standing reason is or is not resolved. Also the review.
    "does not resolve",
    "doesn't resolve",
    "did not resolve",
    "fails to resolve",
    "remains unaddressed",
    "remains unresolved",
    "still unresolved",
    "core issue remains",
)
"""Phrases a note may never contain.

Deliberately narrower than an earlier attempt, which also banned "did not
address" and "cannot be accepted". Both appear in the hand-written note this one
is measured against -- "if sources like that do not exist yet, the draft cannot
be accepted no matter how well it is written" is a statement about what the
guideline requires, not a finding about this draft -- and banning them sent
every generated note to the template. A check that fires on everything protects
nothing.

What is left is the set that cannot be innocent: a verdict on the draft, a
verdict on a source, or an assertion that the standing reason is or is not
resolved. Everything else rests on the last line of every note, which is that a
reviewer reads it before it goes out under their name.
"""


def _comment_is_safe(text: str, card: ReviewCard) -> bool:
    """Whether a written note stayed inside what First Reader is allowed to say.

    Two ways to fail. Saying something only a reviewer may say -- a verdict on
    the draft, the subject or a source. And naming a decline code that is not
    standing on this draft, which would be inventing a reason.
    """
    lowered = text.lower()
    for phrase in _FORBIDDEN:
        if phrase in lowered:
            logger.warning("written note rejected: it contains %r", phrase)
            return False

    standing = {b.code for b in card.blockers}
    for match in re.findall(r"`([a-z][a-z0-9-]{1,12})`", text):
        if match not in standing:
            logger.warning("written note rejected: it names `%s`, which is not standing", match)
            return False
    return True


def _comment_facts(card: ReviewCard, interpretation: Interpretation | None) -> dict[str, Any]:
    """Everything the model is allowed to use, already decided."""
    draft = card.draft
    top = card.top_blocker
    assert top is not None
    attempt = top.attempt

    facts: dict[str, Any] = {
        "draft title": draft.title,
        # Counts arrive already worded. Handed the integer 2, the model wrote
        # "this reason has been given twice before", which is one decline more
        # than happened. The model does not get to render a number.
        "how many times it has been declined": _times(draft.decline_count),
        "standing reason code": top.code,
        # The code's meaning comes from the AfC reason map, verbatim. Left to
        # itself the model glossed `corp` as "corporate affiliation" -- a
        # conflict-of-interest problem, which is not what the code means at all.
        # It does not get to say what a decline code means.
        "what that code means, exactly": _code_meaning(top.code),
        "what that reason is about": _FAMILY_SENTENCE[top.family],
        "how many times this reason has been given": _times(top.repeat_count)
        + " in total, counting the current decline",
        "first given on": top.first_seen.date().isoformat(),
        "can rewriting fix it": {True: "yes", False: "no", None: "unknown"}[
            top.fixable_by_rewrite
        ],
        "other reasons still standing": [b.code for b in card.blockers[1:]] or None,
    }

    if attempt is None:
        facts["what changed since the last decline"] = "not measured -- say nothing about it"
    else:
        facts["references cited before and after"] = (
            f"{attempt.refs_before} before, {attempt.refs_after} after"
        )
        facts["source domains cited that were not there before"] = (
            list(attempt.new_domains) or "none"
        )
        facts["size change"] = f"{attempt.bytes_changed:+,} bytes"
        facts["how much of the text is new"] = (
            f"about {round((1 - attempt.similarity) * 100)} percent"
        )
        facts["did the edit touch the part the reason pointed at"] = (
            "yes" if attempt.touched_target_area else "no"
        )
        facts["contributor's own edit summaries"] = list(attempt.editor_comments) or None

    # Stated either way. Told nothing, the model filled the silence itself --
    # it advised a contributor to "provide a clear explanation of how the draft
    # meets the relevant guidelines" for a reason whose meaning nobody had
    # established. An explicit "unknown" is an instruction; an absent key is not.
    facts["what kind of thing would move this reason"] = _WHAT_MOVES_IT.get(
        top.family,
        "unknown -- nobody has established what this reason is about, so write "
        "nothing at all about what would help",
    )

    if card.wasted_rewrite:
        facts["IMPORTANT"] = (
            "The contributor rewrote this substantively against a reason that rewriting "
            "cannot fix. Tell them so plainly and early."
        )

    if interpretation:
        for reading in interpretation.families.values():
            if reading.family is None and not reading.corrected_by_reviewer:
                continue
            label = "a reviewer says" if reading.corrected_by_reviewer else "ESTIMATE ONLY"
            facts[f"the free-text `{reading.subject}` notice ({label})"] = (
                f"{reading.value.replace('_', ' ')} -- {reading.reading}. "
                f"Mention this only as something First Reader read, never as a finding."
            )
    return facts


#: What would move each family, in fixed words. The model is told to say this
#: only when it is given, so an UNKNOWN family gets no advice at all -- an
#: earlier version invented "sources that are more widely recognized" for a
#: reason nobody had established the meaning of.
_WHAT_MOVES_IT = {
    ReasonFamily.SOURCE_EXISTENCE: (
        "sources that already exist in the world: independent of the subject, "
        "published somewhere with editorial oversight, and covering the subject in "
        "some depth rather than in passing. If sources like that do not exist yet, "
        "no amount of rewriting reaches this."
    ),
    ReasonFamily.WRITING: (
        "an edit to the prose the reason pointed at, made against the sources "
        "already cited."
    ),
    ReasonFamily.STRUCTURAL: (
        "an action somewhere other than this draft -- the reviewer's note says which."
    ),
}


def _code_meaning(code: str) -> str:
    """The AfC reason map's own description of a code, or a plain admission."""
    from first_reader.analysis import describe

    info = describe(code)
    if info is None or not getattr(info, "description", ""):
        return "not in the AfC reason map; do not say what it means"
    return info.description


_NUMBER_WORDS = {
    1: "once",
    2: "twice",
    3: "three times",
    4: "four times",
    5: "five times",
    6: "six times",
}


def _times(count: int) -> str:
    """A count as words, so the model never has to render one itself."""
    return _NUMBER_WORDS.get(count, f"{count} times")


def template_comment(card: ReviewCard, *, interpretation: Interpretation | None = None) -> str:
    """The note, assembled from a template. The fallback, and the reference.

    Says everything accurately and says it woodenly. Used whenever no model is
    available, and kept as the thing a written note is checked against.
    """
    draft = card.draft
    top = card.top_blocker
    if top is None:
        return _with_footer(
            f"{draft.title} has no decline standing on it. Nothing to carry forward.",
            interpretation,
        )

    lines = [
        f"Carried forward for {draft.title}.",
        "",
        (
            f"Declined {_times(draft.decline_count)}. The reason standing on this resubmission "
            f"is `{top.code}`, first given on {top.first_seen.date().isoformat()} and raised "
            f"{_times(top.repeat_count)} by "
            f"{', '.join(dict.fromkeys(top.carried_from)) or 'a reviewer'}."
        ),
        "",
        _FAMILY_SENTENCE[top.family],
        "",
        _attempt_sentence(top),
    ]

    if card.wasted_rewrite:
        lines += [
            "",
            (
                "Worth flagging: the contributor rewrote substantively against a reason that "
                "rewriting cannot fix. Whatever the outcome, telling them that plainly saves them "
                "the next round."
            ),
        ]

    others = card.blockers[1:]
    if others:
        lines += ["", "Also standing: " + ", ".join(f"`{b.code}`" for b in others) + "."]

    return _with_footer("\n".join(lines), interpretation)


def interpret_card(
    card: ReviewCard,
    interpreter: Interpreter | None = None,
    *,
    known: dict[str, dict[str, Any]] | None = None,
) -> Interpretation:
    """Blocking form of :func:`interpret_card_async`, for scripts and tests."""
    return _run_sync(lambda: interpret_card_async(card, interpreter, known=known))


async def interpret_card_async(
    card: ReviewCard,
    interpreter: Interpreter | None = None,
    *,
    known: dict[str, dict[str, Any]] | None = None,
) -> Interpretation:
    """Read the free-text decline reasons on one card. Estimates only.

    72 of the 1,472 declines in the collected set were written as prose instead
    of a code -- ``{{AfC submission|d|reason|...}}`` -- and the deterministic
    path discards every one of them, because there is no code to map. This puts
    an estimate beside the UNKNOWN, quoting the reviewer's own words.

    Nothing here changes ``Blocker.family``. An UNKNOWN family stays UNKNOWN.

    Args:
        card: The card to read.
        interpreter: Supply one, or nothing is read. There is no implicit model
            call: bulk passes and unit tests go through here too.
        known: Readings a reviewer has already corrected, keyed by decline code.
            Used verbatim; the model is not consulted for them.
    """
    result = Interpretation()
    known = known or {}

    for decline in reversed(card.draft.declines):
        if decline.code in result.families:
            continue
        settled = known.get(decline.code)
        if settled is not None:
            result.families[decline.code] = Reading(**settled)
            continue
        text = free_text_reason(decline)
        if text is None or interpreter is None or not interpreter.available:
            continue
        reading = await interpreter.read_free_text_reason_async(decline.code, text)
        if reading is not None:
            result.families[decline.code] = reading

    return result


def build_card(
    title: str,
    *,
    source: Source | str | None = None,
    measure: bool = True,
    dataset: Path | str | None = None,
    interpreter: Interpreter | None = None,
    interpretation: Interpretation | None = None,
) -> ReviewCard:
    """History, blockers and comment for one draft, with escalation not yet decided.

    ``needs_human`` and ``escalation_reason`` are filled in by
    ``escalation.judge`` and the policy store, in ``core``.

    Args:
        title: Full draft title.
        source: Override the configured history source.
        measure: Fetch wikitext and measure what the contributor did. Turning
            this off leaves every blocker's attempt unmeasured, which
            ``escalation`` treats as a reason to involve a human rather than as
            evidence that nothing was done. Off is for bulk offline passes.
        dataset: The local dataset to look in first.
        interpreter: When given, the note is written by the model instead of
            assembled from the template. Omitted everywhere that runs in bulk.
        interpretation: Readings to hand the writer, from ``interpret_card``.
    """
    history = fetch_history(title, source=source, dataset=dataset)
    chosen = current_source() if source is None else _SOURCE_ALIASES.get(str(source), source)
    attempts = measure_attempts(history) if (measure and chosen is not Source.STUB) else None
    if chosen is Source.STUB:
        from . import _stubs

        attempts = _stubs.attempts_for(history)

    blockers = compute_blockers(history, attempts)
    card = ReviewCard(
        draft=history,
        blockers=blockers,
        draft_comment="",
        needs_human=False,
        escalation_reason="not yet judged",
    )
    card.draft_comment = compose_comment(
        card, interpretation=interpretation, interpreter=interpreter
    )
    return card


# --- the reviewer queue -----------------------------------------------------


def _slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_") or "untitled"


def queue_card(
    card: ReviewCard,
    queue_dir: Path | str = DEFAULT_QUEUE_DIR,
    *,
    interpretation: Any | None = None,
) -> Path:
    """Write a card to the reviewer queue and return its path.

    The queue is a directory of JSON files on purpose. A reviewer can open one,
    a second process can read it, and nothing has to stay running between the
    agent finishing and a human looking.
    """
    directory = Path(queue_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slug(card.draft.title)}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(_serde.card_to_dict(card, interpretation), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


# --- agent state carrier ----------------------------------------------------


def _agent_interpreter(agent: Any) -> Interpreter:
    """Read prose with the agent's own model.

    Not with a freshly resolved one. ``resolve_interpreter`` builds a provider,
    and a provider builds an HTTP client; calling it per tool leaked one client
    per call, whose response generators were then torn down outside any loop --
    ``RuntimeError: generator didn't stop after athrow()`` at the end of an
    otherwise clean run. One agent, one model, one client.
    """
    cached = getattr(agent, "_first_reader_interpreter", None)
    if cached is None:
        # Only reached when an Agent was assembled without build_agent.
        cached = Interpreter(agent.model)
        agent._first_reader_interpreter = cached
    return cached


def _state_get(agent: Any, key: str, title: str) -> dict[str, Any] | None:
    bucket = agent.state.get(key) or {}
    return bucket.get(title)


def _state_put(agent: Any, key: str, title: str, value: Any) -> None:
    bucket = agent.state.get(key) or {}
    bucket[title] = value
    agent.state.set(key, bucket)


def stored_card(agent: Any, title: str) -> ReviewCard | None:
    """Read back a card the agent prepared during this run."""
    raw = _state_get(agent, CARD_KEY, title)
    return _serde.card_from_dict(raw) if raw else None


def stored_judgment(agent: Any, title: str) -> dict[str, Any] | None:
    """Read back the escalation verdict computed during this run."""
    return _state_get(agent, JUDGMENT_KEY, title)


def stored_interpretation(agent: Any, title: str) -> Interpretation:
    """Read back the readings made during this run."""
    return Interpretation.from_dict(_state_get(agent, INTERPRETATION_KEY, title))


def queued_path(agent: Any, title: str) -> str | None:
    """Where this run filed the card, if it filed one."""
    return _state_get(agent, QUEUED_KEY, title)


def audit_log(agent: Any) -> list[dict[str, Any]]:
    """Every graded tool call in this run, in order."""
    return agent.state.get(AUDIT_KEY) or []


# --- tools ------------------------------------------------------------------


@tool(context=True)
def get_draft_history(title: str, tool_context: ToolContext) -> dict[str, Any]:
    """Load the decline and revision history of one AfC draft.

    Args:
        title: Full draft title, for example "Draft:Corner Bakery Collective".

    Returns:
        A summary of the history. The full record is kept for later tools.
    """
    history = fetch_history(title)
    agent = tool_context.agent
    _state_put(agent, HISTORY_KEY, title, _serde.history_to_dict(history))

    latest = history.latest_decline
    return {
        "title": history.title,
        "pageid": history.pageid,
        "decline_count": history.decline_count,
        "is_pending": history.is_pending,
        "revision_count": len(history.revisions),
        "latest_decline": (
            {
                "code": latest.code,
                "family": latest.family.value,
                "decliner": latest.decliner,
                "declined_at": latest.declined_at.date().isoformat(),
                "fixable_by_rewrite": latest.fixable_by_rewrite,
            }
            if latest
            else None
        ),
        "distinct_decliners": sorted({d.decliner for d in history.declines}),
        "distinct_codes": sorted({d.code for d in history.declines}),
    }


@tool(context=True)
def analyze_blockers(title: str, tool_context: ToolContext) -> dict[str, Any]:
    """Work out which decline reasons are still standing, and what was attempted.

    Call get_draft_history first. Reports evidence only; it does not decide
    whether any reason was resolved. An attempt of null means it could not be
    measured -- never that the contributor did nothing.

    Args:
        title: The draft title passed to get_draft_history.

    Returns:
        The standing reasons, most important first, with the observed attempt
        against each.
    """
    agent = tool_context.agent
    card = build_card(title)
    _state_put(agent, CARD_KEY, title, _serde.card_to_dict(card))
    blockers = card.blockers
    _state_put(agent, BLOCKERS_KEY, title, [_serde.blocker_to_dict(b) for b in blockers])

    return {
        "title": title,
        "blockers": [
            {
                "code": b.code,
                "family": b.family.value,
                "repeat_count": b.repeat_count,
                "fixable_by_rewrite": b.fixable_by_rewrite,
                "carried_from": b.carried_from,
                "attempt": (
                    None
                    if b.attempt is None
                    else {
                        "added_sources": b.attempt.added_sources,
                        "new_domains": list(b.attempt.new_domains),
                        "bytes_changed": b.attempt.bytes_changed,
                        "substantive_rewrite": b.attempt.substantive_rewrite,
                        "touched_target_area": b.attempt.touched_target_area,
                    }
                ),
            }
            for b in blockers
        ],
    }


@tool(context=True)
async def interpret_reasons(
    title: str, tool_context: ToolContext, known_readings: str = ""
) -> dict[str, Any]:
    """Read the prose on this draft that the decline codes do not capture.

    Some reviewers write the reason as free text instead of picking a code, and
    the code map has nothing to say about those. This reads them.

    Everything this returns is an estimate and is labelled as one. It never
    changes a reason family, never invents a decline code, and never says
    whether the draft is acceptable. A reviewer confirms or corrects it.

    Args:
        title: The draft title passed to the earlier tools.
        known_readings: Readings a reviewer has already corrected, as JSON.
            Do not set this yourself. The intervention layer fills it in when a
            reviewer has settled something, so their answer is used instead of
            a fresh guess.

    Returns:
        The readings, each with the verbatim words it came from.
    """
    agent = tool_context.agent
    raw = _state_get(agent, CARD_KEY, title)
    card = _serde.card_from_dict(raw) if raw else build_card(title)

    known: dict[str, dict[str, Any]] = {}
    if known_readings:
        try:
            known = json.loads(known_readings)
        except json.JSONDecodeError:
            logger.warning("%s: ignoring malformed known_readings", title)

    interpretation = await interpret_card_async(card, _agent_interpreter(agent), known=known)
    _state_put(agent, INTERPRETATION_KEY, title, interpretation.to_dict())

    return {
        "title": title,
        "estimated": True,
        "note": (
            "Estimates about prose a reviewer wrote, not decisions. None of these changes "
            "a reason family; an unknown code stays unknown. A reviewer confirms or corrects them."
        ),
        "free_text_reasons": [
            {
                "code": r.subject,
                "reads_as": r.value,
                "meaning": r.reading,
                "evidence": r.evidence,
                "from_a_reviewer_correction": r.corrected_by_reviewer,
            }
            for r in interpretation.families.values()
        ],
    }


@tool(context=True)
async def prepare_comment(title: str, tool_context: ToolContext) -> dict[str, Any]:
    """Draft the comment a reviewer could post under their own account.

    Call analyze_blockers first. The text carries prior decisions forward and
    reports what changed. It never says whether the draft should be accepted.

    Args:
        title: The draft title passed to the earlier tools.

    Returns:
        The prepared comment text.
    """
    agent = tool_context.agent
    raw = _state_get(agent, CARD_KEY, title)
    card = _serde.card_from_dict(raw) if raw else build_card(title)

    interpretation = Interpretation.from_dict(_state_get(agent, INTERPRETATION_KEY, title))
    reader = _agent_interpreter(agent)
    card.draft_comment = await compose_comment_async(
        card, interpretation=interpretation, interpreter=reader
    )
    _state_put(agent, CARD_KEY, title, _serde.card_to_dict(card))

    return {
        "title": title,
        "draft_comment": card.draft_comment,
        "wasted_rewrite": card.wasted_rewrite,
        "written_by": "model" if reader.available else "template",
    }


@tool(context=True)
def escalate_to_reviewer(title: str, why: str, tool_context: ToolContext) -> dict[str, Any]:
    """Put a prepared card in the reviewer queue and stop working on this draft.

    Returns immediately. Nothing waits for the reviewer: the card sits in a
    durable queue and a human opens it whenever they open it.

    Args:
        title: The draft title passed to the earlier tools.
        why: One plain sentence saying what a reviewer would see that the agent
            could not settle. This is shown to them verbatim.

    Returns:
        Where the card was filed.
    """
    agent = tool_context.agent
    raw = _state_get(agent, CARD_KEY, title)
    card = _serde.card_from_dict(raw) if raw else build_card(title)
    card.needs_human = True
    if why:
        card.escalation_reason = why

    interpretation = Interpretation.from_dict(_state_get(agent, INTERPRETATION_KEY, title))
    queue_dir = agent.state.get(QUEUE_DIR_KEY) or str(DEFAULT_QUEUE_DIR)
    path = queue_card(card, queue_dir, interpretation=interpretation)
    _state_put(agent, CARD_KEY, title, _serde.card_to_dict(card, interpretation))
    _state_put(agent, QUEUED_KEY, title, str(path))

    return {"title": title, "queued_at": str(path), "needs_human": True}


@tool(context=True)
def post_reviewer_decision(title: str, verdict: str, tool_context: ToolContext) -> dict[str, Any]:
    """Reserved. Records a review verdict on a draft.

    First Reader does not review drafts, so every call to this is refused by the
    reversibility gate before it runs. It exists so that the refusal is a
    property of the harness rather than of a system prompt: an agent that is
    only asked not to review can still review.

    Args:
        title: The draft title.
        verdict: The verdict that would be recorded.

    Returns:
        Never returns; the call is denied first.
    """
    raise RuntimeError(
        "post_reviewer_decision ran. The reversibility gate should have denied it. "
        "Check that ReversibilityGate is registered in Agent(interventions=...)."
    )


def make_judge_escalation(
    policy: PolicyStore,
    weights: Weights = DEFAULT_WEIGHTS,
) -> Any:
    """Build the tool that runs the escalation judgment against a policy store.

    The judgment is arithmetic over evidence plus whatever rules a reviewer has
    let the system learn. It is a tool only so that the transcript shows the
    agent consulting it; the agent does not get to decide the answer, and
    ``core.EscalationPolicyGate`` refuses any escalation this verdict did not
    authorise.
    """

    @tool(name="judge_escalation", context=True)
    def judge_escalation(title: str, tool_context: ToolContext) -> dict[str, Any]:
        """Decide whether this draft goes in front of a reviewer.

        Call prepare_comment first. The answer is binding: escalate_to_reviewer
        is refused unless this says to escalate.

        Args:
            title: The draft title passed to the earlier tools.

        Returns:
            The verdict, the evidence state behind it, and the arithmetic.
        """
        agent = tool_context.agent
        raw = _state_get(agent, CARD_KEY, title)
        card = _serde.card_from_dict(raw) if raw else build_card(title)

        interpretation = Interpretation.from_dict(_state_get(agent, INTERPRETATION_KEY, title))

        key = pattern_key(card)
        verdict = judge(card, precedent_seen=policy.precedent_seen(key), weights=weights)
        sig = signature(card, verdict.state)

        rule = policy.active_rule_for(sig)
        if rule is not None:
            verdict = apply_rule(
                verdict,
                rule_id=rule.id,
                rule_sentence=rule.sentence,
                forces_escalation=(rule.kind == TIGHTEN),
            )

        policy.observe(key)

        card.needs_human = verdict.escalate
        card.escalation_reason = verdict.reason
        _state_put(agent, CARD_KEY, title, _serde.card_to_dict(card, interpretation))

        payload = {
            "title": title,
            "escalate": verdict.escalate,
            "reason": verdict.reason,
            "state": verdict.state.value,
            "advantage": round(verdict.advantage, 3),
            "cost": round(verdict.cost, 3),
            "recoverability": round(verdict.recoverability, 3),
            "score": round(verdict.score, 3),
            "threshold": verdict.interruption_cost,
            "signature": sig,
            "pattern_key": key,
            "policy_rule_id": verdict.policy_rule_id,
        }
        _state_put(agent, JUDGMENT_KEY, title, payload)
        return payload

    declare(
        judge_escalation,
        Reversibility.REVERSIBLE,
        "Computes a verdict and records that the pattern was seen. Changes no draft.",
    )
    return judge_escalation


declare(
    get_draft_history,
    Reversibility.REVERSIBLE,
    "Reads public revision history. Nothing outside this process changes.",
)
declare(
    analyze_blockers,
    Reversibility.REVERSIBLE,
    "Derives standing reasons from data already fetched. Pure computation.",
)
declare(
    interpret_reasons,
    Reversibility.REVERSIBLE,
    "Reads prose already fetched and records estimates. Changes no draft and no verdict.",
)
declare(
    prepare_comment,
    Reversibility.REVERSIBLE,
    "Composes text into memory. Posting it is a separate human action.",
)
declare(
    escalate_to_reviewer,
    Reversibility.ESCALATING,
    "Writes one JSON file to the reviewer queue. Undone by deleting that file.",
)
declare(
    post_reviewer_decision,
    Reversibility.IRREVERSIBLE,
    "Would record a verdict on someone's draft. There is no undo for that.",
)


#: Every tool except ``judge_escalation``, which is built per policy store by
#: :func:`make_judge_escalation`.
STATIC_TOOLS = [
    get_draft_history,
    analyze_blockers,
    interpret_reasons,
    prepare_comment,
    escalate_to_reviewer,
    post_reviewer_decision,
]
