"""Shared data contract for First Reader.

Every module in this package agrees on these types. Do not change a field
without updating the modules that read it.

Design rule that shapes this file: First Reader never decides whether a draft
is acceptable. It carries forward decisions humans already made, and reports
whether an attempt was made to address them. Nothing here should be able to
express "this draft is good/bad".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ReasonFamily(str, Enum):
    """What kind of failure a decline code represents.

    The distinction that matters: whether rewriting the draft can fix it.
    A contributor who receives `ai` and rewrites is doing the right thing.
    A contributor who receives `nn` and rewrites is wasting their time --
    notability depends on sources existing in the world, not on the prose.
    """

    SOURCE_EXISTENCE = "source_existence"  # not fixable by rewriting
    WRITING = "writing"  # fixable by rewriting
    STRUCTURAL = "structural"  # needs a different action entirely
    UNKNOWN = "unknown"  # code not in our map; never guess a family


# Complete AfC decline-code map, extracted 2026-09-07 from AFCH
# (wikimedia-gadgets/afc-helper, src/templates/tpl-submissions.html) and from
# Template:AfC submission/comments. Codes absent from this map resolve to
# UNKNOWN, never to a guess -- that includes the seven decline codes we
# deliberately refused to classify (cv, cv-cleaned, blp, dict, not, reason, afd)
# and AFCH's reject codes (n, j, e).
#
# Aliases are included as separate keys because both spellings occur in live
# wikitext: AFCH writes `adv`/`dup`/`v`/`ns`/`lang`, while
# Template:AfC submission/doc templatedata documents `advert`/`duplicate`/
# `source`/`nosource`/`notenglish`. In 1,472 real declines the AFCH spellings
# account for every occurrence and the templatedata spellings for none.
#
# The reasoning behind each row, with the verbatim AFCH text it rests on, lives
# in analysis/reasons.py. That module fails on import if the two disagree.
REASON_FAMILY: dict[str, ReasonFamily] = {
    # source existence -- rewriting cannot fix these
    "nn": ReasonFamily.SOURCE_EXISTENCE,
    "v": ReasonFamily.SOURCE_EXISTENCE,
    "rs": ReasonFamily.SOURCE_EXISTENCE,
    "source": ReasonFamily.SOURCE_EXISTENCE,
    "ns": ReasonFamily.SOURCE_EXISTENCE,
    "nosource": ReasonFamily.SOURCE_EXISTENCE,
    "med": ReasonFamily.SOURCE_EXISTENCE,
    "bio": ReasonFamily.SOURCE_EXISTENCE,
    "corp": ReasonFamily.SOURCE_EXISTENCE,
    "org": ReasonFamily.SOURCE_EXISTENCE,
    "inc": ReasonFamily.SOURCE_EXISTENCE,
    "event": ReasonFamily.SOURCE_EXISTENCE,
    "music": ReasonFamily.SOURCE_EXISTENCE,
    "m": ReasonFamily.SOURCE_EXISTENCE,
    "band": ReasonFamily.SOURCE_EXISTENCE,
    "athlete": ReasonFamily.SOURCE_EXISTENCE,
    "sport": ReasonFamily.SOURCE_EXISTENCE,
    "film": ReasonFamily.SOURCE_EXISTENCE,
    "book": ReasonFamily.SOURCE_EXISTENCE,
    "web": ReasonFamily.SOURCE_EXISTENCE,
    "astro": ReasonFamily.SOURCE_EXISTENCE,
    "creative": ReasonFamily.SOURCE_EXISTENCE,
    "geo": ReasonFamily.SOURCE_EXISTENCE,
    "list": ReasonFamily.SOURCE_EXISTENCE,
    "neo": ReasonFamily.SOURCE_EXISTENCE,
    "number": ReasonFamily.SOURCE_EXISTENCE,
    "prof": ReasonFamily.SOURCE_EXISTENCE,
    "academic": ReasonFamily.SOURCE_EXISTENCE,
    "school": ReasonFamily.SOURCE_EXISTENCE,
    "species": ReasonFamily.SOURCE_EXISTENCE,
    "plot": ReasonFamily.SOURCE_EXISTENCE,
    "news": ReasonFamily.SOURCE_EXISTENCE,
    "oneevent": ReasonFamily.SOURCE_EXISTENCE,
    # writing -- rewriting can fix these
    "ai": ReasonFamily.WRITING,
    "llm": ReasonFamily.WRITING,
    "adv": ReasonFamily.WRITING,
    "spam": ReasonFamily.WRITING,
    "advert": ReasonFamily.WRITING,
    "npov": ReasonFamily.WRITING,
    "essay": ReasonFamily.WRITING,
    "resume": ReasonFamily.WRITING,
    "context": ReasonFamily.WRITING,
    "lang": ReasonFamily.WRITING,
    "notenglish": ReasonFamily.WRITING,
    "ilc": ReasonFamily.WRITING,
    "cite": ReasonFamily.WRITING,
    "footnote": ReasonFamily.WRITING,
    # structural -- needs something other than editing this draft
    "exists": ReasonFamily.STRUCTURAL,
    "dup": ReasonFamily.STRUCTURAL,
    "duplicate": ReasonFamily.STRUCTURAL,
    "mergeto": ReasonFamily.STRUCTURAL,
    "merge": ReasonFamily.STRUCTURAL,
    "blank": ReasonFamily.STRUCTURAL,
    "empty": ReasonFamily.STRUCTURAL,
    "test": ReasonFamily.STRUCTURAL,
    "redirect": ReasonFamily.STRUCTURAL,
    "cat": ReasonFamily.STRUCTURAL,
    "ecr": ReasonFamily.STRUCTURAL,
    "van": ReasonFamily.STRUCTURAL,
    "joke": ReasonFamily.STRUCTURAL,
    "hoax": ReasonFamily.STRUCTURAL,
}


@dataclass(frozen=True)
class Decline:
    """One `{{AfC submission|d|...}}` template instance.

    `submitted_at` comes from `ts`, `declined_at` from `declinets`. Their
    difference includes queue wait, so it is NOT review effort. Do not use it
    as a proxy for how long a reviewer spent.
    """

    code: str
    secondary_code: str | None
    decliner: str
    submitted_at: datetime
    declined_at: datetime
    raw: str

    @property
    def family(self) -> ReasonFamily:
        return REASON_FAMILY.get(self.code, ReasonFamily.UNKNOWN)

    @property
    def fixable_by_rewrite(self) -> bool | None:
        """None when the family is unknown. Never guess."""
        fam = self.family
        if fam is ReasonFamily.UNKNOWN:
            return None
        return fam is ReasonFamily.WRITING


@dataclass(frozen=True)
class Revision:
    revid: int
    parentid: int
    user: str
    timestamp: datetime
    comment: str
    size: int


@dataclass
class DraftHistory:
    """Full observable history of one draft."""

    title: str
    pageid: int
    declines: list[Decline] = field(default_factory=list)  # oldest first
    revisions: list[Revision] = field(default_factory=list)  # oldest first
    is_pending: bool = False

    @property
    def decline_count(self) -> int:
        return len(self.declines)

    @property
    def latest_decline(self) -> Decline | None:
        return self.declines[-1] if self.declines else None


@dataclass(frozen=True)
class AttemptSignal:
    """Evidence that the contributor tried to address a decline reason.

    This is deliberately NOT a judgment of whether the reason was resolved.
    Deciding whether a newly added source is reliable, independent and
    substantial IS the review itself; doing that here would break the safety
    argument that First Reader only carries human decisions forward.

    Every field must be mechanically derivable from diffs and wikitext.
    """

    refs_before: int
    refs_after: int
    new_domains: tuple[str, ...]
    bytes_changed: int
    similarity: float  # 0.0 = fully rewritten, 1.0 = unchanged
    touched_target_area: bool  # did the edit touch what the reason pointed at
    editor_comments: tuple[str, ...]  # edit summaries between the two submissions

    @property
    def added_sources(self) -> int:
        return max(0, self.refs_after - self.refs_before)

    @property
    def substantive_rewrite(self) -> bool:
        return self.similarity < 0.7 and abs(self.bytes_changed) > 500


@dataclass(frozen=True)
class Blocker:
    """One reason that is still standing on the current resubmission."""

    code: str
    family: ReasonFamily
    first_seen: datetime
    repeat_count: int  # how many times this same code has been given
    attempt: AttemptSignal | None
    carried_from: list[str]  # decliners who previously raised it

    @property
    def fixable_by_rewrite(self) -> bool | None:
        fam = self.family
        if fam is ReasonFamily.UNKNOWN:
            return None
        return fam is ReasonFamily.WRITING


@dataclass
class ReviewCard:
    """What a reviewer sees for one draft. The unit of the UI.

    `blockers` is ordered: what is actually standing in the way comes first.
    `draft_comment` is prepared text the reviewer may post under their OWN
    account with one click. First Reader never writes to Wikipedia itself.
    """

    draft: DraftHistory
    blockers: list[Blocker]
    draft_comment: str
    needs_human: bool
    escalation_reason: str  # why this surfaced, in plain words

    @property
    def top_blocker(self) -> Blocker | None:
        return self.blockers[0] if self.blockers else None

    @property
    def wasted_rewrite(self) -> bool:
        """Contributor rewrote substantively against a reason rewriting cannot fix."""
        top = self.top_blocker
        if top is None or top.attempt is None:
            return False
        return top.fixable_by_rewrite is False and top.attempt.substantive_rewrite
