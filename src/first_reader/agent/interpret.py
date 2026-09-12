"""Reading what reviewers wrote, as opposed to deciding anything.

Everything in this module is an *estimate*, and every estimate is carried
beside the fact it interprets, never on top of it. ``Blocker.family`` stays
UNKNOWN when the code is unknown. A reading sits next to it, labelled, for a
reviewer to accept or correct.

That division is the whole design. The deterministic path decides -- reason
family from the code map, ``fixable_by_rewrite``, the escalation inequality.
The model reads prose the deterministic path has no way to read: 72 of the
1,472 declines in the collected set were written as free text instead of a
code, and were being discarded whole.

Two jobs, and only two. Reading free-text decline reasons, and writing the
note a reviewer posts. Reading ``{{AfC comment}}`` blocks, reading the
contributor's edit summaries, wording the learned rules and narrating the
escalation are all deferred; see ``docs/next-steps.md``.

Three rules hold everywhere below.

**No new reason codes.** A reading says which of three existing families some
prose seems to be about. It cannot invent a code and it cannot overwrite one.

**Every reading quotes its source.** A reading with no verbatim evidence from
the reviewer's own words is dropped, because a reviewer cannot check a claim
that does not say where it came from.

**Failure is silence, not error.** No key, no network, a malformed response, a
model that does not do structured output -- every one of these returns None and
the deterministic path continues unchanged. The system has to work with the
model switched off.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

from ..models import Decline, ReasonFamily

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Families a reading may name. ``cannot_tell`` is a first-class answer: prose
#: that does not say what it is about must be allowed to say so.
FAMILY_VALUES = ("source_existence", "writing", "structural", "cannot_tell")

FAMILY_BRIEF = (
    "source_existence -- cannot be fixed by rewriting; it is about whether "
    "suitable published sources exist (notability, verifiability, sourcing depth).\n"
    "writing -- fixable by rewriting; it is about the prose (tone, promotion, "
    "neutrality, machine-generated text, formatting, language).\n"
    "structural -- needs an action other than editing this draft (duplicate of an "
    "existing article, wrong namespace, blank, test page, should be a redirect).\n"
    "cannot_tell -- the text does not say clearly enough to place it."
)


# --- what the model is allowed to return ------------------------------------


class FamilyReading(BaseModel):
    """Which family a free-text decline reason seems to be about."""

    family: Literal["source_existence", "writing", "structural", "cannot_tell"]
    evidence: str = Field(
        description="A short phrase copied verbatim from the reviewer's text that "
        "carries the meaning. Copy exactly; do not paraphrase."
    )
    reading: str = Field(
        description="One plain sentence saying what the reviewer asked the "
        "contributor to do. No opinion on whether the draft is acceptable."
    )


class WrittenComment(BaseModel):
    """The reviewer-facing note, written from facts that were handed over fixed."""

    text: str = Field(
        description="The note itself. Plain prose, no headings, no markdown "
        "emphasis, no signature."
    )


# --- readings, as carried through the system --------------------------------


@dataclass(frozen=True)
class Reading:
    """One interpretation, marked as such, next to the fact it interprets."""

    kind: str
    subject: str
    value: str
    evidence: str
    reading: str
    source_text: str
    model_id: str = ""
    estimated: bool = True
    corrected_by_reviewer: bool = False
    original_value: str | None = None

    @property
    def family(self) -> ReasonFamily | None:
        """The inferred family, or None when the reading declined to say.

        Never assigned to a ``Blocker.family``. Nothing in this package does
        that, and the queued card carries readings under a separate
        ``interpretation`` key so that a reader of the file can tell an estimate
        from a fact without knowing the code.
        """
        if self.value in ("cannot_tell", ""):
            return None
        try:
            return ReasonFamily(self.value)
        except ValueError:
            return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Interpretation:
    """Every reading made about one draft.

    Keyed by decline code. A code absent from ``families`` was not read -- either
    it was not free text, or the reading was dropped for quoting something the
    reviewer did not write.
    """

    families: dict[str, Reading] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"families": {k: v.to_dict() for k, v in self.families.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Interpretation":
        if not data:
            return cls()
        return cls(families={k: Reading(**v) for k, v in (data.get("families") or {}).items()})


# --- deterministic extraction (no model involved) ---------------------------


FREE_TEXT_CODES = frozenset({"reason"})

def free_text_reason(decline: Decline) -> str | None:
    """The reviewer's prose when they wrote a reason instead of picking a code.

    ``{{AfC submission|d|reason|<prose>|...}}`` puts the prose in the third
    positional parameter. Parsed with the same parser that produced the decline
    rather than a second regex, so the two cannot drift apart.
    """
    if decline.code not in FREE_TEXT_CODES:
        return None
    from ..wiki.parser import iter_afc_templates

    templates = iter_afc_templates(decline.raw)
    if not templates:
        return None
    positional = templates[0].positional
    if len(positional) < 3:
        return None
    text = positional[2].strip()
    return text or None


# --- the interpreter --------------------------------------------------------


def _run_sync(factory: Callable[[], Awaitable[T]]) -> T:
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(factory())).result()


#: Per-call output cap. Interpretation is a sentence and a quote; anything
#: longer is a model that has lost the plot, and paying for it is the cost
#: failure mode this project cannot afford.
MAX_OUTPUT_TOKENS = 400


class Interpreter:
    """Reads prose with a model. Returns None whenever it cannot.

    Holds a ``strands.models.Model`` and calls ``structured_output`` on it
    directly. No second agent: an interpreter behind a tool would be the nested
    agent-as-tool shape that harness-sdk #3076 reports as broken, and there is
    nothing here that needs a loop.
    """

    def __init__(self, model: Any | None = None) -> None:
        self._model = model

    @property
    def model(self) -> Any | None:
        return self._model

    @property
    def available(self) -> bool:
        """Whether this interpreter can actually read anything."""
        if self._model is None:
            return False
        return not getattr(self._model, "is_offline_stub", False)

    @property
    def model_id(self) -> str:
        if self._model is None:
            return ""
        config = self._model.get_config()
        if isinstance(config, dict):
            return str(config.get("model_id", ""))
        return str(getattr(config, "model_id", ""))

    async def _ask_async(
        self, schema: type[BaseModel], system_prompt: str, user_text: str
    ) -> Any | None:
        """One structured-output call, on the caller's event loop.

        Async is the primary form. The tools that call this run inside the
        agent's loop, and driving an HTTP client from a second loop on a second
        thread leaves its response generator to be torn down on the wrong loop --
        which surfaced as ``RuntimeError: generator didn't stop after athrow()``
        on an otherwise clean run.
        """
        if not self.available:
            return None

        prompt = [{"role": "user", "content": [{"text": user_text}]}]
        try:
            event: Any = None
            async for chunk in self._model.structured_output(
                schema, prompt, system_prompt=system_prompt
            ):
                event = chunk
        except Exception as exc:  # noqa: BLE001 - any failure is a fallback, not a crash
            logger.warning("interpretation failed (%s): %s", schema.__name__, exc)
            return None

        output = event.get("output") if isinstance(event, dict) else None
        if not isinstance(output, schema):
            logger.warning("interpretation returned no %s", schema.__name__)
            return None
        return output

    def _ask(self, schema: type[BaseModel], system_prompt: str, user_text: str) -> Any | None:
        """Blocking form, for scripts and tests that have no loop of their own."""
        if not self.available:
            return None
        return _run_sync(lambda: self._ask_async(schema, system_prompt, user_text))

    @staticmethod
    def _evidence_holds(evidence: str, source: str) -> bool:
        """Whether the quoted evidence really appears in the source.

        A reading whose quote is not in the text it claims to quote is dropped.
        The check is on normalised whitespace only -- models reflow line breaks,
        which is not the same as making the quote up.
        """
        if not evidence.strip():
            return False
        squash = lambda s: re.sub(r"\s+", " ", s).strip().lower()  # noqa: E731
        return squash(evidence) in squash(source)

    # -- free-text decline reasons --

    def read_free_text_reason(self, code: str, text: str) -> Reading | None:
        """Blocking form of :meth:`read_free_text_reason_async`."""
        return _run_sync(lambda: self.read_free_text_reason_async(code, text))

    async def read_free_text_reason_async(self, code: str, text: str) -> Reading | None:
        """Which family a reviewer's free-text decline reason seems to be about."""
        system_prompt = (
            "You read decline notices that Wikipedia Articles for Creation reviewers "
            "wrote as free text instead of picking a code. Say which family the "
            "reviewer's objection belongs to.\n\n"
            f"{FAMILY_BRIEF}\n\n"
            "You are not reviewing the draft and you have not seen it. You are only "
            "reporting what the reviewer's words are about. If the text is unclear, "
            "answer cannot_tell -- that is a correct answer, not a failure. Copy the "
            "evidence phrase exactly from the text."
        )
        result = await self._ask_async(
            FamilyReading, system_prompt, f"Reviewer's decline notice:\n\n{text}"
        )
        if result is None:
            return None
        if not self._evidence_holds(result.evidence, text):
            logger.warning("dropped a reading whose quote is not in the source text")
            return None
        return Reading(
            kind="reason_family",
            subject=code,
            value=result.family,
            evidence=result.evidence.strip(),
            reading=result.reading.strip(),
            source_text=text,
            model_id=self.model_id,
        )

    # -- the reviewer-facing note --

    def write_comment(self, facts: dict[str, Any]) -> str | None:
        """Blocking form of :meth:`write_comment_async`."""
        return _run_sync(lambda: self.write_comment_async(facts))

    async def write_comment_async(self, facts: dict[str, Any]) -> str | None:
        """Write the note, from facts that are handed over already settled.

        Every number, code and family in ``facts`` was decided deterministically.
        The model chooses words and nothing else. This is the one place tone
        matters: the note frequently has to tell someone that the evenings they
        spent rewriting could not have worked, and a template says that badly.
        """
        system_prompt = (
            "You write a short note that a Wikipedia reviewer will post, under "
            "their own name, on a draft that has been declined more than once.\n\n"
            "EVERY FACT YOU NEED IS GIVEN TO YOU. Use only those. Do not add a "
            "fact, do not change a number, and do not restate the list as a list. "
            "Counts are given to you as words; use those words. Never explain what "
            "a decline code means beyond the meaning you were given for it.\n\n"
            "WHAT YOU ARE NOT. You have not seen the draft and you are not "
            "reviewing it. Never say whether it should be accepted, whether a "
            "source is good or independent, whether the subject is notable, or "
            "whether an edit resolved anything. Deciding those is the review, and "
            "the reviewer posting this note is the one who does it. To say what a "
            "guideline requires, say it conditionally -- 'if sources like that do "
            "not exist yet, then...' -- never as a finding about this draft.\n\n"
            "WHAT THE NOTE DOES, in this order:\n"
            "1. Say what is standing in the way now and whether rewriting can move "
            "it. If it cannot, say so in the first two sentences. A contributor "
            "rewriting against a reason rewriting cannot fix is spending their "
            "evenings on the wrong thing, and every sentence spent warming up "
            "costs them another week.\n"
            "2. Say what changed since the last decline, from the measurements "
            "given. Describe them; do not grade them. If the edit did not touch "
            "the part the reason pointed at, say it did not change that part -- "
            "not that it failed. Name new source domains and say nothing about "
            "their quality.\n"
            "3. Say what kind of thing would move the standing reason ONLY if you "
            "were given a fact headed 'what kind of thing would move this reason'. "
            "Use what it says. If you were not given one, write nothing about what "
            "would help -- you do not know, and guessing sends someone down a "
            "second wrong path.\n\n"
            "TONE. Write to the contributor, plainly. No greeting, no sign-off, no "
            "praise, no apology, no headings, no bold, no bullets. Do not tell "
            "them to 'consider' or 'focus on' anything, and do not close by "
            "assigning them a task. Say what is true and stop. Three short "
            "paragraphs at most, fewer when there is less to say."
        )
        result = await self._ask_async(WrittenComment, system_prompt, _facts_block(facts))
        if result is None:
            return None
        text = result.text.strip()
        return text or None


def _facts_block(facts: dict[str, Any]) -> str:
    """Facts as flat labelled lines, so nothing looks like an instruction."""
    lines = []
    for key, value in facts.items():
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, (list, tuple)):
            value = "; ".join(str(v) for v in value)
        lines.append(f"{key.replace('_', ' ')}: {value}")
    return "\n".join(lines)


def resolve_interpreter(model: Any | None = None) -> Interpreter:
    """Build an interpreter around a model, or an inert one if there is none."""
    if model is not None:
        return Interpreter(model)
    if os.getenv("FIRST_READER_NO_LLM"):
        return Interpreter(None)
    from ._model import resolve_model

    try:
        return Interpreter(resolve_model())
    except Exception as exc:  # noqa: BLE001
        logger.warning("no interpretation model available: %s", exc)
        return Interpreter(None)
