"""Reading prose: what the model is allowed to say, and what a reviewer overrides.

No network and no inference here. A stand-in interpreter returns fixed readings,
so these test the boundary rather than the model: that an estimate never becomes
a fact, that a quote is checked against its source, and that a reviewer's
correction reaches the next run through the intervention layer.
"""

from __future__ import annotations

import json

import pytest

from _agent_support import ScriptedModel, no_network, offline  # noqa: F401

from first_reader.agent import tools as t
from first_reader.agent.core import (
    ReviewerCorrections,
    build_agent,
    record_reviewer_correction,
    run_review,
)
from first_reader.agent.interpret import Interpretation, Interpreter, Reading, free_text_reason
from first_reader.agent.policy import PolicyStore
from first_reader.models import ReasonFamily

FREE_TEXT = "Draft:Ravensworth Signal Box"
CODED = "Draft:Corner Bakery Collective"


class FakeInterpreter(Interpreter):
    """Returns a fixed reading. Records what it was asked, so silence is testable."""

    def __init__(self, family: str = "source_existence", evidence: str | None = None) -> None:
        super().__init__(model=object())
        self.family = family
        self.evidence = evidence
        self.asked: list[str] = []

    @property
    def available(self) -> bool:
        return True

    @property
    def model_id(self) -> str:
        return "fake"

    async def read_free_text_reason_async(self, code: str, text: str) -> Reading | None:
        self.asked.append(code)
        evidence = self.evidence if self.evidence is not None else text[:40]
        if not self._evidence_holds(evidence, text):
            return None
        return Reading(
            kind="reason_family",
            subject=code,
            value=self.family,
            evidence=evidence,
            reading="the reviewer asked for sources from outside the society",
            source_text=text,
            model_id="fake",
        )


@pytest.fixture
def store(tmp_path):
    return PolicyStore(tmp_path / "policy.json")


# --- extraction, which needs no model at all --------------------------------


def test_free_text_is_pulled_out_of_the_template():
    card = t.build_card(FREE_TEXT, source="stub")
    text = free_text_reason(card.draft.declines[0])

    assert text is not None
    assert "preservation society's own newsletter" in text
    assert "{{AfC submission" not in text  # the prose, not the template


def test_a_coded_decline_has_no_free_text_to_read():
    card = t.build_card(CODED, source="stub")
    assert free_text_reason(card.draft.declines[0]) is None


# --- an estimate stays an estimate ------------------------------------------


def test_a_reading_never_becomes_the_reason_family():
    card = t.build_card(FREE_TEXT, source="stub")
    before = card.top_blocker.family

    interpretation = t.interpret_card(card, FakeInterpreter("source_existence"))

    assert before is ReasonFamily.UNKNOWN
    assert card.top_blocker.family is ReasonFamily.UNKNOWN  # untouched
    assert card.top_blocker.fixable_by_rewrite is None  # still not guessed
    reading = interpretation.families["reason"]
    assert reading.estimated is True
    assert reading.family is ReasonFamily.SOURCE_EXISTENCE  # the estimate, beside it


def test_a_reading_quotes_the_reviewer_verbatim():
    card = t.build_card(FREE_TEXT, source="stub")
    interpretation = t.interpret_card(card, FakeInterpreter())
    reading = interpretation.families["reason"]

    assert reading.evidence
    assert reading.evidence in reading.source_text


def test_a_reading_whose_quote_is_not_in_the_source_is_dropped():
    card = t.build_card(FREE_TEXT, source="stub")
    interpretation = t.interpret_card(
        card, FakeInterpreter(evidence="the reviewer never wrote this sentence")
    )
    assert interpretation.families == {}


def test_no_model_means_no_readings_not_a_failure():
    card = t.build_card(FREE_TEXT, source="stub")
    assert t.interpret_card(card, None).families == {}
    assert t.interpret_card(card, Interpreter(None)).families == {}


def test_cannot_tell_is_an_answer_and_carries_no_family():
    card = t.build_card(FREE_TEXT, source="stub")
    interpretation = t.interpret_card(card, FakeInterpreter("cannot_tell"))
    assert interpretation.families["reason"].family is None


# --- the estimate is disclosed wherever the note goes -----------------------


def test_the_note_says_the_reading_was_estimated_and_what_it_read():
    card = t.build_card(FREE_TEXT, source="stub")
    interpretation = t.interpret_card(card, FakeInterpreter("source_existence"))

    note = t.template_comment(card, interpretation=interpretation)

    assert "an estimate" in note
    assert "not a decision" in note
    assert "changed nothing above" in note
    assert interpretation.families["reason"].evidence in note


def test_a_corrected_reading_is_attributed_to_the_reviewer_not_to_us():
    card = t.build_card(FREE_TEXT, source="stub")
    corrected = Reading(
        kind="reason_family",
        subject="reason",
        value="writing",
        evidence="newsletter",
        reading="a reviewer's own words",
        source_text="newsletter",
        estimated=False,
        corrected_by_reviewer=True,
        original_value="source_existence",
    )
    note = t.template_comment(
        card, interpretation=Interpretation(families={"reason": corrected})
    )

    assert "A reviewer has recorded" in note
    assert "their reading, not ours" in note
    assert "an estimate" not in note


# --- what a written note may not say ----------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The draft should be accepted now.",
        "The subject is notable.",
        "That source is unreliable.",
        "Your edit does not resolve the standing reason.",
        "The core issue remains unaddressed.",
    ],
)
def test_a_note_asserting_a_verdict_is_refused(text):
    card = t.build_card(CODED, source="stub")
    assert t._comment_is_safe(text, card) is False


def test_a_note_naming_a_code_that_is_not_standing_is_refused():
    card = t.build_card(CODED, source="stub")
    assert t._comment_is_safe("The `nn` reason is standing.", card) is True
    assert t._comment_is_safe("The `blp` reason is standing.", card) is False


def test_a_conditional_statement_about_the_guideline_is_allowed():
    """The hand-written note this is measured against contains exactly this."""
    card = t.build_card(CODED, source="stub")
    text = (
        "If sources like that do not exist yet, the draft cannot be accepted no "
        "matter how well it is written."
    )
    assert t._comment_is_safe(text, card) is True


# --- the reviewer's correction, through the intervention layer --------------


def test_the_transform_puts_a_correction_into_the_next_tool_call(store):
    store.record_correction(
        FREE_TEXT,
        "reason",
        was="writing",
        now="source_existence",
        evidence="nothing to build on",
        reading="the reviewer wants sources from outside the society",
    )
    handler = ReviewerCorrections(store)

    event = type(
        "Event",
        (),
        {"tool_use": {"name": "interpret_reasons", "input": {"title": FREE_TEXT}}},
    )()
    action = handler.before_tool_call(event)

    assert type(action).__name__ == "Transform"
    action.apply(event)
    injected = json.loads(event.tool_use["input"]["known_readings"])
    assert injected["reason"]["value"] == "source_existence"
    assert injected["reason"]["corrected_by_reviewer"] is True


def test_the_transform_leaves_other_tools_and_uncorrected_drafts_alone(store):
    handler = ReviewerCorrections(store)

    other = type(
        "Event", (), {"tool_use": {"name": "prepare_comment", "input": {"title": FREE_TEXT}}}
    )()
    assert type(handler.before_tool_call(other)).__name__ == "Proceed"

    uncorrected = type(
        "Event", (), {"tool_use": {"name": "interpret_reasons", "input": {"title": CODED}}}
    )()
    assert type(handler.before_tool_call(uncorrected)).__name__ == "Proceed"


def test_a_correction_reaches_the_next_run_and_the_model_is_not_asked_again(store, tmp_path):
    """The whole co-authoring path, end to end.

    The reviewer does not approve or reject the agent's output. Their edit
    becomes an argument of the agent's next tool call.
    """
    first = run_review(FREE_TEXT, policy=store, queue_dir=tmp_path / "q")
    assert first.ok
    assert first.interpretation.families == {}  # offline model reads nothing

    message = record_reviewer_correction(
        first,
        code="reason",
        family="source_existence",
        policy=store,
        reading="they want somebody outside the society to have written about it",
        note="the newsletter is the society's own",
    )
    assert "source existence" in message

    second = run_review(FREE_TEXT, policy=store, queue_dir=tmp_path / "q")
    reading = second.interpretation.families["reason"]

    assert reading.value == "source_existence"
    assert reading.corrected_by_reviewer is True
    assert reading.estimated is False
    assert reading.original_value == "not read"
    assert second.card.top_blocker.family is ReasonFamily.UNKNOWN  # still not overwritten


def test_the_correction_records_what_the_model_had_said(store, tmp_path):
    outcome = run_review(FREE_TEXT, policy=store, queue_dir=tmp_path / "q")
    outcome.interpretation.families["reason"] = Reading(
        kind="reason_family",
        subject="reason",
        value="writing",
        evidence="nothing to build on",
        reading="a machine reading",
        source_text="nothing to build on",
    )

    record_reviewer_correction(outcome, code="reason", family="source_existence", policy=store)
    record = store.correction_record()

    assert len(record) == 1
    assert record[0]["was"] == "writing"
    assert record[0]["now"] == "source_existence"


def test_a_correction_must_name_a_family_that_exists(store, tmp_path):
    outcome = run_review(FREE_TEXT, policy=store, queue_dir=tmp_path / "q")
    with pytest.raises(ValueError, match="family must be one of"):
        record_reviewer_correction(outcome, code="reason", family="made_up", policy=store)


def test_the_loop_model_and_the_reading_model_are_separable(store, tmp_path):
    """The cheap configuration: a free planner for the plan, a model for the prose.

    Walking a five-step plan is deterministic work. Reading a reviewer's
    sentence is not. Paying per turn for the first to get the second is the cost
    failure this project cannot afford, so the two are chosen separately.
    """
    reader = FakeInterpreter("source_existence")
    outcome = run_review(
        FREE_TEXT, policy=store, queue_dir=tmp_path / "q", interpreter=reader
    )

    assert outcome.ok
    assert reader.asked == ["reason"]  # the reading model was used
    assert outcome.interpretation.families["reason"].value == "source_existence"
    # ...and the loop still ran on the offline planner, which reads nothing.
    assert outcome.stop_reason == "end_turn"


def test_the_handler_is_registered_on_the_assembled_agent(store):
    agent, _ = build_agent(policy=store)
    handlers = agent._intervention_registry.handlers
    assert any(isinstance(h, ReviewerCorrections) for h in handlers)
    assert [h.name for h in handlers][:4] == [
        "first-reader:reversibility",
        "first-reader:escalation-policy",
        "first-reader:reviewer-corrections",
        "first-reader:audit",
    ]
