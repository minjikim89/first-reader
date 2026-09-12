"""The assembled agent: what it does, and what it structurally cannot do.

Every test here runs the real event loop, the real tools and the real
intervention layer against the offline planner model. No network, no inference.
"""

from __future__ import annotations

import json

import pytest

from _agent_support import no_network, offline  # noqa: F401

from first_reader.agent import _stubs, tools as t
from first_reader.agent.core import (
    DEFAULT_LIMITS,
    build_agent,
    record_reviewer_response,
    run_batch,
    run_review,
)
from first_reader.agent.policy import DEFAULT_RELAX_AFTER, PolicyStore
from first_reader.agent.reversibility import grade_of

WASTED = "Draft:Corner Bakery Collective"
INCONSISTENT = "Draft:Mira Okonkwo"
REPEATED = "Draft:Blue Ridge Analytics"


@pytest.fixture
def store(tmp_path):
    return PolicyStore(tmp_path / "policy.json")


@pytest.fixture
def queue(tmp_path):
    return tmp_path / "queue"


# --- end to end -------------------------------------------------------------


def test_one_draft_runs_start_to_finish(store, queue):
    outcome = run_review(WASTED, policy=store, queue_dir=queue)

    assert outcome.stop_reason == "end_turn"
    assert outcome.card is not None
    assert outcome.judgment is not None
    assert [entry["tool"] for entry in outcome.audit] == [
        "get_draft_history",
        "analyze_blockers",
        "interpret_reasons",
        "prepare_comment",
        "judge_escalation",
        "escalate_to_reviewer",
    ]
    assert outcome.denied_calls == []
    assert outcome.backstopped is False


def test_every_fixture_draft_runs_and_lands_somewhere(store, queue):
    outcomes = run_batch(_stubs.titles(), policy=store, queue_dir=queue)

    assert len(outcomes) == len(_stubs.titles())
    for outcome in outcomes:
        assert outcome.stop_reason == "end_turn"
        assert outcome.judgment is not None
        assert outcome.card is not None
        assert outcome.card.draft_comment
        # A card is queued if and only if the judgment said to escalate.
        assert bool(outcome.queued_at) == outcome.escalated


def test_an_escalated_card_is_written_to_disk_and_reads_back(store, queue):
    outcome = run_review(WASTED, policy=store, queue_dir=queue)

    assert outcome.escalated is True
    written = json.loads((queue / "Draft_Corner_Bakery_Collective.json").read_text())
    assert written["needs_human"] is True
    assert written["draft"]["title"] == WASTED
    assert written["escalation_reason"]
    assert "First Reader" in written["draft_comment"]


def test_the_prepared_comment_reports_and_does_not_judge(store, queue):
    outcome = run_review(WASTED, policy=store, queue_dir=queue)
    comment = outcome.card.draft_comment

    assert "Rewriting cannot clear this reason" in comment
    assert "did not assess this one" in comment
    for verdict_word in ("should be accepted", "should be declined", "is notable", "not notable"):
        assert verdict_word not in comment


def test_nothing_is_left_waiting_for_a_human(store, queue):
    """The run finishes. The card waits on disk; no process waits on a reviewer."""
    outcome = run_review(WASTED, policy=store, queue_dir=queue)

    assert outcome.stop_reason == "end_turn"
    assert outcome.stop_reason != "interrupt"
    assert (queue / "Draft_Corner_Bakery_Collective.json").exists()


def test_the_run_is_capped(store, queue):
    assert DEFAULT_LIMITS["turns"] <= 10
    assert DEFAULT_LIMITS["total_tokens"] <= 200_000
    outcome = run_review(WASTED, policy=store, queue_dir=queue, limits={"turns": 2})
    assert outcome.stop_reason == "limit_turns"


def test_the_stub_is_not_reachable_from_a_production_source(tmp_path):
    """No fallback to fixtures, ever. This is the regression test for the bug
    where every unit test passed while the agent raised KeyError on every real
    draft, because the tools were bound to `_stubs` and nothing said so."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")

    with pytest.raises(t.HistoryUnavailable) as caught:
        t.fetch_history(WASTED, source="dataset", dataset=empty)

    message = str(caught.value)
    assert WASTED in message
    assert "wiki.collect" in message or "FIRST_READER_SOURCE" in message


def test_the_production_default_is_not_the_stub(monkeypatch):
    monkeypatch.delenv("FIRST_READER_SOURCE", raising=False)
    assert t.current_source() is t.Source.AUTO


# --- what the agent cannot do ----------------------------------------------


def test_every_tool_declares_a_reversibility_grade(store):
    for tool in [*t.STATIC_TOOLS, t.make_judge_escalation(store)]:
        assert grade_of(tool.tool_spec) is not None, tool.tool_spec["name"]


def test_an_irreversible_tool_is_refused_even_when_the_model_calls_it(store, queue):
    from _agent_support import ScriptedModel

    model = ScriptedModel(
        [
            {"type": "tool", "name": "post_reviewer_decision", "input": {"title": WASTED, "verdict": "accept"}},
            {"type": "text", "text": "refused"},
        ]
    )
    agent, _ = build_agent(policy=store, model=model, queue_dir=queue)
    result = agent(f"Prepare this draft for review: <title>{WASTED}</title>", limits={"turns": 4})

    assert result.stop_reason == "end_turn"
    denials = [
        block["toolResult"]
        for message in agent.messages
        for block in message["content"]
        if isinstance(block, dict) and "toolResult" in block
    ]
    assert len(denials) == 1
    text = denials[0]["content"][0]["text"]
    assert denials[0]["status"] == "error"
    assert text.startswith("DENIED:")
    assert "does not review drafts" in text


def test_an_undeclared_tool_is_refused_rather_than_allowed(store, queue):
    """A new tool arrives with no permissions. The gate fails closed."""
    from _agent_support import ScriptedModel
    from strands import tool

    @tool
    def undeclared_tool(title: str) -> str:
        """A tool nobody graded.

        Args:
            title: anything
        """
        raise AssertionError("an undeclared tool ran")

    model = ScriptedModel(
        [
            {"type": "tool", "name": "undeclared_tool", "input": {"title": WASTED}},
            {"type": "text", "text": "refused"},
        ]
    )
    agent, _ = build_agent(policy=store, model=model, queue_dir=queue)
    agent.tool_registry.register_tool(undeclared_tool)
    agent(f"<title>{WASTED}</title>", limits={"turns": 4})

    text = next(
        block["toolResult"]["content"][0]["text"]
        for message in agent.messages
        for block in message["content"]
        if isinstance(block, dict) and "toolResult" in block
    )
    assert "no declared reversibility grade" in text


def test_the_agent_cannot_escalate_against_the_judgment(store, queue):
    from _agent_support import ScriptedModel

    model = ScriptedModel(
        [
            {"type": "tool", "name": "escalate_to_reviewer", "input": {"title": REPEATED, "why": "because"}},
            {"type": "text", "text": "refused"},
        ]
    )
    agent, _ = build_agent(policy=store, model=model, queue_dir=queue)
    agent(f"<title>{REPEATED}</title>", limits={"turns": 4})

    text = next(
        block["toolResult"]["content"][0]["text"]
        for message in agent.messages
        for block in message["content"]
        if isinstance(block, dict) and "toolResult" in block
    )
    assert "No escalation judgment has been made" in text
    assert not queue.exists() or not list(queue.glob("*.json"))


def test_a_card_the_judgment_wanted_escalated_is_filed_even_if_the_model_forgets(store, queue):
    from _agent_support import ScriptedModel

    model = ScriptedModel(
        [
            {"type": "tool", "name": "get_draft_history", "input": {"title": WASTED}},
            {"type": "tool", "name": "analyze_blockers", "input": {"title": WASTED}},
            {"type": "tool", "name": "interpret_reasons", "input": {"title": WASTED}},
            {"type": "tool", "name": "prepare_comment", "input": {"title": WASTED}},
            {"type": "tool", "name": "judge_escalation", "input": {"title": WASTED}},
            {"type": "text", "text": "I have decided to stop here."},
        ]
    )
    outcome = run_review(WASTED, policy=store, model=model, queue_dir=queue)

    assert outcome.escalated is True
    assert outcome.backstopped is True
    assert (queue / "Draft_Corner_Bakery_Collective.json").exists()


# --- learning, through the agent -------------------------------------------


def test_precedent_alone_settles_a_repeated_reason_on_the_second_run(store, queue):
    first = run_review(REPEATED, policy=store, queue_dir=queue)
    assert first.escalated is True
    assert first.judgment["state"] == "no_precedent"

    second = run_review(REPEATED, policy=store, queue_dir=queue)
    assert second.judgment["state"] == "repeated_reason_no_attempt"
    assert second.escalated is False


def test_five_agreements_stop_a_pattern_reaching_the_reviewer(store, queue):
    outcome = run_review(INCONSISTENT, policy=store, queue_dir=queue)
    assert outcome.escalated is True

    for _ in range(DEFAULT_RELAX_AFTER - 1):
        message = record_reviewer_response(outcome, agreed=True, policy=store)
        assert "needed before" in message
        assert run_review(INCONSISTENT, policy=store, queue_dir=queue).escalated is True

    message = record_reviewer_response(outcome, agreed=True, policy=store)
    assert "Learned a new rule" in message

    relaxed = run_review(INCONSISTENT, policy=store, queue_dir=queue)
    assert relaxed.escalated is False
    assert relaxed.judgment["policy_rule_id"] is not None


def test_one_reversal_brings_the_reviewer_straight_back(store, queue):
    outcome = run_review(INCONSISTENT, policy=store, queue_dir=queue)
    for _ in range(DEFAULT_RELAX_AFTER):
        record_reviewer_response(outcome, agreed=True, policy=store)
    assert run_review(INCONSISTENT, policy=store, queue_dir=queue).escalated is False

    message = record_reviewer_response(outcome, agreed=False, policy=store, note="this one was wrong")
    assert "Reversed immediately" in message

    back = run_review(INCONSISTENT, policy=store, queue_dir=queue)
    assert back.escalated is True
    assert back.judgment["policy_rule_id"] is not None


def test_the_policy_a_run_used_is_visible_in_the_transcript(store):
    for _ in range(DEFAULT_RELAX_AFTER):
        store.record_agreement("inconsistent_reviewers|adv|writing|repeats=1")
    agent, _ = build_agent(policy=store)

    mirrored = agent.state.get("first_reader:policy")
    assert mirrored["rules"][0]["kind"] == "relax"
    assert mirrored["relax_after"] == DEFAULT_RELAX_AFTER
