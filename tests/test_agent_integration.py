"""The agent against real data. Live Wikipedia and the collected dataset.

    pytest -m integration tests/test_agent_integration.py

Deselected by default. These exist because the unit suite cannot catch the
failure they cover: every unit test runs against `_stubs.py`, and for a while
every one of them passed while the agent raised `KeyError` on every real draft
title. A suite that only exercises the stub proves the stub works.

`Draft:0→1 Doctrine for AI` is the reference case. It was declined three times
in mid-2026: `ai`+`nn`, then `v`+`ai`, then `nn` alone. Between the second
decline and the third submission the contributor rewrote the draft almost
completely -- and the reason that came back was `nn`, which rewriting cannot
fix. That is the case this whole project is named for, and it is real.

If that draft is later accepted, moved or blanked these will fail. Replace it
with another draft whose history has the same shape rather than weakening the
assertions.
"""

from __future__ import annotations

import pytest

from first_reader.agent import tools as t
from first_reader.agent.core import run_review
from first_reader.agent.escalation import AdvantageState, judge
from first_reader.agent.policy import PolicyStore
from first_reader.models import ReasonFamily

pytestmark = pytest.mark.integration

REFERENCE = "Draft:0→1 Doctrine for AI"
PAGEID = 83560144
DATASET = t.default_dataset()


@pytest.fixture
def store(tmp_path):
    return PolicyStore(tmp_path / "policy.json")


@pytest.fixture(autouse=True)
def real_source(monkeypatch):
    """Dataset first, live API second -- the production default, stated."""
    monkeypatch.setenv("FIRST_READER_SOURCE", "auto")
    monkeypatch.setenv("FIRST_READER_MODEL", "offline")


# --- the live API path ------------------------------------------------------


def test_the_reference_draft_comes_back_from_the_live_api():
    history = t.fetch_history(REFERENCE, source="wikipedia")

    assert history.pageid == PAGEID
    assert history.decline_count == 3
    assert [d.code for d in history.declines] == ["ai", "v", "nn"]
    assert len(history.revisions) >= 11


def test_the_reference_draft_builds_the_card_the_project_is_named_for():
    card = t.build_card(REFERENCE, source="wikipedia")

    top = card.top_blocker
    assert top is not None
    assert top.code == "nn"
    assert top.family is ReasonFamily.SOURCE_EXISTENCE
    assert top.fixable_by_rewrite is False
    assert top.repeat_count == 2  # secondary on decline 1, primary on decline 3

    # The contributor did rewrite, substantively, against a reason rewriting
    # cannot fix. Measured from the wikitext of two real revisions.
    assert top.attempt is not None
    assert top.attempt.substantive_rewrite is True
    assert top.attempt.similarity < 0.7
    assert card.wasted_rewrite is True

    assert "Rewriting cannot clear this reason" in card.draft_comment
    assert "did not assess this one" in card.draft_comment


def test_the_full_agent_runs_on_the_reference_draft(store, tmp_path):
    outcome = run_review(REFERENCE, policy=store, queue_dir=tmp_path / "queue")

    assert outcome.ok, outcome.error
    assert outcome.stop_reason == "end_turn"
    assert [entry["tool"] for entry in outcome.audit] == [
        "get_draft_history",
        "analyze_blockers",
        "interpret_reasons",
        "prepare_comment",
        "judge_escalation",
        "escalate_to_reviewer",
    ]
    assert outcome.judgment["state"] == AdvantageState.WASTED_REWRITE.value
    assert outcome.escalated is True
    assert (tmp_path / "queue" / "Draft_0_1_Doctrine_for_AI.json").exists()


# --- the local dataset path -------------------------------------------------


@pytest.mark.skipif(not DATASET.exists(), reason="dataset not collected")
def test_an_arbitrary_dataset_draft_gets_a_judgment(store, tmp_path):
    titles = t.dataset_titles()
    assert len(titles) > 100

    title = titles[len(titles) // 2]  # arbitrary, but the same one every run
    outcome = run_review(title, policy=store, queue_dir=tmp_path / "queue")

    assert outcome.ok, f"{title}: {outcome.error}"
    assert outcome.judgment["state"] in {s.value for s in AdvantageState}
    assert outcome.card.blockers
    assert outcome.card.draft_comment


@pytest.mark.skipif(not DATASET.exists(), reason="dataset not collected")
def test_the_dataset_path_needs_no_network():
    """A title in the dataset resolves locally, without touching the API."""
    title = t.dataset_titles()[0]
    history = t.fetch_history(title, source="dataset")

    assert history.title == title
    assert history.decline_count >= 1


@pytest.mark.skipif(not DATASET.exists(), reason="dataset not collected")
def test_every_dataset_draft_builds_a_card():
    """The failure rate is the demo's credibility. scripts/backtest.py reports it."""
    from first_reader.analysis import build_blockers
    from first_reader.models import ReviewCard
    from first_reader.wiki.collect import load_dataset

    failures = []
    for history in load_dataset(DATASET):
        try:
            card = ReviewCard(
                draft=history,
                blockers=build_blockers(history, None),
                draft_comment="",
                needs_human=False,
                escalation_reason="",
            )
            card.draft_comment = t.compose_comment(card)
            judge(card, precedent_seen=False)
        except Exception as exc:  # noqa: BLE001
            failures.append((history.title, f"{type(exc).__name__}: {exc}"))

    assert not failures, f"{len(failures)} drafts failed to build: {failures[:5]}"


# --- the bug this file exists for -------------------------------------------


def test_an_unknown_draft_raises_rather_than_answering_about_a_fixture():
    with pytest.raises(t.HistoryUnavailable):
        t.fetch_history("Draft:This Page Does Not Exist 8f3a1c", source="wikipedia")


def test_a_fixture_title_is_not_reachable_from_the_production_source():
    """The stub is opt-in. Nothing falls back to it."""
    with pytest.raises(t.HistoryUnavailable):
        t.fetch_history("Draft:Corner Bakery Collective", source="wikipedia")
