"""The learned policy, and the asymmetry in how it learns.

Widening what the agent handles alone takes five agreements. Narrowing it takes
one reversal. The four-agreement case is tested explicitly: a threshold that
fires early is the same failure as no threshold at all.
"""

from __future__ import annotations

import json

import pytest

from _agent_support import no_network, offline  # noqa: F401

from first_reader.agent.policy import (
    DEFAULT_RELAX_AFTER,
    RELAX,
    TIGHTEN,
    PolicyStore,
    relax_sentence,
    tighten_sentence,
)

SIGNATURE = "repeated_reason_no_attempt|nn|source_existence|repeats=3+"
OTHER = "unknown_family|cv|unknown|repeats=1"


@pytest.fixture
def store(tmp_path):
    return PolicyStore(tmp_path / "policy.json")


# --- the asymmetry ----------------------------------------------------------


def test_four_agreements_do_not_relax_anything(store):
    for _ in range(4):
        assert store.record_agreement(SIGNATURE) is None

    assert store.rules == []
    assert store.active_rule_for(SIGNATURE) is None
    assert store.observation(SIGNATURE).agreements == 4


def test_the_fifth_agreement_relaxes(store):
    for _ in range(DEFAULT_RELAX_AFTER - 1):
        assert store.record_agreement(SIGNATURE) is None

    rule = store.record_agreement(SIGNATURE)
    assert rule is not None
    assert rule.kind == RELAX
    assert rule.active is True
    assert store.active_rule_for(SIGNATURE) is rule


def test_one_reversal_tightens_immediately(store):
    rule = store.record_reversal(SIGNATURE)

    assert rule.kind == TIGHTEN
    assert rule.active is True
    assert store.active_rule_for(SIGNATURE) is rule
    assert store.observation(SIGNATURE).reversals == 1


def test_a_reversal_revokes_a_relaxation_that_took_five_agreements_to_earn(store):
    for _ in range(DEFAULT_RELAX_AFTER):
        store.record_agreement(SIGNATURE)
    relaxed = store.active_rule_for(SIGNATURE)
    assert relaxed is not None and relaxed.kind == RELAX

    tightened = store.record_reversal(SIGNATURE)

    assert relaxed.active is False
    assert tightened.kind == TIGHTEN
    assert store.active_rule_for(SIGNATURE) is tightened
    assert f"#{relaxed.id}" in tightened.evidence
    assert store.observation(SIGNATURE).agreements == 0


def test_agreements_cannot_relax_a_pattern_a_human_has_already_overruled(store):
    store.record_reversal(SIGNATURE)
    for _ in range(DEFAULT_RELAX_AFTER * 2):
        assert store.record_agreement(SIGNATURE) is None

    assert store.active_rule_for(SIGNATURE).kind == TIGHTEN


def test_a_human_disabling_the_tightening_rule_clears_the_way_again(store):
    tightened = store.record_reversal(SIGNATURE)
    store.disable(tightened.id)

    for _ in range(DEFAULT_RELAX_AFTER - 1):
        assert store.record_agreement(SIGNATURE) is None
    assert store.record_agreement(SIGNATURE).kind == RELAX


def test_learning_on_one_pattern_does_not_move_another(store):
    for _ in range(DEFAULT_RELAX_AFTER):
        store.record_agreement(SIGNATURE)

    assert store.active_rule_for(SIGNATURE) is not None
    assert store.active_rule_for(OTHER) is None


# --- rules are readable, not weights ----------------------------------------


def test_a_rule_is_a_sentence_with_its_evidence(store):
    for _ in range(DEFAULT_RELAX_AFTER):
        store.record_agreement(SIGNATURE, note="reviewers agreed on sight")
    rule = store.active_rule_for(SIGNATURE)

    assert isinstance(rule.sentence, str)
    assert rule.sentence.endswith(".")
    assert len(rule.sentence.split()) > 12  # a sentence, not a label
    assert "nn" in rule.sentence
    assert "5 comparable cases" in rule.evidence
    assert "reviewers agreed on sight" in rule.evidence

    rendered = rule.render()
    assert f"Rule #{rule.id}" in rendered
    assert "evidence:" in rendered
    assert "status: active" in rendered


def test_no_rule_carries_a_weight(store):
    for _ in range(DEFAULT_RELAX_AFTER):
        store.record_agreement(SIGNATURE)
    rule = store.active_rule_for(SIGNATURE)

    fields = json.loads(json.dumps(rule.__dict__))
    numeric = {k: v for k, v in fields.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    assert set(numeric) == {"id"}  # the identifier, and nothing else that is a number


def test_the_sentences_say_what_the_rule_does():
    assert "do not" in relax_sentence(SIGNATURE)
    assert "put it in front of a reviewer" in relax_sentence(SIGNATURE)
    assert "always put it in front of a reviewer" in tighten_sentence(SIGNATURE)
    assert "rewriting cannot fix" in relax_sentence(SIGNATURE)


# --- the three switches -----------------------------------------------------


def test_a_rule_can_be_disabled_and_enabled_by_a_human(store):
    for _ in range(DEFAULT_RELAX_AFTER):
        store.record_agreement(SIGNATURE)
    rule = store.active_rule_for(SIGNATURE)

    store.disable(rule.id)
    assert rule.status == "disabled"
    assert store.active_rule_for(SIGNATURE) is None

    store.enable(rule.id)
    assert store.active_rule_for(SIGNATURE) is rule


def test_a_locked_rule_survives_a_reversal_and_names_itself_in_the_evidence(store):
    for _ in range(DEFAULT_RELAX_AFTER):
        store.record_agreement(SIGNATURE)
    relaxed = store.active_rule_for(SIGNATURE)
    store.lock(relaxed.id)

    tightened = store.record_reversal(SIGNATURE)

    assert relaxed.active is True
    assert relaxed.status == "locked"
    assert "locked" in tightened.evidence
    # The tightening rule still wins while both stand, so the reviewer is not
    # silently overruled by their own pin.
    assert store.active_rule_for(SIGNATURE) is tightened


def test_a_locked_rule_cannot_be_disabled_without_unlocking_it(store):
    for _ in range(DEFAULT_RELAX_AFTER):
        store.record_agreement(SIGNATURE)
    rule = store.active_rule_for(SIGNATURE)
    store.lock(rule.id)

    with pytest.raises(ValueError, match="locked"):
        store.disable(rule.id)

    store.unlock(rule.id)
    store.disable(rule.id)
    assert rule.active is False


# --- persistence ------------------------------------------------------------


def test_the_file_is_the_source_of_truth(tmp_path):
    path = tmp_path / "policy.json"
    first = PolicyStore(path)
    for _ in range(DEFAULT_RELAX_AFTER):
        first.record_agreement(SIGNATURE)
    first.observe("nn|source_existence|repeats=3+")

    reopened = PolicyStore(path)
    assert [r.sentence for r in reopened.rules] == [r.sentence for r in first.rules]
    assert reopened.active_rule_for(SIGNATURE) is not None
    assert reopened.precedent_seen("nn|source_existence|repeats=3+") is True
    assert reopened._next_rule_id == first._next_rule_id

    written = json.loads(path.read_text())
    assert written["rules"][0]["kind"] == RELAX


def test_precedent_is_counted_separately_from_agreement(store):
    """Having seen a pattern is not evidence that handling it alone was right."""
    key = "nn|source_existence|repeats=3+"
    for _ in range(10):
        store.observe(key)

    assert store.precedent_seen(key) is True
    assert store.observation(key).agreements == 0
    assert store.rules == []


def test_an_absent_policy_file_is_an_empty_policy_not_an_error(tmp_path):
    store = PolicyStore(tmp_path / "nothing" / "policy.json", autosave=False)
    assert store.rules == []
    assert "No rules learned yet" in store.render()
