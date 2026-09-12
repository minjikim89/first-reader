"""The escalation judgment, one test per evidence state.

The table these tests pin down:

    evidence state                  InterventionAdvantage   escalates
    ------------------------------  ---------------------   ---------
    repeated reason, no attempt     low                     no
    reason family UNKNOWN           high                    yes
    reviewers gave different reasons  high                  yes
    wasted rewrite                  highest                 yes
    no precedent                    high                    yes

``precedent_seen`` is passed explicitly in each test rather than read from a
policy store, so that a state is tested for itself and not for the cold-start
behaviour that would escalate it anyway.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from _agent_support import no_network, offline  # noqa: F401

from first_reader.agent.escalation import (
    ADVANTAGE,
    NON_RELAXABLE,
    AdvantageState,
    Weights,
    apply_rule,
    classify,
    judge,
    pattern_key,
    signature,
)
from first_reader.agent.tools import build_card
from first_reader.models import ReasonFamily

WASTED = "Draft:Corner Bakery Collective"
UNKNOWN = "Draft:Helvetia Signals"
INCONSISTENT = "Draft:Mira Okonkwo"
FRESH = "Draft:Tuvan Throat Singing in Film"
REPEATED = "Draft:Blue Ridge Analytics"


# --- one test per row of the table ------------------------------------------


def test_repeated_reason_with_no_attempt_is_handled_without_a_reviewer():
    card = build_card(REPEATED)
    verdict = judge(card, precedent_seen=True)

    assert verdict.state is AdvantageState.REPEATED_REASON_NO_ATTEMPT
    assert verdict.advantage == ADVANTAGE[AdvantageState.REPEATED_REASON_NO_ATTEMPT]
    assert verdict.escalate is False
    assert verdict.score < verdict.interruption_cost
    # The prior reason is what gets carried forward, so it has to still be there.
    assert card.top_blocker is not None
    assert card.top_blocker.repeat_count >= 3
    # Measured, and it measured nothing. Not the same as unmeasured -- see below.
    assert card.top_blocker.attempt is not None
    assert card.top_blocker.attempt.added_sources == 0
    assert card.top_blocker.attempt.substantive_rewrite is False


def test_unknown_reason_family_goes_to_a_reviewer():
    card = build_card(UNKNOWN)
    verdict = judge(card, precedent_seen=True)

    assert card.top_blocker is not None
    assert card.top_blocker.family is ReasonFamily.UNKNOWN
    assert card.top_blocker.fixable_by_rewrite is None  # never guessed
    assert verdict.state is AdvantageState.UNKNOWN_FAMILY
    assert verdict.escalate is True


def test_reviewers_giving_different_reasons_goes_to_a_reviewer():
    card = build_card(INCONSISTENT)
    verdict = judge(card, precedent_seen=True)

    codes = {d.code for d in card.draft.declines}
    decliners = {d.decliner for d in card.draft.declines}
    assert len(codes) > 1 and len(decliners) > 1
    assert verdict.state is AdvantageState.INCONSISTENT_REVIEWERS
    assert verdict.escalate is True


def test_wasted_rewrite_is_the_strongest_case_for_a_reviewer():
    card = build_card(WASTED)
    verdict = judge(card, precedent_seen=True)

    assert card.wasted_rewrite is True
    assert verdict.state is AdvantageState.WASTED_REWRITE
    assert verdict.escalate is True
    # Highest advantage of any state, and by a margin over the others.
    assert verdict.advantage == max(ADVANTAGE.values())
    assert verdict.score > judge(build_card(UNKNOWN), precedent_seen=True).score


def test_an_unmeasured_attempt_goes_to_a_reviewer_rather_than_being_assumed_away():
    """"We did not look" must never be scored as "the contributor did nothing"."""
    card = build_card(REPEATED)
    assert card.top_blocker is not None

    stripped = replace(card.top_blocker, attempt=None)
    card.blockers = [stripped, *card.blockers[1:]]
    verdict = judge(card, precedent_seen=True)

    assert verdict.state is AdvantageState.ATTEMPT_NOT_MEASURED
    assert verdict.escalate is True
    # Same draft, same everything, except that the evidence was measured.
    assert judge(build_card(REPEATED), precedent_seen=True).escalate is False


def test_a_pattern_with_no_precedent_goes_to_a_reviewer():
    card = build_card(FRESH)
    verdict = judge(card, precedent_seen=False)

    assert verdict.state is AdvantageState.NO_PRECEDENT
    assert verdict.escalate is True


# --- the shape of the judgment itself ---------------------------------------


def test_the_same_draft_is_routine_once_the_pattern_has_precedent():
    """Cold start escalates; autonomy is earned by having seen the case before."""
    card = build_card(FRESH)
    assert judge(card, precedent_seen=False).escalate is True

    settled = judge(card, precedent_seen=True)
    assert settled.state is AdvantageState.ROUTINE
    assert settled.escalate is False


def test_every_state_is_reachable_and_each_row_of_the_table_is_distinct():
    unmeasured = build_card(REPEATED)
    unmeasured.blockers = [
        replace(unmeasured.blockers[0], attempt=None),
        *unmeasured.blockers[1:],
    ]
    states = {
        classify(build_card(REPEATED), precedent_seen=True),
        classify(build_card(UNKNOWN), precedent_seen=True),
        classify(build_card(INCONSISTENT), precedent_seen=True),
        classify(build_card(WASTED), precedent_seen=True),
        classify(unmeasured, precedent_seen=True),
        classify(build_card(FRESH), precedent_seen=False),
        classify(build_card(FRESH), precedent_seen=True),
    }
    assert states == set(AdvantageState)


def test_the_judgment_shows_its_arithmetic():
    verdict = judge(build_card(WASTED), precedent_seen=True)
    assert verdict.score == pytest.approx(
        verdict.advantage * verdict.cost * (1 - verdict.recoverability)
    )
    assert verdict.escalate == (verdict.score > verdict.interruption_cost)
    for term in ("advantage", "cost", "irrecoverable", "threshold"):
        assert term in verdict.reason


def test_raising_the_cost_of_an_interruption_narrows_what_gets_escalated():
    card = build_card(UNKNOWN)
    assert judge(card, precedent_seen=True).escalate is True
    guarded = judge(card, precedent_seen=True, weights=Weights(interruption_cost=0.9))
    assert guarded.escalate is False


# --- policy rules on top of the judgment ------------------------------------


def test_a_relaxation_rule_can_silence_an_ordinary_escalation():
    verdict = judge(build_card(UNKNOWN), precedent_seen=True)
    relaxed = apply_rule(verdict, rule_id=7, rule_sentence="handle it", forces_escalation=False)

    assert relaxed.escalate is False
    assert relaxed.policy_rule_id == 7
    assert "#7" in relaxed.reason
    assert relaxed.score == verdict.score  # the arithmetic is preserved, not rewritten


def test_no_learned_rule_can_silence_a_wasted_rewrite():
    """A contributor burning time on the wrong problem is a harm, not an inefficiency."""
    verdict = judge(build_card(WASTED), precedent_seen=True)
    attempted = apply_rule(verdict, rule_id=7, rule_sentence="handle it", forces_escalation=False)

    assert AdvantageState.WASTED_REWRITE in NON_RELAXABLE
    assert attempted.escalate is True
    assert attempted.policy_rule_id is None


def test_a_tightening_rule_forces_an_escalation_the_arithmetic_would_have_skipped():
    verdict = judge(build_card(REPEATED), precedent_seen=True)
    assert verdict.escalate is False

    tightened = apply_rule(verdict, rule_id=9, rule_sentence="always ask", forces_escalation=True)
    assert tightened.escalate is True
    assert tightened.policy_rule_id == 9


# --- keys -------------------------------------------------------------------


def test_pattern_key_does_not_depend_on_the_state_it_is_used_to_decide():
    card = build_card(FRESH)
    assert pattern_key(card) == pattern_key(card)
    for state in AdvantageState:
        assert signature(card, state).endswith(pattern_key(card))
        assert signature(card, state).startswith(state.value)


def test_repeat_counts_are_bucketed_so_a_rule_can_accumulate_evidence():
    assert "repeats=3+" in pattern_key(build_card(REPEATED))
    assert "repeats=2" in pattern_key(build_card(WASTED))
    assert "repeats=1" in pattern_key(build_card(FRESH))
