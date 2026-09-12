"""Blocker ordering, driven by the case this project exists for.

`Draft:0->1 Doctrine for AI` was declined three times:

    #1  ai   (primary) + nn (reason2)   -- Reviewer:Aster
    #2  v    (primary) + ai (reason2)   -- Reviewer:Brill
    #3  nn   (primary)                  -- Reviewer:Cato

`nn` went quiet on the second decline and came back as the primary reason on the
third. The contributor, reading only the most recent notice each time, rewrote
prose against a reason that no amount of rewriting can fix.
"""

from __future__ import annotations

import collections
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from first_reader.analysis.blockers import build_blockers
from first_reader.analysis.reasons import canonical_code
from first_reader.models import AttemptSignal, Decline, DraftHistory, ReasonFamily

T0 = datetime(2026, 1, 10, tzinfo=timezone.utc)


def _decline(code, decliner, day, secondary=None):
    submitted = T0 + timedelta(days=day)
    return Decline(
        code=code,
        secondary_code=secondary,
        decliner=decliner,
        submitted_at=submitted,
        declined_at=submitted + timedelta(days=2),
        raw=f"{{{{AfC submission|d|{code}|decliner={decliner}}}}}",
    )


def doctrine_history() -> DraftHistory:
    return DraftHistory(
        title="Draft:0->1 Doctrine for AI",
        pageid=76543210,
        declines=[
            _decline("ai", "Reviewer:Aster", 0, secondary="nn"),
            _decline("v", "Reviewer:Brill", 30),
            _decline("nn", "Reviewer:Cato", 60),
        ],
        revisions=[],
        is_pending=True,
    )


# The second decline names `ai` again as its secondary reason.
def doctrine_history_full() -> DraftHistory:
    h = doctrine_history()
    h.declines[1] = _decline("v", "Reviewer:Brill", 30, secondary="ai")
    return h


class TestDoctrineScenario:
    def test_nn_is_the_top_blocker(self):
        blockers = build_blockers(doctrine_history_full())
        assert blockers[0].code == "nn", [b.code for b in blockers]

    def test_nn_was_raised_at_least_twice(self):
        top = build_blockers(doctrine_history_full())[0]
        assert top.repeat_count >= 2, "secondary on decline #1, primary on decline #3"

    def test_nn_is_not_fixable_by_rewriting(self):
        top = build_blockers(doctrine_history_full())[0]
        assert top.family is ReasonFamily.SOURCE_EXISTENCE
        assert top.fixable_by_rewrite is False

    def test_v_is_also_not_fixable_by_rewriting(self):
        by_code = {b.code: b for b in build_blockers(doctrine_history_full())}
        assert by_code["v"].family is ReasonFamily.SOURCE_EXISTENCE
        assert by_code["v"].fixable_by_rewrite is False

    def test_nn_first_seen_is_the_first_decline_not_the_third(self):
        top = build_blockers(doctrine_history_full())[0]
        assert top.first_seen == doctrine_history_full().declines[0].declined_at

    def test_nn_carries_both_reviewers_who_raised_it(self):
        top = build_blockers(doctrine_history_full())[0]
        assert top.carried_from == ["Reviewer:Aster", "Reviewer:Cato"]

    def test_the_reason_that_went_quiet_is_still_a_blocker(self):
        """Decline #2 never mentions `nn`. That is not resolution."""
        blockers = build_blockers(doctrine_history_full())
        assert "nn" in {b.code for b in blockers}

    def test_full_order(self):
        # nn: latest primary. ai: 2 mentions, first seen day 2. v: 1 mention.
        blockers = build_blockers(doctrine_history_full())
        assert [b.code for b in blockers] == ["nn", "ai", "v"]

    def test_only_ai_is_worth_rewriting_against(self):
        by_code = {b.code: b for b in build_blockers(doctrine_history_full())}
        assert by_code["ai"].fixable_by_rewrite is True
        assert by_code["nn"].fixable_by_rewrite is False
        assert by_code["v"].fixable_by_rewrite is False


class TestOrdering:
    def test_latest_primary_wins_over_a_more_repeated_reason(self):
        history = DraftHistory(
            title="Draft:X",
            pageid=1,
            declines=[
                _decline("adv", "R1", 0),
                _decline("adv", "R2", 10),
                _decline("adv", "R3", 20),
                _decline("nn", "R4", 30),
            ],
        )
        blockers = build_blockers(history)
        assert blockers[0].code == "nn", "what is blocking now comes first"
        assert blockers[1].code == "adv"
        assert blockers[1].repeat_count == 3

    def test_repeat_count_breaks_ties_below_the_top(self):
        history = DraftHistory(
            title="Draft:X",
            pageid=1,
            declines=[
                _decline("adv", "R1", 0, secondary="context"),
                _decline("adv", "R2", 10),
                _decline("nn", "R3", 20),
            ],
        )
        assert [b.code for b in build_blockers(history)] == ["nn", "adv", "context"]

    def test_earliest_first_seen_breaks_a_repeat_count_tie(self):
        history = DraftHistory(
            title="Draft:X",
            pageid=1,
            declines=[
                _decline("context", "R1", 0),
                _decline("adv", "R2", 10),
                _decline("nn", "R3", 20),
            ],
        )
        assert [b.code for b in build_blockers(history)] == ["nn", "context", "adv"]

    def test_order_is_deterministic_for_fully_tied_reasons(self):
        history = DraftHistory(
            title="Draft:X",
            pageid=1,
            declines=[_decline("npov", "R1", 0, secondary="context"), _decline("nn", "R2", 10)],
        )
        first = [b.code for b in build_blockers(history)]
        assert first == [b.code for b in build_blockers(history)]
        assert first[0] == "nn"


class TestCodeHandling:
    def test_aliases_and_case_collapse_into_one_blocker(self):
        history = DraftHistory(
            title="Draft:X",
            pageid=1,
            declines=[_decline("ADV", "R1", 0), _decline("advert", "R2", 10)],
        )
        blockers = build_blockers(history)
        assert len(blockers) == 1
        assert blockers[0].code == "adv"
        assert blockers[0].repeat_count == 2

    def test_a_code_repeated_within_one_decline_counts_once(self):
        history = DraftHistory(
            title="Draft:X", pageid=1, declines=[_decline("nn", "R1", 0, secondary="nn")]
        )
        assert build_blockers(history)[0].repeat_count == 1

    def test_unknown_code_keeps_its_place_but_admits_it_does_not_know(self):
        history = DraftHistory(title="Draft:X", pageid=1, declines=[_decline("cv", "R1", 0)])
        top = build_blockers(history)[0]
        assert top.code == "cv"
        assert top.family is ReasonFamily.UNKNOWN
        assert top.fixable_by_rewrite is None

    def test_empty_and_missing_codes_are_dropped(self):
        history = DraftHistory(
            title="Draft:X", pageid=1, declines=[_decline("", "R1", 0), _decline("nn", "R2", 10)]
        )
        assert [b.code for b in build_blockers(history)] == ["nn"]

    def test_no_declines_means_no_blockers(self):
        assert build_blockers(DraftHistory(title="Draft:X", pageid=1)) == []

    def test_declines_out_of_order_are_sorted_before_use(self):
        history = doctrine_history_full()
        history.declines = list(reversed(history.declines))
        assert [b.code for b in build_blockers(history)] == ["nn", "ai", "v"]


class TestAttemptWiring:
    def _signal(self):
        return AttemptSignal(
            refs_before=2,
            refs_after=2,
            new_domains=(),
            bytes_changed=2400,
            similarity=0.35,
            touched_target_area=False,
            editor_comments=("rewrote the whole thing",),
        )

    def test_attempt_is_attached_by_code(self):
        blockers = build_blockers(doctrine_history_full(), {"nn": self._signal()})
        by_code = {b.code: b for b in blockers}
        assert by_code["nn"].attempt is not None
        assert by_code["ai"].attempt is None, "not measured is not the same as no attempt"

    def test_attempt_keys_are_normalised(self):
        blockers = build_blockers(doctrine_history_full(), {"NN ": self._signal()})
        assert blockers[0].attempt is not None

    def test_the_wasted_rewrite_this_project_exists_to_catch(self):
        """A substantive rewrite aimed at a reason rewriting cannot fix."""
        from first_reader.models import ReviewCard

        blockers = build_blockers(doctrine_history_full(), {"nn": self._signal()})
        card = ReviewCard(
            draft=doctrine_history_full(),
            blockers=blockers,
            draft_comment="",
            needs_human=True,
            escalation_reason="",
        )
        assert card.top_blocker.code == "nn"
        assert card.top_blocker.attempt.substantive_rewrite is True
        assert card.wasted_rewrite is True


# ---------------------------------------------------------------------------
# Measured on the collected corpus, not on fixtures.
# ---------------------------------------------------------------------------

DATASET = Path(__file__).resolve().parent.parent / "data" / "dataset.jsonl"
needs_dataset = pytest.mark.skipif(not DATASET.exists(), reason="data/dataset.jsonl not collected")


@pytest.fixture(scope="module")
def corpus():
    from first_reader.wiki.collect import load_dataset

    return load_dataset(DATASET)


@needs_dataset
def test_most_drafts_are_told_the_same_thing_more_than_once(corpus):
    """63.5% of drafts get the same code twice or more.

    A separate count of primary codes alone gives 56.7%; the difference is the
    `reason2` parameter, which `build_blockers` counts because a secondary
    reason is a reason. Both numbers are recorded so neither looks like a bug.
    """
    assert len(corpus) == 540

    repeated = sum(1 for h in corpus if any(b.repeat_count >= 2 for b in build_blockers(h)))
    assert repeated == 343
    assert repeated / len(corpus) == pytest.approx(0.635, abs=0.005)

    primary_only = 0
    for h in corpus:
        counts = collections.Counter(
            (d.code or "").strip().lower() for d in h.declines if (d.code or "").strip()
        )
        if any(v >= 2 for v in counts.values()):
            primary_only += 1
    assert primary_only == 306
    assert primary_only / len(corpus) == pytest.approx(0.567, abs=0.005)


@needs_dataset
def test_two_thirds_of_drafts_are_blocked_by_something_rewriting_cannot_fix(corpus):
    """The finding the whole tool rests on.

    For 67.6% of these drafts the reason at the top of the card is one where
    editing prose is beside the point -- and 45.9% have been told such a reason
    at least twice.
    """
    unfixable_top = 0
    repeated_source_existence = 0
    for h in corpus:
        blockers = build_blockers(h)
        if blockers and blockers[0].fixable_by_rewrite is False:
            unfixable_top += 1
        if any(
            b.repeat_count >= 2 and b.family is ReasonFamily.SOURCE_EXISTENCE for b in blockers
        ):
            repeated_source_existence += 1

    assert unfixable_top == 365
    assert unfixable_top / len(corpus) == pytest.approx(0.676, abs=0.005)
    assert repeated_source_existence == 248


@needs_dataset
def test_no_draft_loses_a_reason_that_was_ever_raised(corpus):
    """Silence in a later decline never removes an earlier blocker."""
    for h in corpus:
        raised = {
            canonical_code(code)
            for d in h.declines
            for code in (d.code, d.secondary_code)
            if (code or "").strip()
        }
        assert {b.code for b in build_blockers(h)} == raised, h.title
