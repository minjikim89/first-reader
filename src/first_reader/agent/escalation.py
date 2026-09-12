"""What gets put in front of a reviewer, and why.

The judgment is::

    Escalate iff InterventionAdvantage x Cost x (1 - Recoverability) > InterruptionCost

The first term is deliberately not P(error). *Calibration Is Not Control*
(2026-06) separates two propositions that are routinely conflated: how likely a
system is to be wrong, and whether a human looking at the case changes the
outcome. A draft can be near-certainly declinable and still be worth nobody's
time; a draft can be entirely ordinary and still be the one case where a
reviewer's eye changes what happens to a contributor. We estimate the second.

Every term is read off evidence already present in the history. Nothing here
asks a language model how confident it is. A number produced by asking a model
for its own certainty is not evidence, and would put a fabricated quantity at
the centre of the only decision this system makes.

One state is not from the product table: ``ATTEMPT_NOT_MEASURED``. The table
assumes we know what the contributor did since the last decline, and on a bulk
offline pass we do not -- measuring it needs wikitext, which means the API.
Treating "not measured" as "nothing was done" would auto-handle exactly the
cases we are least entitled to auto-handle, so it escalates instead.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from ..models import Blocker, ReasonFamily, ReviewCard


class AdvantageState(str, Enum):
    """The evidence state that sets InterventionAdvantage.

    Checked in the order listed in ``classify``; the first match wins.
    """

    WASTED_REWRITE = "wasted_rewrite"
    UNKNOWN_FAMILY = "unknown_family"
    INCONSISTENT_REVIEWERS = "inconsistent_reviewers"
    ATTEMPT_NOT_MEASURED = "attempt_not_measured"
    NO_PRECEDENT = "no_precedent"
    REPEATED_REASON_NO_ATTEMPT = "repeated_reason_no_attempt"
    ROUTINE = "routine"


#: InterventionAdvantage per evidence state. Not probabilities -- these are
#: "how much does a human looking at this change the outcome", on 0..1.
ADVANTAGE: dict[AdvantageState, float] = {
    AdvantageState.WASTED_REWRITE: 0.95,
    AdvantageState.UNKNOWN_FAMILY: 0.80,
    AdvantageState.INCONSISTENT_REVIEWERS: 0.75,
    AdvantageState.ATTEMPT_NOT_MEASURED: 0.70,
    AdvantageState.NO_PRECEDENT: 0.70,
    AdvantageState.REPEATED_REASON_NO_ATTEMPT: 0.15,
    AdvantageState.ROUTINE: 0.20,
}

EXPLANATION: dict[AdvantageState, str] = {
    AdvantageState.WASTED_REWRITE: (
        "The contributor rewrote the draft substantively against a reason that rewriting "
        "cannot fix. They are spending time on the wrong thing right now."
    ),
    AdvantageState.UNKNOWN_FAMILY: (
        "The standing decline code is not in our map, so we cannot say whether rewriting "
        "would help. Guessing here is exactly what this tool refuses to do."
    ),
    AdvantageState.INCONSISTENT_REVIEWERS: (
        "Different reviewers gave different reasons on this draft. There is no single "
        "prior decision to carry forward."
    ),
    AdvantageState.ATTEMPT_NOT_MEASURED: (
        "We could not measure what the contributor did since the last decline, so we cannot "
        "tell whether the reason was addressed or ignored. Not knowing is not the same as "
        "knowing nothing happened."
    ),
    AdvantageState.NO_PRECEDENT: (
        "No comparable case has been handled before, so there is nothing to carry forward "
        "from."
    ),
    AdvantageState.REPEATED_REASON_NO_ATTEMPT: (
        "The same reason has been given before and nothing in the draft has moved toward "
        "it. Carrying the prior reason forward is enough."
    ),
    AdvantageState.ROUTINE: (
        "Nothing in the history distinguishes this from cases already handled."
    ),
}


@dataclass(frozen=True)
class Weights:
    """Every constant in the judgment, in one auditable place.

    These are not fitted parameters. Each is a stated position about how much a
    piece of evidence should move the decision, and each is defensible on its
    own. They are grouped here so that a reviewer disputing an escalation can
    point at the specific number they disagree with.
    """

    # Cost: harm if this is handled automatically and that turns out wrong.
    cost_base: float = 0.70  # a contributor is already waiting on every declined draft
    cost_source_existence: float = 0.35  # rewriting can never clear it; effort compounds
    cost_per_extra_decline: float = 0.15
    cost_extra_decline_cap: float = 0.30
    cost_wasted_rewrite: float = 0.20

    # Recoverability: if handling this automatically was wrong, can it be caught
    # and undone? First Reader posts nothing, so the floor is high -- what lowers
    # it is our inability to notice the mistake, and contributor time already burned.
    recoverability_base: float = 0.90
    recoverability_unknown_family: float = 0.45
    recoverability_inconsistent: float = 0.30
    recoverability_no_precedent: float = 0.40
    recoverability_not_measured: float = 0.40
    recoverability_wasted_rewrite: float = 0.35
    recoverability_per_extra_decline: float = 0.08
    recoverability_extra_decline_cap: float = 0.24
    recoverability_floor: float = 0.05

    # InterruptionCost: what one item on a reviewer's queue costs them.
    # Deliberately NOT scaled by queue depth. A busy queue must not be able to
    # silence the cases the queue exists for; depth orders the queue instead.
    interruption_cost: float = 0.20


DEFAULT_WEIGHTS = Weights()


@dataclass(frozen=True)
class EscalationJudgment:
    """The full worked judgment, kept so it can be shown and argued with."""

    escalate: bool
    state: AdvantageState
    advantage: float
    cost: float
    recoverability: float
    interruption_cost: float
    score: float
    reason: str
    policy_rule_id: int | None = None

    @property
    def margin(self) -> float:
        """How far past the threshold. Negative means it stayed off the queue."""
        return self.score - self.interruption_cost


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _no_attempt(blocker: Blocker) -> bool:
    """True when the draft measurably did not move toward the standing reason.

    ``attempt is None`` means *not measured*, and returns False here. An earlier
    version returned True, which quietly turned "we did not look" into "the
    contributor did nothing" -- the exact inference this project exists to stop
    a reviewer from having to make. Unmeasured attempts are handled by
    :attr:`AdvantageState.ATTEMPT_NOT_MEASURED` instead, which escalates.
    """
    attempt = blocker.attempt
    if attempt is None:
        return False
    return (
        attempt.added_sources == 0
        and not attempt.substantive_rewrite
        and not attempt.touched_target_area
    )


def _reviewers_disagreed(card: ReviewCard) -> bool:
    """Different reviewers gave different reasons on the same draft."""
    declines = card.draft.declines
    if len(declines) < 2:
        return False
    codes = {d.code for d in declines}
    decliners = {d.decliner for d in declines}
    return len(codes) > 1 and len(decliners) > 1


def classify(card: ReviewCard, *, precedent_seen: bool) -> AdvantageState:
    """Name the evidence state that sets InterventionAdvantage.

    Args:
        card: The card under consideration.
        precedent_seen: Whether the policy store has observed this pattern
            before. False at cold start, which is why a fresh install sends
            almost everything to a human: autonomy is earned, not assumed.
    """
    if card.wasted_rewrite:
        return AdvantageState.WASTED_REWRITE

    top = card.top_blocker
    if top is not None and top.family is ReasonFamily.UNKNOWN:
        return AdvantageState.UNKNOWN_FAMILY

    if _reviewers_disagreed(card):
        return AdvantageState.INCONSISTENT_REVIEWERS

    if top is not None and top.attempt is None:
        return AdvantageState.ATTEMPT_NOT_MEASURED

    if not precedent_seen:
        return AdvantageState.NO_PRECEDENT

    if top is not None and top.repeat_count >= 2 and _no_attempt(top):
        return AdvantageState.REPEATED_REASON_NO_ATTEMPT

    return AdvantageState.ROUTINE


def _cost(card: ReviewCard, w: Weights) -> float:
    cost = w.cost_base
    top = card.top_blocker
    if top is not None and top.family is ReasonFamily.SOURCE_EXISTENCE:
        cost += w.cost_source_existence
    extra = max(0, card.draft.decline_count - 1)
    cost += min(extra * w.cost_per_extra_decline, w.cost_extra_decline_cap)
    if card.wasted_rewrite:
        cost += w.cost_wasted_rewrite
    return _clamp(cost)


def _recoverability(card: ReviewCard, state: AdvantageState, w: Weights) -> float:
    value = w.recoverability_base
    if state is AdvantageState.UNKNOWN_FAMILY:
        value -= w.recoverability_unknown_family
    if state is AdvantageState.INCONSISTENT_REVIEWERS:
        value -= w.recoverability_inconsistent
    if state is AdvantageState.NO_PRECEDENT:
        value -= w.recoverability_no_precedent
    if state is AdvantageState.ATTEMPT_NOT_MEASURED:
        value -= w.recoverability_not_measured
    if card.wasted_rewrite:
        value -= w.recoverability_wasted_rewrite
    extra = max(0, card.draft.decline_count - 1)
    value -= min(extra * w.recoverability_per_extra_decline, w.recoverability_extra_decline_cap)
    return _clamp(value, low=w.recoverability_floor)


def pattern_key(card: ReviewCard) -> str:
    """What makes two drafts "the same case", before any judgment is made.

    Deliberately independent of :class:`AdvantageState`: precedent has to be
    looked up *in order to* classify, so it cannot be keyed on the answer.

    Repeat counts are bucketed because a rule that only ever fired on exactly
    three repeats would never accumulate the agreements it needs to be learned.
    """
    top = card.top_blocker
    code = top.code if top else "none"
    family = top.family.value if top else "none"
    repeats = top.repeat_count if top else 0
    bucket = "1" if repeats <= 1 else "2" if repeats == 2 else "3+"
    return f"{code}|{family}|repeats={bucket}"


def signature(card: ReviewCard, state: AdvantageState) -> str:
    """The key a policy rule is written against: the pattern plus its state."""
    return f"{state.value}|{pattern_key(card)}"


def judge(
    card: ReviewCard,
    *,
    precedent_seen: bool,
    weights: Weights = DEFAULT_WEIGHTS,
) -> EscalationJudgment:
    """Decide whether this card belongs in front of a reviewer.

    Args:
        card: The card under consideration. ``needs_human`` and
            ``escalation_reason`` on it are ignored; this function produces them.
        precedent_seen: Whether a comparable case has been handled before.
        weights: The constants to judge with.

    Returns:
        The worked judgment, including every intermediate term.
    """
    state = classify(card, precedent_seen=precedent_seen)
    advantage = ADVANTAGE[state]
    cost = _cost(card, weights)
    recoverability = _recoverability(card, state, weights)
    score = advantage * cost * (1.0 - recoverability)
    escalate = score > weights.interruption_cost

    verdict = "Sending this to you" if escalate else "Handling this without you"
    reason = (
        f"{verdict}: {EXPLANATION[state]} "
        f"(advantage {advantage:.2f} x cost {cost:.2f} x irrecoverable "
        f"{1 - recoverability:.2f} = {score:.2f}, threshold {weights.interruption_cost:.2f})"
    )

    return EscalationJudgment(
        escalate=escalate,
        state=state,
        advantage=advantage,
        cost=cost,
        recoverability=recoverability,
        interruption_cost=weights.interruption_cost,
        score=score,
        reason=reason,
    )


#: States a learned relaxation rule may never suppress. A contributor burning
#: their evenings on a reason that rewriting cannot fix is an active harm, not a
#: pattern to become efficient about.
NON_RELAXABLE: frozenset[AdvantageState] = frozenset({AdvantageState.WASTED_REWRITE})


def apply_rule(
    judgment: EscalationJudgment,
    *,
    rule_id: int,
    rule_sentence: str,
    forces_escalation: bool,
) -> EscalationJudgment:
    """Overlay a policy rule on a judgment, preserving the arithmetic underneath.

    Returns the judgment unchanged when a relaxation rule targets a state that
    is not relaxable.
    """
    if not forces_escalation and judgment.state in NON_RELAXABLE:
        return judgment

    verb = "Sending this to you" if forces_escalation else "Handling this without you"
    return replace(
        judgment,
        escalate=forces_escalation,
        policy_rule_id=rule_id,
        reason=(
            f"{verb} under rule #{rule_id}: {rule_sentence} "
            f"(the arithmetic on its own said "
            f"{'escalate' if judgment.escalate else 'no escalation'}, score {judgment.score:.2f})"
        ),
    )
