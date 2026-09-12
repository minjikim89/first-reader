"""Assembling the agent.

One agent. Not a supervisor with sub-agents behind tools.

That is a structural choice, not an aesthetic one. harness-sdk issue #3076
(P1, bug-validated) reports that a nested agent-as-tool fails to resume after an
interrupt, and a nested reviewer agent behind a tool is exactly that shape. A
second reason stands on its own: with one agent there is one transcript, and
"why was this put in front of me" has a single readable answer.

Nothing in this module holds a process open waiting for a person. Escalation
writes a card to a durable queue and the run ends. Issue #3651 describes what
happens when a turn is cut while a tool is in flight -- the ``toolUse`` keeps no
matching ``toolResult`` and the session cannot be reinvoked. Every path here
either completes its tool call or never starts it.

Where the decisions actually live:

* Whether a draft is escalated -- ``escalation.judge``, arithmetic over evidence.
* Whether the agent may escalate -- ``EscalationPolicyGate``, which refuses any
  escalation the judgment did not authorise.
* Whether a tool may run at all -- ``ReversibilityGate``, which reads the grade
  declared on the tool.
* What the comment says -- the model, which is the only part that needs language.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strands import Agent
from strands.interventions import Deny, InterventionHandler, Proceed, Transform
from strands.types.agent import Limits

from ..models import ReviewCard
from . import tools as t
from ._model import resolve_model
from .escalation import DEFAULT_WEIGHTS, Weights
from .interpret import Interpretation, Interpreter
from .policy import DEFAULT_POLICY_PATH, PolicyStore
from .reversibility import Reversibility, grade_of, note_of

SYSTEM_PROMPT = """You are First Reader. You prepare Wikipedia AfC drafts for a human reviewer.

You do not review. You never say whether a draft should be accepted or declined, and you
never assess a source. What you do is carry forward the decisions reviewers already made
and report, factually, what changed since.

For the draft you are given, in this order:
1. get_draft_history
2. analyze_blockers
3. interpret_reasons
4. prepare_comment
5. judge_escalation
6. If and only if judge_escalation said to escalate, call escalate_to_reviewer, passing
   its reason through in your own plain words.

Everything interpret_reasons returns is an estimate about what a reviewer wrote. Treat it
as one. Never let it override a reason family, and never invent a decline code.

Then stop with one sentence saying what you did. Do not call a tool twice.

judge_escalation is binding. If it says not to escalate, escalate_to_reviewer will be
refused, and that is correct: a reviewer's attention is the scarce thing here."""

#: Per-invocation caps. The agent's job is five tool calls; anything past that is
#: a loop, and a loop against a metered provider is the failure mode that costs money.
DEFAULT_LIMITS: Limits = {"turns": 10, "total_tokens": 120_000}


class ReversibilityGate(InterventionHandler):
    """Refuses any tool whose declared grade says it cannot be walked back.

    The gate holds no list of tool names. It reads the grade declared beside each
    tool, and treats an undeclared tool the same as an irreversible one. A new
    tool therefore arrives with no permissions and has to be given them
    explicitly, which is the direction that fails safely.
    """

    name = "first-reader:reversibility"

    def before_tool_call(self, event: Any, **kwargs: Any) -> Any:
        spec = event.selected_tool.tool_spec if event.selected_tool else None
        grade = grade_of(spec)
        tool_name = event.tool_use["name"]

        if grade is None:
            return Deny(
                reason=(
                    f"{tool_name} has no declared reversibility grade, so it is treated as "
                    "irreversible and refused. Declare it in tools.py."
                )
            )

        if grade is Reversibility.IRREVERSIBLE:
            return Deny(
                reason=(
                    f"{tool_name} is irreversible: {note_of(spec)} First Reader does not review "
                    "drafts and does not write to Wikipedia. Prepare the material and hand it to "
                    "a reviewer instead."
                )
            )

        return Proceed()


class EscalationPolicyGate(InterventionHandler):
    """Refuses an escalation the judgment did not authorise.

    Without this, the escalation policy would be a suggestion in a system prompt.
    A model that escalates everything is not obviously broken from the outside;
    it just quietly returns a reviewer's queue to what it was before.
    """

    name = "first-reader:escalation-policy"

    def before_tool_call(self, event: Any, **kwargs: Any) -> Any:
        if event.tool_use["name"] != "escalate_to_reviewer":
            return Proceed()

        title = (event.tool_use.get("input") or {}).get("title", "")
        verdict = t.stored_judgment(event.agent, title)

        if verdict is None:
            return Deny(
                reason=(
                    f"No escalation judgment has been made for {title!r}. "
                    "Call judge_escalation before escalate_to_reviewer."
                )
            )
        if not verdict.get("escalate"):
            return Deny(reason=f"The escalation judgment for {title!r} was: {verdict.get('reason', '')}")
        return Proceed()


class ReviewerCorrections(InterventionHandler):
    """Puts a reviewer's corrections into the agent's next tool call.

    This is the only place in the system where a person edits what the agent
    does rather than approving or rejecting it, and it is the reason
    ``interpret_reasons`` takes a ``known_readings`` argument it is told never
    to fill in itself.

    When a reviewer corrects an estimate -- "that free-text reason is about
    sourcing, not tone" -- the correction is written to the policy store. On the
    next run over that draft, this handler rewrites the arguments of the
    interpretation call before it executes, so the tool uses the reviewer's
    reading and does not ask the model again. The reviewer is not a gate on the
    agent's output; they are upstream of its input.

    ``Transform`` is what makes that possible: it mutates the event in place
    before execution, and handlers registered after this one see the change.
    ``Deny`` could only have thrown the call away, and ``Confirm`` could only
    have asked permission to make it.
    """

    name = "first-reader:reviewer-corrections"

    def __init__(self, policy: PolicyStore) -> None:
        self._policy = policy

    def before_tool_call(self, event: Any, **kwargs: Any) -> Any:
        if event.tool_use["name"] != "interpret_reasons":
            return Proceed()

        arguments = event.tool_use.get("input") or {}
        title = arguments.get("title", "")
        corrections = self._policy.corrections_for(title)
        if not corrections:
            return Proceed()

        settled = {
            code: correction.to_reading(code, correction.evidence)
            for code, correction in corrections.items()
        }
        payload = json.dumps(settled, ensure_ascii=False)

        def apply(evt: Any) -> None:
            evt.tool_use.setdefault("input", {})["known_readings"] = payload

        count = len(settled)
        return Transform(
            apply=apply,
            reason=(
                f"a reviewer has already settled "
                f"{'one reading' if count == 1 else f'{count} readings'} on {title}; "
                "using theirs instead of asking the model again"
            ),
        )


class AuditTrail(InterventionHandler):
    """Records every graded tool call, allowed or refused, in ``agent.state``."""

    name = "first-reader:audit"

    def before_tool_call(self, event: Any, **kwargs: Any) -> Any:
        spec = event.selected_tool.tool_spec if event.selected_tool else None
        grade = grade_of(spec)
        log = event.agent.state.get(t.AUDIT_KEY) or []
        log.append(
            {
                "tool": event.tool_use["name"],
                "grade": grade.value if grade else "undeclared",
                "input": event.tool_use.get("input"),
            }
        )
        event.agent.state.set(t.AUDIT_KEY, log)
        return Proceed()


@dataclass
class ReviewOutcome:
    """What one draft's run produced."""

    title: str
    card: ReviewCard | None
    judgment: dict[str, Any] | None
    queued_at: str | None
    stop_reason: str
    closing_text: str
    denied_calls: list[dict[str, str]] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    backstopped: bool = False
    error: str | None = None
    interpretation: Interpretation = field(default_factory=Interpretation)

    @property
    def ok(self) -> bool:
        """Whether a card was actually produced. A run that errored is not a card."""
        return self.error is None and self.card is not None and self.judgment is not None

    @property
    def escalated(self) -> bool:
        return bool(self.judgment and self.judgment.get("escalate"))

    @property
    def signature(self) -> str:
        return (self.judgment or {}).get("signature", "")


def build_agent(
    *,
    policy: PolicyStore | None = None,
    model: Any = None,
    queue_dir: Path | str = t.DEFAULT_QUEUE_DIR,
    weights: Weights = DEFAULT_WEIGHTS,
    mirror_policy: bool = True,
    interpreter: Interpreter | None = None,
) -> tuple[Agent, PolicyStore]:
    """Assemble the agent and return it with the policy store it is bound to.

    Args:
        policy: The learned policy. Loaded from ``data/policy.json`` if omitted.
        model: A ``strands.models.Model``. Resolved from ``FIRST_READER_MODEL``
            if omitted, which defaults to the offline planner.
        queue_dir: Where escalated cards are filed.
        weights: The escalation constants.
        mirror_policy: Copy the policy into ``agent.state`` so a transcript shows
            what was in force. The file remains the source of truth.
        interpreter: Reads prose. When given, it is used instead of the loop
            model, which lets the two be chosen separately -- see below.
    """
    store = policy if policy is not None else PolicyStore(DEFAULT_POLICY_PATH)
    chosen_model = model if model is not None else resolve_model()

    agent = Agent(
        model=chosen_model,
        tools=[*t.STATIC_TOOLS, t.make_judge_escalation(store, weights)],
        system_prompt=SYSTEM_PROMPT,
        interventions=[
            ReversibilityGate(),
            EscalationPolicyGate(),
            ReviewerCorrections(store),
            AuditTrail(),
        ],
        callback_handler=None,
        name="first-reader",
        description="Prepares AfC drafts for a human reviewer. Does not review.",
    )
    # The loop model and the reading model are separable on purpose. Walking a
    # five-step plan is deterministic work that the offline planner does for
    # nothing; reading a reviewer's prose is not. Pairing
    # ``model=OfflinePlannerModel()`` with a real ``interpreter`` costs one call
    # per draft plus one per free-text reason, instead of one per turn.
    agent._first_reader_interpreter = interpreter or Interpreter(chosen_model)

    agent.state.set(t.QUEUE_DIR_KEY, str(queue_dir))
    if mirror_policy:
        store.mirror_into_state(agent)
    return agent, store


def _denied_calls(agent: Agent) -> list[dict[str, str]]:
    """Tool calls the intervention layer refused, read off the transcript."""
    names: dict[str, str] = {}
    for message in agent.messages:
        for block in message.get("content", []):
            if isinstance(block, dict) and "toolUse" in block:
                names[block["toolUse"]["toolUseId"]] = block["toolUse"]["name"]

    denied: list[dict[str, str]] = []
    for message in agent.messages:
        for block in message.get("content", []):
            if not (isinstance(block, dict) and "toolResult" in block):
                continue
            result = block["toolResult"]
            if result.get("status") != "error":
                continue
            text = " ".join(
                item.get("text", "") for item in result.get("content", []) if isinstance(item, dict)
            )
            if text.startswith("DENIED:"):
                denied.append(
                    {
                        "tool": names.get(result.get("toolUseId", ""), "?"),
                        "reason": text.removeprefix("DENIED:").strip(),
                    }
                )
    return denied


def _first_tool_error(agent: Agent) -> str | None:
    """The first tool error in the transcript, so a failed run says why."""
    for message in agent.messages:
        for block in message.get("content", []):
            if not (isinstance(block, dict) and "toolResult" in block):
                continue
            result = block["toolResult"]
            if result.get("status") != "error":
                continue
            text = " ".join(
                item.get("text", "") for item in result.get("content", []) if isinstance(item, dict)
            ).strip()
            if text and not text.startswith("DENIED:"):
                return text
    return None


def run_review(
    title: str,
    *,
    policy: PolicyStore | None = None,
    model: Any = None,
    queue_dir: Path | str = t.DEFAULT_QUEUE_DIR,
    weights: Weights = DEFAULT_WEIGHTS,
    limits: Limits = DEFAULT_LIMITS,
    interpreter: Interpreter | None = None,
) -> ReviewOutcome:
    """Run one draft through the agent and return what came out.

    A fresh agent per draft: runs stay independent, no transcript grows across
    drafts, and nothing is left alive between them.
    """
    agent, store = build_agent(
        policy=policy,
        model=model,
        queue_dir=queue_dir,
        weights=weights,
        interpreter=interpreter,
    )

    result = agent(f"Prepare this draft for review: <title>{title}</title>", limits=limits)

    card = t.stored_card(agent, title)
    judgment = t.stored_judgment(agent, title)
    interpretation = t.stored_interpretation(agent, title)
    audit = t.audit_log(agent)
    queued = t.queued_path(agent, title)

    # Backstop. If the judgment said escalate and the model did not file the card,
    # file it here. A card that a reviewer should have seen must not be lost to a
    # model that lost the thread.
    backstopped = False
    if judgment and judgment.get("escalate") and queued is None and card is not None:
        card.needs_human = True
        card.escalation_reason = judgment.get("reason", card.escalation_reason)
        queued = str(t.queue_card(card, queue_dir))
        backstopped = True

    denied = _denied_calls(agent)
    error = None
    if card is None or judgment is None:
        error = _first_tool_error(agent) or (
            "the run produced no review card and no escalation judgment"
        )

    return ReviewOutcome(
        title=title,
        card=card,
        judgment=judgment,
        queued_at=queued,
        stop_reason=result.stop_reason,
        closing_text=str(result).strip(),
        denied_calls=denied,
        audit=audit,
        backstopped=backstopped,
        error=error,
        interpretation=interpretation,
    )


def run_batch(titles: list[str], **kwargs: Any) -> list[ReviewOutcome]:
    """Run several drafts, sharing one policy store so learning accumulates."""
    store = kwargs.pop("policy", None) or PolicyStore(DEFAULT_POLICY_PATH)
    return [run_review(title, policy=store, **kwargs) for title in titles]


def record_reviewer_response(
    outcome: ReviewOutcome,
    *,
    agreed: bool,
    policy: PolicyStore,
    note: str = "",
) -> str:
    """Feed a reviewer's response back into the policy.

    Args:
        outcome: The run the reviewer is responding to.
        agreed: True if they agreed with how it was handled, False if they
            overruled it.
        policy: The store to update.
        note: Anything they said, kept as evidence on the rule.

    Returns:
        A line saying what changed, for the reviewer to read back.
    """
    signature = outcome.signature
    if not signature:
        return "Nothing to learn from: this run produced no escalation judgment."

    if agreed:
        rule = policy.record_agreement(signature, note=note)
        if rule is None:
            obs = policy.observation(signature)
            return (
                f"Agreement recorded ({obs.agreements} of {policy.relax_after} needed before "
                f"this pattern is handled without you)."
            )
        return f"Learned a new rule.\n\n{rule.render()}"

    rule = policy.record_reversal(signature, note=note)
    return f"Reversed immediately, on this one case.\n\n{rule.render()}"


def record_reviewer_correction(
    outcome: ReviewOutcome,
    *,
    code: str,
    family: str,
    policy: PolicyStore,
    reading: str = "",
    note: str = "",
) -> str:
    """A reviewer corrects an estimate the model made about a free-text reason.

    This is the co-authoring path. The correction is not filed as feedback for
    someone to read later: on the next run over this draft,
    ``ReviewerCorrections`` rewrites the arguments of the interpretation call
    with the reviewer's answer, and the model is not asked again.

    Args:
        outcome: The run whose reading is being corrected.
        code: The decline code whose reading was wrong.
        family: What the reviewer says it actually is -- one of
            ``source_existence``, ``writing``, ``structural``, ``cannot_tell``.
        policy: The store to record it in.
        reading: The reviewer's own sentence, if they wrote one.
        note: Anything else they said, kept with the correction.

    Returns:
        A line saying what changed, for the reviewer to read back.
    """
    from .interpret import FAMILY_VALUES

    if family not in FAMILY_VALUES:
        raise ValueError(f"family must be one of {FAMILY_VALUES}, got {family!r}")

    existing = outcome.interpretation.families.get(code)
    was = existing.value if existing else "not read"
    evidence = existing.evidence if existing else ""

    correction = policy.record_correction(
        outcome.title,
        code,
        was=was,
        now=family,
        evidence=evidence,
        reading=reading or (existing.reading if existing else ""),
        note=note,
    )
    return (
        f"Recorded. `{code}` on {outcome.title} read as {was.replace('_', ' ')}; "
        f"you say {family.replace('_', ' ')}. The next run uses yours and does not ask "
        f"the model again.\n  key: {correction.key}"
    )


def demo(queue_dir: Path | str = t.DEFAULT_QUEUE_DIR, policy_path: Path | str | None = None) -> str:
    """Run every fixture draft and return a plain-text report.

    Used by ``python -m first_reader.agent.core``.

    Runs against the fixtures, and says so by setting the source explicitly for
    the duration rather than relying on a default. The production default is the
    collected dataset and then the live API; ``run_review`` on a real title needs
    no such setting.
    """
    import os

    from . import _stubs

    previous = os.environ.get("FIRST_READER_SOURCE")
    os.environ["FIRST_READER_SOURCE"] = "stub"
    try:
        return _demo_body(_stubs.titles(), queue_dir, policy_path)
    finally:
        if previous is None:
            os.environ.pop("FIRST_READER_SOURCE", None)
        else:
            os.environ["FIRST_READER_SOURCE"] = previous


def _demo_body(
    titles: list[str], queue_dir: Path | str, policy_path: Path | str | None
) -> str:
    store = PolicyStore(policy_path or DEFAULT_POLICY_PATH)
    lines: list[str] = []
    for outcome in run_batch(titles, policy=store, queue_dir=queue_dir):
        verdict = "-> reviewer" if outcome.escalated else "   handled alone"
        if not outcome.ok:
            verdict = "!! failed   "
        state = (outcome.judgment or {}).get("state", "?")
        score = (outcome.judgment or {}).get("score", 0)
        lines.append(f"{verdict}  {outcome.title}")
        lines.append(f"      state={state} score={score} stop={outcome.stop_reason}")
        for denial in outcome.denied_calls:
            lines.append(f"      refused {denial['tool']}: {denial['reason'][:90]}")
        if outcome.queued_at:
            lines.append(f"      filed at {outcome.queued_at}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(demo())
