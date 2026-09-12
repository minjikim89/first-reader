"""The policy that decides what stops being escalated, and how it changes.

Two commitments shape this file.

**The rules are sentences.** A learned policy that exists only as a weight
vector cannot be argued with. A reviewer who disagrees with an automatic
handling has to be able to read the rule that caused it, see the evidence it
was learned from, and switch it off. So a rule is a sentence, its evidence, and
three switches: active, locked, disabled.

**Learning is asymmetric.** Widening what the agent handles alone takes
``relax_after`` agreements (5 by default). Narrowing it takes one reversal, and
takes effect immediately. This is not a tuning choice. The two errors are not
symmetric: an unnecessary escalation costs a reviewer a few seconds, and a
wrongly automated case costs a contributor weeks of work on the wrong problem.
A learner that treated them as equal evidence would drift toward the cheaper
error for the machine.

After a reversal, agreements keep accumulating but no relaxation rule is
created while the tightening rule stands. Getting autonomy back over a pattern
a human has already overruled requires a human to disable that rule.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_POLICY_PATH = Path("data/policy.json")
DEFAULT_RELAX_AFTER = 5
STATE_KEY = "first_reader:policy"
SCHEMA_VERSION = 1

RELAX = "relax"
TIGHTEN = "tighten"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Rule:
    """One policy rule, written so a human can read and overrule it.

    Attributes:
        id: Stable identifier, cited in every escalation reason it decides.
        kind: ``relax`` widens what runs without a reviewer; ``tighten`` forces
            a reviewer back in.
        sentence: The rule itself, in plain words.
        signature: The pattern it applies to.
        evidence: Why it exists, in plain words.
        active: Whether it currently decides anything.
        locked: A locked rule cannot be switched off by learning. Only a human
            can unlock it. Use this to pin a rule a team has agreed on.
    """

    id: int
    kind: str
    sentence: str
    signature: str
    evidence: str
    active: bool = True
    locked: bool = False
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @property
    def status(self) -> str:
        """One word for the UI: active, locked or disabled."""
        if not self.active:
            return "disabled"
        return "locked" if self.locked else "active"

    def render(self) -> str:
        """The audit view of this rule."""
        return (
            f"Rule #{self.id}  \"{self.sentence}\"\n"
            f"  evidence: {self.evidence}\n"
            f"  status: {self.status}"
        )


@dataclass
class Observation:
    """What has been seen for one pattern."""

    seen: int = 0
    agreements: int = 0
    reversals: int = 0
    last_seen: str = ""


@dataclass
class Correction:
    """A reviewer's edit to an estimate the model made.

    Two things at once. It is the answer -- next time this draft's free-text
    reason comes up, use the reviewer's reading and do not ask the model again.
    And it is a training signal: ``was`` to ``now`` is a recorded case of the
    model reading prose the way a reviewer would not.

    Keyed per draft and code. A reviewer correcting one reviewer's particular
    wording has not told us anything general about other wording, and pretending
    otherwise would be the same over-reach this project exists to avoid.
    """

    key: str
    was: str
    now: str
    evidence: str
    reading: str
    note: str = ""
    at: str = field(default_factory=_now)

    def to_reading(self, subject: str, source_text: str) -> dict[str, Any]:
        """The shape ``interpret_card`` expects for a settled reading."""
        return {
            "kind": "reason_family",
            "subject": subject,
            "value": self.now,
            "evidence": self.evidence,
            "reading": self.reading,
            "source_text": source_text,
            "model_id": "",
            "estimated": False,
            "corrected_by_reviewer": True,
            "original_value": self.was,
        }


# --- sentence construction -------------------------------------------------

_FAMILY_WORDS = {
    "source_existence": "a reason that rewriting cannot fix",
    "writing": "a reason that rewriting can fix",
    "structural": "a reason that needs a different action entirely",
    "unknown": "a reason whose family we do not know",
}

_STATE_WORDS = {
    "wasted_rewrite": "the contributor rewrote substantively against it",
    "unknown_family": "its code is not in our map",
    "inconsistent_reviewers": "reviewers had given different reasons",
    "no_precedent": "no comparable case had been handled before",
    "repeated_reason_no_attempt": "nothing in the draft had moved toward it",
    "routine": "nothing distinguished it from cases already handled",
}


def parse_signature(signature: str) -> dict[str, str]:
    """Split a signature into its parts. Unknown shapes degrade, never raise."""
    parts = signature.split("|")
    out = {"state": "", "code": "", "family": "", "repeats": ""}
    keys = ["state", "code", "family", "repeats"]
    for key, value in zip(keys, parts):
        out[key] = value
    out["repeats"] = out["repeats"].removeprefix("repeats=")
    return out


def _article(word: str) -> str:
    """"a" or "an" for a decline code. These sentences are read by people."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def _repeat_words(repeats: str) -> str:
    if repeats == "3+":
        return "has been given three or more times"
    if repeats == "2":
        return "has been given twice"
    return "has been given once"


def relax_sentence(signature: str) -> str:
    """The natural-language form of a relaxation rule."""
    p = parse_signature(signature)
    family = _FAMILY_WORDS.get(p["family"], "a reason of an unrecognised kind")
    state = _STATE_WORDS.get(p["state"], "the pattern matched")
    code = p["code"] or "standing"
    return (
        f"When {_article(code)} {code} decline -- {family} -- {_repeat_words(p['repeats'])} "
        f"and {state}, carry the prior reason forward with a summary and do not "
        f"put it in front of a reviewer."
    )


def tighten_sentence(signature: str) -> str:
    """The natural-language form of a tightening rule."""
    p = parse_signature(signature)
    family = _FAMILY_WORDS.get(p["family"], "a reason of an unrecognised kind")
    code = p["code"] or "standing"
    return (
        f"When {_article(code)} {code} decline -- {family} -- "
        f"{_repeat_words(p['repeats'])}, always put it in front of a reviewer. "
        f"A reviewer overruled the automatic handling of this exact pattern."
    )


# --- store -----------------------------------------------------------------


class PolicyStore:
    """The learned policy. The JSON file is the source of truth.

    ``agent.state`` can carry a mirror of it for the duration of a run (see
    :meth:`mirror_into_state`), but the mirror is never read back as authority:
    ``agent.state`` does not outlive the process, and a policy that a reviewer
    cannot open in a text editor is not auditable.
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_POLICY_PATH,
        *,
        relax_after: int = DEFAULT_RELAX_AFTER,
        autosave: bool = True,
    ) -> None:
        self.path = Path(path)
        self.relax_after = relax_after
        self.autosave = autosave
        self.rules: list[Rule] = []
        self.observations: dict[str, Observation] = {}
        self.corrections: dict[str, Correction] = {}
        self._next_rule_id = 1
        self.load()

    # -- persistence --

    def load(self) -> None:
        """Read the policy file. A missing file is an empty policy, not an error."""
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.relax_after = data.get("relax_after", self.relax_after)
        self._next_rule_id = data.get("next_rule_id", 1)
        self.rules = [Rule(**r) for r in data.get("rules", [])]
        self.observations = {
            key: Observation(**value) for key, value in data.get("observations", {}).items()
        }
        self.corrections = {
            key: Correction(**value) for key, value in data.get("corrections", {}).items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SCHEMA_VERSION,
            "relax_after": self.relax_after,
            "next_rule_id": self._next_rule_id,
            "rules": [asdict(r) for r in self.rules],
            "observations": {k: asdict(v) for k, v in self.observations.items()},
            "corrections": {k: asdict(v) for k, v in self.corrections.items()},
        }

    def save(self) -> None:
        """Write the policy file atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def _maybe_save(self) -> None:
        if self.autosave:
            self.save()

    def mirror_into_state(self, agent: Any) -> None:
        """Copy the policy into ``agent.state`` for inspection during a run.

        The mirror is written, never read as authority. ``agent.state`` validates
        JSON-serialisability on assignment, which this dict satisfies.
        """
        agent.state.set(STATE_KEY, self.to_dict())

    # -- reading --

    def precedent_seen(self, signature: str) -> bool:
        """Whether a comparable case has been handled before."""
        return self.observations.get(signature, Observation()).seen > 0

    def observation(self, signature: str) -> Observation:
        return self.observations.get(signature, Observation())

    def active_rule_for(self, signature: str) -> Rule | None:
        """The rule that currently decides this pattern.

        A tightening rule always wins over a relaxation rule: the asymmetry is
        in the resolution too, not only in the learning.
        """
        matches = [r for r in self.rules if r.signature == signature and r.active]
        for rule in matches:
            if rule.kind == TIGHTEN:
                return rule
        return matches[0] if matches else None

    def rule_by_id(self, rule_id: int) -> Rule | None:
        return next((r for r in self.rules if r.id == rule_id), None)

    # -- writing --

    def observe(self, signature: str) -> None:
        """Record that this pattern was encountered.

        Precedent is counted separately from agreement. Seeing a pattern is not
        evidence that handling it automatically was right.
        """
        obs = self.observations.setdefault(signature, Observation())
        obs.seen += 1
        obs.last_seen = _now()
        self._maybe_save()

    def record_agreement(self, signature: str, *, note: str = "") -> Rule | None:
        """A reviewer agreed with how this pattern was handled.

        Returns the relaxation rule if this agreement was the one that created
        it, otherwise None. Creates nothing while a tightening rule stands.
        """
        obs = self.observations.setdefault(signature, Observation())
        obs.agreements += 1
        obs.last_seen = _now()

        blocked = any(r.signature == signature and r.kind == TIGHTEN and r.active for r in self.rules)
        already = any(r.signature == signature and r.kind == RELAX and r.active for r in self.rules)

        rule: Rule | None = None
        if not blocked and not already and obs.agreements >= self.relax_after:
            evidence = (
                f"{obs.agreements} comparable cases, every one agreed with by a reviewer"
                f"{'; ' + note if note else ''}."
            )
            rule = self._add_rule(RELAX, relax_sentence(signature), signature, evidence)

        self._maybe_save()
        return rule

    def record_reversal(self, signature: str, *, note: str = "") -> Rule:
        """A reviewer overruled how this pattern was handled.

        Takes effect on the first reversal. Any active relaxation rule for the
        pattern is switched off unless it is locked, the agreement count is
        reset, and a tightening rule is put in its place.
        """
        obs = self.observations.setdefault(signature, Observation())
        obs.reversals += 1
        obs.agreements = 0
        obs.last_seen = _now()

        revoked: list[int] = []
        locked_survivors: list[int] = []
        for rule in self.rules:
            if rule.signature == signature and rule.kind == RELAX and rule.active:
                if rule.locked:
                    locked_survivors.append(rule.id)
                    continue
                rule.active = False
                rule.updated_at = _now()
                revoked.append(rule.id)

        existing = next(
            (r for r in self.rules if r.signature == signature and r.kind == TIGHTEN), None
        )
        evidence = f"Reversed by a reviewer on {_now()[:10]}"
        if revoked:
            evidence += f"; revoked rule {', '.join(f'#{i}' for i in revoked)}"
        if locked_survivors:
            evidence += (
                f"; rule {', '.join(f'#{i}' for i in locked_survivors)} is locked and was left "
                f"standing for a human to resolve"
            )
        if note:
            evidence += f"; {note}"
        evidence += "."

        if existing is not None:
            existing.active = True
            existing.evidence = evidence
            existing.updated_at = _now()
            rule = existing
        else:
            rule = self._add_rule(TIGHTEN, tighten_sentence(signature), signature, evidence)

        self._maybe_save()
        return rule

    def _add_rule(self, kind: str, sentence: str, signature: str, evidence: str) -> Rule:
        rule = Rule(
            id=self._next_rule_id,
            kind=kind,
            sentence=sentence,
            signature=signature,
            evidence=evidence,
        )
        self._next_rule_id += 1
        self.rules.append(rule)
        return rule

    # -- reviewer corrections to estimates --

    @staticmethod
    def correction_key(title: str, code: str) -> str:
        return f"{title}|{code}"

    def record_correction(
        self,
        title: str,
        code: str,
        *,
        was: str,
        now: str,
        evidence: str,
        reading: str,
        note: str = "",
    ) -> Correction:
        """A reviewer corrected an estimate. Their answer stands from now on."""
        key = self.correction_key(title, code)
        correction = Correction(
            key=key, was=was, now=now, evidence=evidence, reading=reading, note=note
        )
        self.corrections[key] = correction
        self._maybe_save()
        return correction

    def corrections_for(self, title: str) -> dict[str, Correction]:
        """Every correction a reviewer has made on one draft, keyed by code."""
        prefix = f"{title}|"
        return {
            key[len(prefix) :]: correction
            for key, correction in self.corrections.items()
            if key.startswith(prefix)
        }

    def correction_record(self) -> list[dict[str, str]]:
        """Every correction, oldest first. The record of where the model was wrong."""
        return [
            {"key": c.key, "was": c.was, "now": c.now, "at": c.at, "note": c.note}
            for c in sorted(self.corrections.values(), key=lambda c: c.at)
        ]

    # -- human controls --

    def disable(self, rule_id: int) -> Rule:
        """Switch a rule off. Refuses on a locked rule; unlock it first."""
        rule = self._require(rule_id)
        if rule.locked:
            raise ValueError(f"rule #{rule_id} is locked; unlock it before disabling")
        rule.active = False
        rule.updated_at = _now()
        self._maybe_save()
        return rule

    def enable(self, rule_id: int) -> Rule:
        rule = self._require(rule_id)
        rule.active = True
        rule.updated_at = _now()
        self._maybe_save()
        return rule

    def lock(self, rule_id: int) -> Rule:
        """Pin a rule so learning cannot switch it off."""
        rule = self._require(rule_id)
        rule.locked = True
        rule.updated_at = _now()
        self._maybe_save()
        return rule

    def unlock(self, rule_id: int) -> Rule:
        rule = self._require(rule_id)
        rule.locked = False
        rule.updated_at = _now()
        self._maybe_save()
        return rule

    def _require(self, rule_id: int) -> Rule:
        rule = self.rule_by_id(rule_id)
        if rule is None:
            raise KeyError(f"no rule #{rule_id}")
        return rule

    def render(self) -> str:
        """The whole policy as an audit log."""
        if not self.rules:
            return "No rules learned yet. Every pattern goes to a reviewer."
        return "\n\n".join(r.render() for r in self.rules)
