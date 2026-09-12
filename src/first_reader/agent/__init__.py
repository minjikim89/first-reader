"""The agent: decides what a reviewer sees, and never reviews.

Public surface:
    ``build_agent``   -- assemble the single agent and its policy store.
    ``run_review``    -- one draft, start to finish.
    ``run_batch``     -- several drafts against one policy store.
    ``record_reviewer_response`` -- feed a reviewer's agreement or reversal back.
    ``PolicyStore``   -- the learned rules, as readable sentences.
    ``judge``         -- the escalation judgment on its own.
"""

from .core import (
    AuditTrail,
    EscalationPolicyGate,
    ReversibilityGate,
    ReviewerCorrections,
    ReviewOutcome,
    build_agent,
    demo,
    record_reviewer_correction,
    record_reviewer_response,
    run_batch,
    run_review,
)
from .interpret import Interpretation, Interpreter, Reading, resolve_interpreter
from .escalation import (
    ADVANTAGE,
    AdvantageState,
    EscalationJudgment,
    Weights,
    judge,
    pattern_key,
    signature,
)
from .policy import PolicyStore, Rule
from .reversibility import Reversibility

__all__ = [
    "ADVANTAGE",
    "AdvantageState",
    "AuditTrail",
    "EscalationJudgment",
    "EscalationPolicyGate",
    "Interpretation",
    "Interpreter",
    "PolicyStore",
    "Reading",
    "Reversibility",
    "ReversibilityGate",
    "ReviewerCorrections",
    "ReviewOutcome",
    "Rule",
    "Weights",
    "build_agent",
    "demo",
    "judge",
    "pattern_key",
    "record_reviewer_correction",
    "record_reviewer_response",
    "resolve_interpreter",
    "run_batch",
    "run_review",
    "signature",
]
