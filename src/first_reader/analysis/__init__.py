"""Analysis layer: decline codes, attempt signals, and blocker ordering.

Importing this package installs the full AfC decline-code table into
`models.REASON_FAMILY`, so `Decline.family` agrees with `reasons.family_for`
instead of falling back to the partial seed shipped in models.py.
"""

from .attempt import count_refs, detect_attempt, extract_domains, prose_similarity, strip_markup
from .blockers import build_blockers
from .reasons import (
    BY_CODE,
    KNOWN_CODES,
    REASONS,
    REJECT_ONLY_CODES,
    UNKNOWN_CODES,
    UNKNOWN_RATIONALE,
    ReasonInfo,
    canonical_code,
    coverage_report,
    describe,
    family_for,
    fixable_by_rewrite,
    normalize_code,
)

__all__ = [
    "BY_CODE",
    "KNOWN_CODES",
    "REASONS",
    "REJECT_ONLY_CODES",
    "ReasonInfo",
    "UNKNOWN_CODES",
    "UNKNOWN_RATIONALE",
    "build_blockers",
    "canonical_code",
    "count_refs",
    "coverage_report",
    "describe",
    "detect_attempt",
    "extract_domains",
    "family_for",
    "fixable_by_rewrite",
    "normalize_code",
    "prose_similarity",
    "strip_markup",
]
