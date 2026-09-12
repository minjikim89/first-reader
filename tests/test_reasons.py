"""Completeness of the decline-code table.

These tests are about coverage and honesty, not about whether any particular
classification is "right" -- that judgment is recorded in the `# why:` comments
in reasons.py and is reviewable by a human.
"""

from __future__ import annotations

import collections
from pathlib import Path

import pytest

from first_reader.analysis import reasons as R
from first_reader.models import REASON_FAMILY, Decline, ReasonFamily

# Every code the AFCH decline dropdown offers, read off
# src/templates/tpl-submissions.html (the `#declineReason` <select>).
AFCH_DROPDOWN_CODES = frozenset(
    {
        # Notability
        "astro", "athlete", "bio", "book", "corp", "creative", "event", "film",
        "geo", "list", "music", "neo", "number", "prof", "school", "species",
        "web", "nn",
        # Verifiability
        "ns", "v", "ilc", "med",
        # Invalid submissions
        "blank", "cat", "lang", "redirect", "test",
        # Content issues
        "ai", "blp", "cv", "context", "dict", "ecr", "joke", "news", "not",
        "plot", "van",
        # Prose issues
        "adv", "essay", "npov", "resume",
        # Duplicate articles
        "exists", "mergeto", "dup",
        # Other
        "reason",
    }
)

# Values in Template:AfC submission/doc templatedata `suggestedvalues`. This is
# an autocomplete hint, NOT a spec: it omits `v` (216 real declines), `adv` (82),
# `ns` (27) and `lang` (7), and it lists `afd`, which has no branch in the
# rendering switch at all and never occurs.
TEMPLATEDATA_SUGGESTED = frozenset(
    {
        "nn", "reason", "neo", "web", "prof", "academic", "athlete", "sport",
        "music", "band", "film", "book", "event", "corp", "org", "bio", "list",
        "med", "geo", "species", "astro", "number", "creative", "school", "cat",
        "blank", "notenglish", "test", "redirect", "van", "ilc", "blp", "cv",
        "source", "rs", "not", "news", "oneevent", "dict", "plot", "joke", "ai",
        "llm", "resume", "ecr", "nosource", "essay", "npov", "advert", "spam",
        "exists", "duplicate", "context", "mergeto", "afd",
    }
)

# AFCH's reject dropdown. Same template parameter, different question.
REJECT_CODES = frozenset({"n", "j", "e"})

# Aliases that Template:AfC submission/comments still accepts but AFCH no longer
# offers. Old drafts carry these, so they must resolve.
TEMPLATE_ALIASES = frozenset(
    {
        "rs", "source", "nosource", "org", "inc", "sport", "m", "band",
        "academic", "llm", "spam", "advert", "notenglish", "cite", "footnote",
        "empty", "duplicate", "merge", "hoax", "oneevent", "d", "cv-n", "cv-c",
        "cv-cleaned",
    }
)


def test_every_afch_dropdown_code_is_in_the_table():
    missing = sorted(AFCH_DROPDOWN_CODES - R.KNOWN_CODES)
    assert missing == [], f"AFCH offers these codes but the table has no entry: {missing}"


def test_every_template_alias_resolves():
    missing = sorted(TEMPLATE_ALIASES - R.KNOWN_CODES)
    assert missing == [], f"old drafts can carry these codes but they do not resolve: {missing}"


def test_alias_shares_its_canonical_family():
    for alias, canonical in [
        ("nosource", "ns"), ("llm", "ai"), ("advert", "adv"), ("hoax", "joke"),
        ("oneevent", "news"), ("academic", "prof"), ("band", "music"),
        ("notenglish", "lang"), ("duplicate", "dup"), ("source", "v"),
        ("footnote", "ilc"), ("d", "dict"), ("cv-n", "cv"),
    ]:
        assert R.canonical_code(alias) == canonical
        assert R.family_for(alias) is R.family_for(canonical)


def test_every_templatedata_suggested_value_resolves():
    missing = sorted(TEMPLATEDATA_SUGGESTED - R.KNOWN_CODES)
    assert missing == [], f"templatedata suggests these but they do not resolve: {missing}"


def test_table_has_no_invented_codes():
    known_canonical = {info.code for info in R.REASONS}
    sources = AFCH_DROPDOWN_CODES | TEMPLATE_ALIASES | TEMPLATEDATA_SUGGESTED | REJECT_CODES
    invented = sorted(known_canonical - sources)
    assert invented == [], f"table contains codes found in no source: {invented}"


def test_afch_spelling_and_templatedata_spelling_both_resolve_to_one_family():
    """templatedata documents `advert`/`duplicate`; AFCH writes `adv`/`dup`.

    In 1,472 real declines, `adv` occurs 82 times and `advert` never. Dropping
    either spelling would be wrong; they must simply agree.
    """
    for afch_spelling, templatedata_spelling in [
        ("adv", "advert"), ("dup", "duplicate"), ("v", "source"),
        ("ns", "nosource"), ("lang", "notenglish"), ("ai", "llm"),
    ]:
        assert R.family_for(afch_spelling) is R.family_for(templatedata_spelling)
        assert R.family_for(afch_spelling) is not ReasonFamily.UNKNOWN


def test_afd_is_documented_but_unrenderable_so_it_stays_unknown():
    """`afd` is in templatedata but has no branch in the /comments #switch."""
    assert R.family_for("afd") is ReasonFamily.UNKNOWN
    assert R.describe("afd") is not None, "afd should be documented, not merely absent"
    assert "no branch" in R.UNKNOWN_RATIONALE["afd"].lower() or "switch" in R.UNKNOWN_RATIONALE["afd"].lower()


def test_every_entry_carries_a_verbatim_description_and_a_rationale():
    for info in R.REASONS:
        assert info.description.strip(), f"{info.code} has no quoted description"
        assert info.source in {"afch", "template", "templatedata"}, f"{info.code} has no provenance"
        assert info.rationale.strip(), f"{info.code} has no recorded rationale"


def test_unknown_set_is_exactly_what_we_declined_to_classify():
    assert R.DECLINE_UNKNOWN_CODES == ("afd", "blp", "cv", "cv-cleaned", "dict", "not", "reason")
    assert R.UNKNOWN_CODES == (
        "afd", "blp", "cv", "cv-cleaned", "dict", "e", "j", "n", "not", "reason",
    )
    for code in R.UNKNOWN_CODES:
        assert R.family_for(code) is ReasonFamily.UNKNOWN
        assert R.fixable_by_rewrite(code) is None, "UNKNOWN must be None, never False"
        assert R.UNKNOWN_RATIONALE[code].strip(), f"{code} is UNKNOWN with no stated reason"


def test_coverage_report_numbers():
    report = R.coverage_report()
    assert report["canonical_codes"] == 51
    assert report["decline_codes"] == 48
    assert report["reject_codes"] == 3
    assert report["codes_including_aliases"] == 74
    assert report["by_family"] == {
        "source_existence": 23,
        "writing": 8,
        "structural": 10,
        "unknown": 10,
    }
    # 7 of 48 decline codes unclassified == 85.4% of the code table.
    assert len(report["unknown_decline_codes"]) == 7


def test_unmapped_code_is_unknown_not_a_guess():
    assert R.family_for("banana") is ReasonFamily.UNKNOWN
    assert R.fixable_by_rewrite("banana") is None
    assert R.describe("banana") is None


def test_reject_codes_are_not_silently_treated_as_declines():
    """`n`/`j`/`e` land in the same template slot but mean something else.

    They occur in real parsed histories (`n` 23 times, `e` 5 times in the
    dataset), so they are documented rather than merely absent -- but a reject
    ends the submission, so "does rewriting fix this" was never asked.
    """
    assert R.REJECT_ONLY_CODES == REJECT_CODES
    for code in R.REJECT_ONLY_CODES:
        assert R.family_for(code) is ReasonFamily.UNKNOWN
        assert R.describe(code).kind == "reject"
        assert R.fixable_by_rewrite(code) is None


def test_codes_are_normalised_the_way_the_template_normalises_them():
    assert R.normalize_code(" NN ") == "nn"
    assert R.family_for("  CORP ") is ReasonFamily.SOURCE_EXISTENCE
    assert R.family_for(None) is ReasonFamily.UNKNOWN
    assert R.family_for("") is ReasonFamily.UNKNOWN


def test_models_map_is_exactly_what_the_table_generates():
    """`Decline.family` reads models.REASON_FAMILY, so it must not drift."""
    assert REASON_FAMILY == R.as_family_map()


def test_original_seed_classifications_survived_completion():
    """Completing the map must not have reclassified anything already seeded."""
    seed = {
        "nn": ReasonFamily.SOURCE_EXISTENCE, "v": ReasonFamily.SOURCE_EXISTENCE,
        "bio": ReasonFamily.SOURCE_EXISTENCE, "corp": ReasonFamily.SOURCE_EXISTENCE,
        "event": ReasonFamily.SOURCE_EXISTENCE, "music": ReasonFamily.SOURCE_EXISTENCE,
        "athlete": ReasonFamily.SOURCE_EXISTENCE, "film": ReasonFamily.SOURCE_EXISTENCE,
        "web": ReasonFamily.SOURCE_EXISTENCE,
        "ai": ReasonFamily.WRITING, "adv": ReasonFamily.WRITING,
        "npov": ReasonFamily.WRITING, "essay": ReasonFamily.WRITING,
        "resume": ReasonFamily.WRITING, "context": ReasonFamily.WRITING,
        "lang": ReasonFamily.WRITING, "ilc": ReasonFamily.WRITING,
        "exists": ReasonFamily.STRUCTURAL, "dup": ReasonFamily.STRUCTURAL,
        "blank": ReasonFamily.STRUCTURAL, "test": ReasonFamily.STRUCTURAL,
        "redirect": ReasonFamily.STRUCTURAL,
    }
    for code, family in seed.items():
        assert R.family_for(code) is family, f"table reclassified seeded code {code}"


def test_Decline_family_uses_the_full_table():
    assert "plot" in REASON_FAMILY, "models.REASON_FAMILY is still the partial seed"
    from datetime import datetime

    def decline(code):
        t = datetime(2026, 1, 1)
        return Decline(code=code, secondary_code=None, decliner="R", submitted_at=t, declined_at=t, raw="")

    assert decline("plot").family is ReasonFamily.SOURCE_EXISTENCE
    assert decline("plot").fixable_by_rewrite is False
    assert decline("nosource").family is ReasonFamily.SOURCE_EXISTENCE
    assert decline("cv").family is ReasonFamily.UNKNOWN
    assert decline("cv").fixable_by_rewrite is None


@pytest.mark.parametrize(
    "code,expected",
    [
        # the whole point of the SOURCE_EXISTENCE family
        ("nn", False), ("v", False), ("ns", False), ("med", False),
        ("plot", False), ("news", False), ("list", False),
        # rewriting genuinely helps here
        ("ai", True), ("npov", True), ("context", True), ("lang", True), ("ilc", True),
        # nothing on this page will help
        ("exists", False), ("ecr", False), ("cat", False), ("van", False),
    ],
)
def test_fixable_by_rewrite(code, expected):
    assert R.fixable_by_rewrite(code) is expected


# ---------------------------------------------------------------------------
# Measured against the collected corpus rather than against our own fixtures.
# These numbers are the project's central empirical claim, so they are pinned:
# if the table changes, the effect on 1,472 real declines shows up here.
# ---------------------------------------------------------------------------

DATASET = Path(__file__).resolve().parent.parent / "data" / "dataset.jsonl"
needs_dataset = pytest.mark.skipif(not DATASET.exists(), reason="data/dataset.jsonl not collected")


@pytest.fixture(scope="module")
def corpus():
    from first_reader.wiki.collect import load_dataset

    return load_dataset(DATASET)


@needs_dataset
def test_full_table_cuts_unknown_declines_by_more_than_a_third(corpus):
    """The partial seed left 11.7% of real declines unclassified; the full table leaves 7.3%."""
    codes = [(d.code or "").strip().lower() for h in corpus for d in h.declines]
    assert len(codes) == 1472, "corpus changed; re-measure before editing the numbers below"

    seed_only = {
        "nn", "v", "bio", "corp", "event", "music", "athlete", "film", "web",
        "ai", "adv", "npov", "essay", "resume", "context", "lang", "ilc",
        "exists", "dup", "blank", "test", "redirect",
    }
    before = sum(1 for c in codes if c not in seed_only)
    after = sum(1 for c in codes if R.family_for(c) is ReasonFamily.UNKNOWN)

    assert before == 172
    assert after == 108
    assert after / len(codes) < 0.08


@needs_dataset
def test_what_remains_unknown_is_free_text_or_a_reject_not_a_refusal_to_look(corpus):
    """Of 108 unresolved declines, 100 are unclassifiable in principle."""
    unresolved = collections.Counter(
        (d.code or "").strip().lower() or "<empty>"
        for h in corpus
        for d in h.declines
        if R.family_for(d.code) is ReasonFamily.UNKNOWN
    )
    assert sum(unresolved.values()) == 108

    # `reason` is a free-text decline: the meaning lives in the reviewer's prose,
    # not in the code. `n` and `e` are rejects, which end the submission.
    inherently = unresolved["reason"] + sum(unresolved[c] for c in R.REJECT_ONLY_CODES)
    assert inherently == 100

    # Only these are real decline codes we looked at and declined to classify.
    ours = {c: n for c, n in unresolved.items() if c not in R.REJECT_ONLY_CODES and c != "reason"}
    assert ours == {"not": 2, "cv-cleaned": 2, "dict": 2, "blp": 1, "<empty>": 1}
    assert sum(ours.values()) / 1472 < 0.006


@needs_dataset
def test_the_project_thesis_holds_on_the_corpus(corpus):
    """~60% of declines are reasons that rewriting cannot fix.

    This is what makes the tool worth building: the most common thing a
    reviewer says is the thing a contributor cannot fix by editing prose.
    """
    families = collections.Counter(
        R.family_for(code)
        for h in corpus
        for d in h.declines
        for code in (d.code, d.secondary_code)
        if (code or "").strip()
    )
    total = sum(families.values())
    assert total == 1731
    assert families[ReasonFamily.SOURCE_EXISTENCE] == 1055
    assert 0.60 <= families[ReasonFamily.SOURCE_EXISTENCE] / total <= 0.62
    assert families[ReasonFamily.WRITING] / total == pytest.approx(0.307, abs=0.01)


@needs_dataset
def test_only_the_afch_spellings_actually_occur(corpus):
    """Live evidence for keeping `adv`/`dup`/`v`/`ns` as the canonical codes.

    Template:AfC submission/doc templatedata documents the other spellings, but
    no draft uses them. Replacing the AFCH spellings with the documented ones
    would send these declines to UNKNOWN.
    """
    seen = collections.Counter(
        (code or "").strip().lower()
        for h in corpus
        for d in h.declines
        for code in (d.code, d.secondary_code)
        if (code or "").strip()
    )
    assert seen["adv"] == 82 and seen["advert"] == 0
    assert seen["v"] == 216 and seen["source"] == 0 and seen["rs"] == 0
    assert seen["ns"] == 27 and seen["nosource"] == 0
    assert seen["dup"] == 0 and seen["duplicate"] == 0
    assert seen["afd"] == 0, "documented in templatedata, absent from the switch, never used"
