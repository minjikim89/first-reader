"""AfC decline reason codes and their families.

The table below is not invented. It was extracted on 2026-09-07 from the two
places that actually define these codes:

  1. https://raw.githubusercontent.com/wikimedia-gadgets/afc-helper/master/
     src/templates/tpl-submissions.html
     -- the `#declineReason` <select>, i.e. the exact 46 codes a reviewer can
     pick in AFCH, with the label AFCH shows next to each one.
  2. https://en.wikipedia.org/wiki/Template:AfC_submission/comments (wikitext)
     -- the `#switch` that renders the decline notice. It carries 47 code
     branches including aliases AFCH no longer offers (`nosource`, `llm`,
     `advert`, `hoax`, `oneevent`, `academic`, `band`, `cv-cleaned`, ...) but
     which still sit in old drafts.

A third list exists and is NOT used as the spec: the `suggestedvalues` array in
Template:AfC submission/doc templatedata. It is an editor autocomplete hint, and
measuring it against 1,472 real declines shows it is not authoritative:

    code       templatedata   real declines
    v          absent         216   <- 3rd most common code overall
    adv        absent          82
    ns         absent          27
    lang       absent           7
    advert     present          0
    duplicate  present          0
    afd        present          0   <- and no branch in the #switch at all

So `adv`/`dup`/`v`/`ns`/`lang` are kept as canonical codes with their
templatedata spellings as aliases, not the other way round. Both spellings
resolve to the same family, so nothing is lost either way -- but dropping the
AFCH spellings would send 300+ real declines to UNKNOWN.

`description` on each entry is quoted verbatim from source 1 where AFCH offers
the code, and from source 2's first sentence otherwise. `# why:` comments record
our own one-line reason for the family, always anchored to that quoted text.

CLASSIFICATION RULE, applied uniformly
--------------------------------------
The only question asked is the one `ReasonFamily` asks: does rewriting the draft
resolve this? A code is classified only when the source text points at exactly
one lever:

  SOURCE_EXISTENCE  the text demands sources that must already exist in the
                    world ("If none exist, the subject is not yet suitable")
  WRITING           the text tells the contributor to rewrite / re-edit the
                    draft's own words
  STRUCTURAL        the text sends the contributor somewhere else entirely
                    (another page, the sandbox, a waiting period, an admin)
  UNKNOWN           the source text offers remedies in two different families,
                    or offers none at all

When AFCH's own optgroup ("Notability", "Verifiability", "Prose issues", ...)
and the template text disagree on emphasis, AFCH's grouping breaks the tie: it
is what the reviewer was looking at when they picked the code. That tie-break
fires exactly once, on `resume`, and it is noted inline there.

Seven decline codes are deliberately left UNKNOWN. Guessing them would be worse
than admitting them: see UNKNOWN_RATIONALE and `coverage_report()`.

AFCH's reject dropdown (`n`, `j`, `e`) writes into the same template parameter
as a decline code, so these show up in parsed decline histories -- `n` 23 times
and `e` 5 times in the dataset. They are in the table, marked `kind="reject"`,
and always UNKNOWN: "reject" ends the submission rather than asking for a fix,
so "does rewriting help" is not a question anyone asked. `blp` exists in both
dropdowns with different meanings ("unsourced possibly defamatory claims" vs
"Topic is an attack page"); the decline sense is the one in this table, and it
is UNKNOWN anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import REASON_FAMILY, ReasonFamily


@dataclass(frozen=True)
class ReasonInfo:
    """One decline code, its family, and the evidence behind that family."""

    code: str
    family: ReasonFamily
    description: str  # verbatim from AFCH or from Template:AfC submission/comments
    source: str  # "afch" | "template" | "templatedata"
    kind: str = "decline"  # "decline" | "reject"; reject codes share the same slot
    aliases: tuple[str, ...] = ()
    rationale: str = ""

    @property
    def all_codes(self) -> tuple[str, ...]:
        return (self.code,) + self.aliases


_SE = ReasonFamily.SOURCE_EXISTENCE
_WR = ReasonFamily.WRITING
_ST = ReasonFamily.STRUCTURAL
_UN = ReasonFamily.UNKNOWN

REASONS: tuple[ReasonInfo, ...] = (
    # ------------------------------------------------------------------------
    # SOURCE_EXISTENCE -- rewriting the draft cannot create sources
    # ------------------------------------------------------------------------
    # why: demands sources that do not yet exist; 'If none exist, the subject is not yet suitable for Wikipedia'
    ReasonInfo(
        code="nn",
        family=_SE,
        description="Submission is about a topic not yet shown to meet general notability guidelines (be more specific if possible)",
        source="afch",
        rationale="demands sources that do not yet exist; 'If none exist, the subject is not yet suitable for Wikipedia'",
    ),
    # why: asks for reliable sources to replace unreliable ones; 'If you cannot find a reliable source for the material, it should be removed'
    ReasonInfo(
        code="v",
        family=_SE,
        description="Submission lacks sufficient sources to verify content",
        source="afch",
        aliases=("rs", "source",),
        rationale="asks for reliable sources to replace unreliable ones; 'If you cannot find a reliable source for the material, it should be removed'",
    ),
    # why: template demands 'multiple published secondary sources' that are significant/reliable/independent, same bar as nn
    ReasonInfo(
        code="ns",
        family=_SE,
        description="Submission contains no sources or inline citations",
        source="afch",
        aliases=("nosource",),
        rationale="template demands 'multiple published secondary sources' that are significant/reliable/independent, same bar as nn",
    ),
    # why: requires high-quality biomedical secondary sources to exist; 'If none exist, the subject is not yet suitable for Wikipedia'
    ReasonInfo(
        code="med",
        family=_SE,
        description="Submission is about a medical or health claim that lacks sufficient sources",
        source="afch",
        rationale="requires high-quality biomedical secondary sources to exist; 'If none exist, the subject is not yet suitable for Wikipedia'",
    ),
    # why: notability optgroup; needs independent secondary coverage of the person
    ReasonInfo(
        code="bio",
        family=_SE,
        description="Submission is about a person not yet shown to meet notability guidelines",
        source="afch",
        rationale="notability optgroup; needs independent secondary coverage of the person",
    ),
    # why: notability optgroup; needs independent secondary coverage of the organization
    ReasonInfo(
        code="corp",
        family=_SE,
        description="Submission is about a company or organization not yet shown to meet notability guidelines",
        source="afch",
        aliases=("org", "inc",),
        rationale="notability optgroup; needs independent secondary coverage of the organization",
    ),
    # why: notability optgroup; needs independent secondary coverage of the event
    ReasonInfo(
        code="event",
        family=_SE,
        description="Submission is about an event not yet shown to meet notability guidelines",
        source="afch",
        rationale="notability optgroup; needs independent secondary coverage of the event",
    ),
    # why: notability optgroup; needs WP:NMUSIC evidence or independent secondary coverage
    ReasonInfo(
        code="music",
        family=_SE,
        description="Submission is about a musician or musical work not yet shown to meet notability guidelines",
        source="afch",
        aliases=("m", "band",),
        rationale="notability optgroup; needs WP:NMUSIC evidence or independent secondary coverage",
    ),
    # why: notability optgroup; needs WP:NSPORT evidence or independent secondary coverage
    ReasonInfo(
        code="athlete",
        family=_SE,
        description="Submission is about an athlete not yet shown to meet notability guidelines",
        source="afch",
        aliases=("sport",),
        rationale="notability optgroup; needs WP:NSPORT evidence or independent secondary coverage",
    ),
    # why: notability optgroup; needs independent secondary coverage of the film
    ReasonInfo(
        code="film",
        family=_SE,
        description="Submission is about a film not yet shown to meet notability guidelines",
        source="afch",
        rationale="notability optgroup; needs independent secondary coverage of the film",
    ),
    # why: notability optgroup; needs independent secondary coverage of the book
    ReasonInfo(
        code="book",
        family=_SE,
        description="Submission is about a book not yet shown to meet notability guidelines",
        source="afch",
        rationale="notability optgroup; needs independent secondary coverage of the book",
    ),
    # why: notability optgroup; needs independent secondary coverage of the web content
    ReasonInfo(
        code="web",
        family=_SE,
        description="Submission is about web content not yet shown to meet notability guidelines",
        source="afch",
        rationale="notability optgroup; needs independent secondary coverage of the web content",
    ),
    # why: notability optgroup; needs independent secondary coverage of the object
    ReasonInfo(
        code="astro",
        family=_SE,
        description="Submission is about an astronomical object not yet shown to meet notability guidelines",
        source="afch",
        rationale="notability optgroup; needs independent secondary coverage of the object",
    ),
    # why: notability optgroup; needs independent secondary coverage of the professional
    ReasonInfo(
        code="creative",
        family=_SE,
        description="Submission is about a creative professional not yet shown to meet notability guidelines",
        source="afch",
        rationale="notability optgroup; needs independent secondary coverage of the professional",
    ),
    # why: notability optgroup; needs independent secondary coverage of the feature
    ReasonInfo(
        code="geo",
        family=_SE,
        description="Submission is about a geographic feature not yet shown to meet notability guidelines",
        source="afch",
        rationale="notability optgroup; needs independent secondary coverage of the feature",
    ),
    # why: notability optgroup; needs sources discussing the group as a whole
    ReasonInfo(
        code="list",
        family=_SE,
        description="Submission is a list not yet shown to meet notability guidelines",
        source="afch",
        rationale="notability optgroup; needs sources discussing the group as a whole",
    ),
    # why: notability optgroup; needs independent secondary coverage of the term
    ReasonInfo(
        code="neo",
        family=_SE,
        description="Submission is about a neologism not yet shown to meet notability guidelines",
        source="afch",
        rationale="notability optgroup; needs independent secondary coverage of the term",
    ),
    # why: notability optgroup; needs independent secondary coverage of the number
    ReasonInfo(
        code="number",
        family=_SE,
        description="Submission is about a number not yet shown to meet notability guidelines",
        source="afch",
        rationale="notability optgroup; needs independent secondary coverage of the number",
    ),
    # why: notability optgroup; needs WP:NACADEMIC evidence or independent secondary coverage
    ReasonInfo(
        code="prof",
        family=_SE,
        description="Submission is about a professor not yet shown to meet notability guidelines",
        source="afch",
        aliases=("academic",),
        rationale="notability optgroup; needs WP:NACADEMIC evidence or independent secondary coverage",
    ),
    # why: notability optgroup; needs independent secondary coverage of the school
    ReasonInfo(
        code="school",
        family=_SE,
        description="Submission is about a school not yet shown to meet notability guidelines",
        source="afch",
        rationale="notability optgroup; needs independent secondary coverage of the school",
    ),
    # why: notability optgroup; needs independent secondary coverage of the species
    ReasonInfo(
        code="species",
        family=_SE,
        description="Submission is about a species not yet shown to meet notability guidelines",
        source="afch",
        rationale="notability optgroup; needs independent secondary coverage of the species",
    ),
    # why: template asks for real-world-context sources: 'Please add references that meet all three of these criteria. If none exist...'
    ReasonInfo(
        code="plot",
        family=_SE,
        description="Submission consists mostly of a plot summary",
        source="afch",
        rationale="template asks for real-world-context sources: 'Please add references that meet all three of these criteria. If none exist...'",
    ),
    # why: template asks for sources 'published after the initial breaking news has passed'; lasting coverage must exist
    ReasonInfo(
        code="news",
        family=_SE,
        description="Submission appears to be a news story of a single event",
        source="afch",
        aliases=("oneevent",),
        rationale="template asks for sources 'published after the initial breaking news has passed'; lasting coverage must exist",
    ),
    # ------------------------------------------------------------------------
    # WRITING -- editing the draft's own text resolves it
    # ------------------------------------------------------------------------
    # why: remedy is to rewrite: 'only summarize in your own words a range of independent, reliable, published sources'
    ReasonInfo(
        code="ai",
        family=_WR,
        description="Submission appears to be a large language model output",
        source="afch",
        aliases=("llm",),
        rationale="remedy is to rewrite: 'only summarize in your own words a range of independent, reliable, published sources'",
    ),
    # why: prose optgroup; remedy is 'Rewrite the draft to remove: promotional language... personal commentary... informal language'
    ReasonInfo(
        code="adv",
        family=_WR,
        description="Submission reads like an advertisement",
        source="afch",
        aliases=("spam", "advert",),
        rationale="prose optgroup; remedy is 'Rewrite the draft to remove: promotional language... personal commentary... informal language'",
    ),
    # why: prose optgroup; remedy is 'Rewrite the draft to remove' non-neutral language
    ReasonInfo(
        code="npov",
        family=_WR,
        description="Submission is not written in a formal, neutral encyclopedic tone",
        source="afch",
        rationale="prose optgroup; remedy is 'Rewrite the draft to remove' non-neutral language",
    ),
    # why: prose optgroup; remedy is to summarize sources instead of arguing, i.e. rewrite
    ReasonInfo(
        code="essay",
        family=_WR,
        description="Submission reads like an essay",
        source="afch",
        rationale="prose optgroup; remedy is to summarize sources instead of arguing, i.e. rewrite",
    ),
    # why: prose optgroup in AFCH; the complaint is how the draft reads. NOTE: the on-wiki template also appends a bio-notability sourcing demand, so this code is family-mixed in practice
    ReasonInfo(
        code="resume",
        family=_WR,
        description="Submission reads like a résumé",
        source="afch",
        rationale="prose optgroup in AFCH; the complaint is how the draft reads. NOTE: the on-wiki template also appends a bio-notability sourcing demand, so this code is family-mixed in practice",
    ),
    # why: sole remedy is prose: 'Please see the guide to writing better articles for how to improve your writing'
    ReasonInfo(
        code="context",
        family=_WR,
        description="Submission provides insufficient context",
        source="afch",
        rationale="sole remedy is prose: 'Please see the guide to writing better articles for how to improve your writing'",
    ),
    # why: remedy is 'Please provide a high-quality English translation' -- a rewrite of the same content
    ReasonInfo(
        code="lang",
        family=_WR,
        description="Submission is not in English",
        source="afch",
        aliases=("notenglish",),
        rationale="remedy is 'Please provide a high-quality English translation' -- a rewrite of the same content",
    ),
    # why: remedy is placement, not existence: 'Please edit your draft to support your statements with inline citations'
    ReasonInfo(
        code="ilc",
        family=_WR,
        description="Submission is a BLP that does not meet minimum inline citation requirements",
        source="afch",
        aliases=("cite", "footnote",),
        rationale="remedy is placement, not existence: 'Please edit your draft to support your statements with inline citations'",
    ),
    # ------------------------------------------------------------------------
    # STRUCTURAL -- the action needed is not on this draft
    # ------------------------------------------------------------------------
    # why: action is elsewhere: 'To improve the existing article, please edit it directly at ...'
    ReasonInfo(
        code="exists",
        family=_ST,
        description="Submission is duplicated by another article already in mainspace",
        source="afch",
        rationale="action is elsewhere: 'To improve the existing article, please edit it directly at ...'",
    ),
    # why: action is elsewhere: 'Any future edits or improvements should be made on that submission, not here'
    ReasonInfo(
        code="dup",
        family=_ST,
        description="Submission is a duplicate of another AfC draft submission",
        source="afch",
        aliases=("duplicate",),
        rationale="action is elsewhere: 'Any future edits or improvements should be made on that submission, not here'",
    ),
    # why: action is elsewhere: 'Edit the existing article to include your text and citations into an appropriate section'
    ReasonInfo(
        code="mergeto",
        family=_ST,
        description="Submission should be merged into an existing article",
        source="afch",
        aliases=("merge",),
        rationale="action is elsewhere: 'Edit the existing article to include your text and citations into an appropriate section'",
    ),
    # why: nothing to rewrite; content must be supplied or un-hidden from a comment tag
    ReasonInfo(
        code="blank",
        family=_ST,
        description="Submission is blank",
        source="afch",
        aliases=("empty",),
        rationale="nothing to rewrite; content must be supplied or un-hidden from a comment tag",
    ),
    # why: not an article attempt: 'Please use the sandbox to practice editing'
    ReasonInfo(
        code="test",
        family=_ST,
        description="Submission appears to be a test edit",
        source="afch",
        rationale="not an article attempt: 'Please use the sandbox to practice editing'",
    ),
    # why: wrong venue: 'This is not the correct place to request new redirects'
    ReasonInfo(
        code="redirect",
        family=_ST,
        description="Submission is a redirect request",
        source="afch",
        rationale="wrong venue: 'This is not the correct place to request new redirects'",
    ),
    # why: wrong venue: 'This is not the correct place to request new categories'
    ReasonInfo(
        code="cat",
        family=_ST,
        description="Submission is a category request",
        source="afch",
        rationale="wrong venue: 'This is not the correct place to request new categories'",
    ),
    # why: nothing about the draft: 'You must wait until your account becomes Extended confirmed'
    ReasonInfo(
        code="ecr",
        family=_ST,
        description="Submission is a contentious topic with an editing restriction",
        source="afch",
        rationale="nothing about the draft: 'You must wait until your account becomes Extended confirmed'",
    ),
    # why: no repair path offered; 'Drafts that meet these criteria may be deleted without notice'
    ReasonInfo(
        code="van",
        family=_ST,
        description="Submission is vandalism, a negative unsourced BLP, or an attack page",
        source="afch",
        rationale="no repair path offered; 'Drafts that meet these criteria may be deleted without notice'",
    ),
    # why: same remedy language as test -- 'Please use the sandbox to practice editing'; no repair path for the draft
    ReasonInfo(
        code="joke",
        family=_ST,
        description="Submission appears to be a joke or hoax",
        source="afch",
        aliases=("hoax",),
        rationale="same remedy language as test -- 'Please use the sandbox to practice editing'; no repair path for the draft",
    ),
    # ------------------------------------------------------------------------
    # UNKNOWN -- deliberately unclassified; never guessed
    # ------------------------------------------------------------------------
    # why: template says 'you must rewrite or summarize it entirely in your own words' (WRITING) but AFCH simultaneously files WP:G12 speedy deletion (STRUCTURAL); two families, no single lever
    ReasonInfo(
        code="cv",
        family=_UN,
        description="Submission is a copyright violation",
        source="afch",
        aliases=("cv-n",),
        rationale="template says 'you must rewrite or summarize it entirely in your own words' (WRITING) but AFCH simultaneously files WP:G12 speedy deletion (STRUCTURAL); two families, no single lever",
    ),
    # why: same split as cv: the infringing text was removed but the contaminated history still needs an administrator action
    ReasonInfo(
        code="cv-cleaned",
        family=_UN,
        description="This draft appears to contain copyrighted material }|||from {{{2}}},}} which has been removed",
        source="template",
        aliases=("cv-c",),
        rationale="same split as cv: the infringing text was removed but the contaminated history still needs an administrator action",
    ),
    # why: stated remedy is 'add inline citations to reliable sources' (SOURCE_EXISTENCE) but simply not restoring the unsourced negative material also resolves it (WRITING)
    ReasonInfo(
        code="blp",
        family=_UN,
        description="BLP contains unsourced, possibly defamatory claims (AGF and wait for sources)",
        source="afch",
        rationale="stated remedy is 'add inline citations to reliable sources' (SOURCE_EXISTENCE) but simply not restoring the unsourced negative material also resolves it (WRITING)",
    ),
    # why: template offers two remedies in different families: 'it must expand on the subject' (WRITING) or 'contribute to our sister project Wiktionary' (STRUCTURAL)
    ReasonInfo(
        code="dict",
        family=_UN,
        description="Submission is a dictionary definition",
        source="afch",
        aliases=("d",),
        rationale="template offers two remedies in different families: 'it must expand on the subject' (WRITING) or 'contribute to our sister project Wiktionary' (STRUCTURAL)",
    ),
    # why: no remedy sentence at all -- 'Please read What Wikipedia is not to understand what content is excluded'; the lever is unstated
    ReasonInfo(
        code="not",
        family=_UN,
        description="Submission fails [[Wikipedia:What Wikipedia is not]]",
        source="afch",
        rationale="no remedy sentence at all -- 'Please read What Wikipedia is not to understand what content is excluded'; the lever is unstated",
    ),
    # why: free-text custom decline; the code carries no fixed semantics, the meaning lives in the reviewer's prose
    ReasonInfo(
        code="reason",
        family=_UN,
        description="Enter decline reason in the box below",
        source="afch",
        rationale="free-text custom decline; the code carries no fixed semantics, the meaning lives in the reviewer's prose",
    ),
    # why: listed in Template:AfC submission/doc templatedata but has NO branch in the /comments #switch, so it renders as the bare string 'afd'; there is no source text to classify from (0 occurrences in the 1,472-decline dataset)
    ReasonInfo(
        code="afd",
        family=_UN,
        description="afd",
        source="templatedata",
        rationale="listed in Template:AfC submission/doc templatedata but has NO branch in the /comments #switch, so it renders as the bare string 'afd'; there is no source text to classify from (0 occurrences in the 1,472-decline dataset)",
    ),
    # why: reject, not decline: 'Topic is not notable' ends the submission rather than asking for a fix; the rewrite question does not apply
    ReasonInfo(
        code="n",
        family=_UN,
        description="Topic is not notable",
        source="afch",
        kind="reject",
        rationale="reject, not decline: 'Topic is not notable' ends the submission rather than asking for a fix; the rewrite question does not apply",
    ),
    # why: reject, not decline: 'Topic is a joke or hoax' ends the submission rather than asking for a fix
    ReasonInfo(
        code="j",
        family=_UN,
        description="Topic is a joke or hoax",
        source="afch",
        kind="reject",
        rationale="reject, not decline: 'Topic is a joke or hoax' ends the submission rather than asking for a fix",
    ),
    # why: reject, not decline: 'Topic is contrary to the purpose of Wikipedia' ends the submission rather than asking for a fix
    ReasonInfo(
        code="e",
        family=_UN,
        description="Topic is contrary to the purpose of Wikipedia",
        source="afch",
        kind="reject",
        rationale="reject, not decline: 'Topic is contrary to the purpose of Wikipedia' ends the submission rather than asking for a fix",
    ),
)


# Why each UNKNOWN stays unknown. Surfaced by `coverage_report()` so a caller can
# show a reviewer *why* First Reader has nothing to say about a code.
UNKNOWN_RATIONALE: dict[str, str] = {r.code: r.rationale for r in REASONS if r.family is ReasonFamily.UNKNOWN}

# Codes that come from AFCH's reject dropdown rather than its decline dropdown.
REJECT_ONLY_CODES: frozenset[str] = frozenset(r.code for r in REASONS if r.kind == "reject")

BY_CODE: dict[str, ReasonInfo] = {}
CANONICAL: dict[str, str] = {}
for _info in REASONS:
    for _c in _info.all_codes:
        if _c in BY_CODE:  # pragma: no cover -- guards against a table typo
            raise ValueError(f"duplicate AfC code in table: {_c}")
        BY_CODE[_c] = _info
        CANONICAL[_c] = _info.code

KNOWN_CODES: frozenset[str] = frozenset(BY_CODE)
UNKNOWN_CODES: tuple[str, ...] = tuple(sorted(UNKNOWN_RATIONALE))
DECLINE_UNKNOWN_CODES: tuple[str, ...] = tuple(
    sorted(r.code for r in REASONS if r.family is ReasonFamily.UNKNOWN and r.kind == "decline")
)


def normalize_code(code: str | None) -> str:
    """Lowercase and trim, the way the on-wiki template does (`{{lc:{{{1}}}}}`).

    Wikitext in the wild carries `NN`, ` corp `, `Adv`. Everything downstream
    compares normalized codes.
    """
    return (code or "").strip().lower()


def canonical_code(code: str | None) -> str:
    """Resolve an alias to its canonical code. Unknown codes pass through."""
    c = normalize_code(code)
    return CANONICAL.get(c, c)


def describe(code: str | None) -> ReasonInfo | None:
    """The table entry for a code, or None when we have never seen the code."""
    return BY_CODE.get(normalize_code(code))


def family_for(code: str | None) -> ReasonFamily:
    """Family of a decline code. Unmapped codes are UNKNOWN, never a guess."""
    info = describe(code)
    return info.family if info is not None else ReasonFamily.UNKNOWN


def fixable_by_rewrite(code: str | None) -> bool | None:
    """None means we do not know, which is different from False."""
    fam = family_for(code)
    if fam is ReasonFamily.UNKNOWN:
        return None
    return fam is ReasonFamily.WRITING


def coverage_report() -> dict[str, object]:
    """What this table does and does not cover. Meant to be shown, not hidden."""
    counts = {fam.value: 0 for fam in ReasonFamily}
    for info in REASONS:
        counts[info.family.value] += 1
    declines = [r for r in REASONS if r.kind == "decline"]
    return {
        "canonical_codes": len(REASONS),
        "decline_codes": len(declines),
        "reject_codes": len(REASONS) - len(declines),
        "codes_including_aliases": len(BY_CODE),
        "by_family": counts,
        "unknown_codes": UNKNOWN_CODES,
        "unknown_decline_codes": DECLINE_UNKNOWN_CODES,
        "unknown_rationale": dict(UNKNOWN_RATIONALE),
        "reject_only_codes": tuple(sorted(REJECT_ONLY_CODES)),
    }


def as_family_map() -> dict[str, ReasonFamily]:
    """The flat `code -> family` map, aliases included, UNKNOWN entries omitted.

    This is exactly what `models.REASON_FAMILY` must contain. It is generated
    here so the two can be compared instead of drifting.
    """
    return {code: info.family for code, info in BY_CODE.items() if info.family is not ReasonFamily.UNKNOWN}


def _check_models_agreement() -> None:
    """Fail loudly if models.REASON_FAMILY and this table have drifted apart.

    `Decline.family` reads `models.REASON_FAMILY` directly, so the two must say
    the same thing. models.py holds the flat map (it is the shared contract);
    this module holds the evidence and the reasoning behind every row.
    """
    expected = as_family_map()
    if REASON_FAMILY != expected:
        missing = sorted(set(expected) - set(REASON_FAMILY))
        extra = sorted(set(REASON_FAMILY) - set(expected))
        conflicts = sorted(
            f"{c}: models={REASON_FAMILY[c].value} table={expected[c].value}"
            for c in set(expected) & set(REASON_FAMILY)
            if REASON_FAMILY[c] is not expected[c]
        )
        raise ValueError(
            "models.REASON_FAMILY has drifted from the extracted AFCH table -- "
            f"missing={missing} unexpected={extra} conflicts={conflicts}"
        )


_check_models_agreement()
