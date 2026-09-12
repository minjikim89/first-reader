"""Mechanical evidence that a contributor tried to address a decline reason.

Read `AttemptSignal` in models.py before changing anything here. The contract
is that every field is derivable from two wikitext blobs and a list of
revisions, with no judgment about source quality mixed in.

What this module deliberately does NOT do
-----------------------------------------
It does not look at whether a newly cited domain is reliable, independent, or
substantial. Deciding that IS the review. `new_domains` is a list of hostnames
and nothing more: a caller may show it to a human, but must not rank it.

Counting references
-------------------
Both citation styles appear in AfC drafts and both must be caught:

    <ref>Smith, J. (2019). ...</ref>          bare ref
    <ref>{{cite web |url=... |title=...}}</ref>   templated ref
    {{cite book |title=...}}                  cite template outside any ref

Counting `<ref>` tags and `{{cite}}` templates and adding the two would double
count the middle case, which is the most common one on Wikipedia -- a draft
with ten templated refs would report twenty. So the total is

    (number of <ref> tags) + (number of {{cite}} templates NOT inside a <ref>)

which catches both styles exactly once each. `count_refs` returns that total;
`count_ref_tags` and `count_cite_templates` expose the two halves for anyone
who needs to audit the number.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable

from ..models import AttemptSignal, ReasonFamily, Revision
from .reasons import family_for

# --- wikitext scanners -------------------------------------------------------

# Matches <ref>, <ref name="x">, and self-closing <ref name="x" />.
# Does not match the closing </ref>.
_RE_REF_TAG = re.compile(r"<ref\b[^>]*?>", re.IGNORECASE)
# A full <ref ...>...</ref> pair. The lookbehind keeps self-closing tags out.
_RE_REF_BLOCK = re.compile(r"<ref\b[^>]*?(?<!/)>.*?</ref\s*>", re.IGNORECASE | re.DOTALL)
_RE_CITE_OPEN = re.compile(r"\{\{\s*cite\b", re.IGNORECASE)
_RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_URL = re.compile(r"https?://([^\s\]\[|{}<>\"']+)", re.IGNORECASE)

_CITATION_SECTIONS = frozenset(
    {
        "references",
        "reference",
        "sources",
        "source",
        "citations",
        "notes",
        "footnotes",
        "bibliography",
        "works cited",
        "further reading",
        "external links",
    }
)


def count_ref_tags(text: str) -> int:
    """Number of `<ref>` openings, self-closing named refs included."""
    return len(_RE_REF_TAG.findall(text))


def _ref_block_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _RE_REF_BLOCK.finditer(text)]


def count_cite_templates(text: str, *, outside_refs_only: bool = False) -> int:
    """Number of `{{cite ...}}` templates.

    With `outside_refs_only`, templates wrapped in a `<ref>` are skipped so the
    caller can add this to `count_ref_tags` without double counting.
    """
    spans = _ref_block_spans(text) if outside_refs_only else []
    total = 0
    for m in _RE_CITE_OPEN.finditer(text):
        start = m.start()
        if any(a <= start < b for a, b in spans):
            continue
        total += 1
    return total


def count_refs(text: str) -> int:
    """Citations in the draft, counting each one once across both styles."""
    return count_ref_tags(text) + count_cite_templates(text, outside_refs_only=True)


def extract_domains(text: str) -> set[str]:
    """Hostnames of every URL in the wikitext. Hostname only, no path.

    Bare URLs, `[url label]` links and `|url=` / `|archive-url=` template
    parameters all reduce to the same regex, so one pass covers them.
    """
    hosts: set[str] = set()
    for raw in _RE_URL.findall(text):
        host = raw.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        host = host.split("@")[-1]  # strip any userinfo
        host = host.split(":")[0]  # strip port
        host = host.strip(" .,;:'\"").lower()
        if host.startswith("www."):
            host = host[4:]
        if host:
            hosts.add(host)
    return hosts


# --- markup stripping --------------------------------------------------------


def _strip_balanced(text: str, open_tok: str, close_tok: str) -> str:
    """Remove `{{...}}` / `{|...|}` runs, honouring nesting."""
    out: list[str] = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        if text.startswith(open_tok, i):
            depth += 1
            i += len(open_tok)
            continue
        if depth and text.startswith(close_tok, i):
            depth -= 1
            i += len(close_tok)
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    return "".join(out)


def strip_markup(text: str) -> str:
    """Wikitext -> the prose a reader would actually see.

    Templates, refs, tables, file and category links are removed rather than
    flattened, so that reshuffling citations does not register as a prose
    change and vice versa. `similarity` is computed on this.
    """
    t = _RE_COMMENT.sub(" ", text)
    t = _RE_REF_BLOCK.sub(" ", t)
    t = _RE_REF_TAG.sub(" ", t)
    t = re.sub(r"</ref\s*>", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"<nowiki>.*?</nowiki>", " ", t, flags=re.IGNORECASE | re.DOTALL)
    t = _strip_balanced(t, "{|", "|}")
    t = _strip_balanced(t, "{{", "}}")
    t = re.sub(r"\[\[\s*(?:File|Image|Category)\s*:[^\]]*\]\]", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", t, flags=re.IGNORECASE)
    t = re.sub(r"\[https?://[^\]\s]*\]", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"https?://\S+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"^\s*=+\s*(.*?)\s*=+\s*$", r"\1", t, flags=re.MULTILINE)
    t = re.sub(r"^[*#:;]+\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.replace("&nbsp;", " ")
    return " ".join(t.split())


def citation_surface(text: str) -> str:
    """Everything in the draft that is citation apparatus, normalised.

    That is: every `<ref>` block, every `{{cite}}` template, and the body of any
    section titled References / Sources / External links / etc. Comparing this
    before and after answers "did the edit touch the sourcing at all" without
    saying anything about whether the sourcing is any good.
    """
    parts: list[str] = []
    parts.extend(m.group(0) for m in _RE_REF_BLOCK.finditer(text))
    parts.extend(m.group(0) for m in _RE_REF_TAG.finditer(text))
    for m in _RE_CITE_OPEN.finditer(text):
        parts.append(_balanced_slice(text, m.start()))
    for heading, body in _sections(text):
        if heading.strip().lower().strip(":") in _CITATION_SECTIONS:
            parts.append(body)
    return " ".join(" ".join(p.split()) for p in parts)


def _balanced_slice(text: str, start: int) -> str:
    """The `{{...}}` beginning at `start`, nesting honoured."""
    depth = 0
    i = start
    n = len(text)
    while i < n:
        if text.startswith("{{", i):
            depth += 1
            i += 2
            continue
        if text.startswith("}}", i):
            depth -= 1
            i += 2
            if depth == 0:
                return text[start:i]
            continue
        i += 1
    return text[start:]


def _sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) pairs. The lead is returned with an empty heading."""
    out: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^\s*=+\s*(.*?)\s*=+\s*$", line)
        if m:
            out.append((heading, "\n".join(buf)))
            heading = m.group(1)
            buf = []
        else:
            buf.append(line)
    out.append((heading, "\n".join(buf)))
    return out


def prose_similarity(before: str, after: str) -> float:
    """1.0 = prose unchanged, 0.0 = fully rewritten.

    Compared on markup-stripped word tokens, so that adding ten citations to
    untouched prose still reads as ~1.0 and rewriting every sentence reads low
    even if the citation block is identical.
    """
    a = strip_markup(before).split()
    b = strip_markup(after).split()
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


# --- the signal --------------------------------------------------------------


def detect_attempt(
    before_wikitext: str,
    after_wikitext: str,
    revisions_between: list[Revision],
    target_reason: str,
) -> AttemptSignal:
    """Measure what changed between two submissions, against one decline reason.

    `target_reason` is an AfC decline code; its family decides what counts as
    "the target area". Nothing here judges whether the reason was resolved --
    see the `AttemptSignal` docstring for why that line exists.
    """
    refs_before = count_refs(before_wikitext)
    refs_after = count_refs(after_wikitext)

    domains_before = extract_domains(before_wikitext)
    domains_after = extract_domains(after_wikitext)
    new_domains = tuple(sorted(domains_after - domains_before))

    bytes_changed = len(after_wikitext.encode("utf-8")) - len(before_wikitext.encode("utf-8"))
    similarity = prose_similarity(before_wikitext, after_wikitext)

    family = family_for(target_reason)
    if family is ReasonFamily.SOURCE_EXISTENCE:
        touched = (
            refs_after != refs_before
            or bool(new_domains)
            or citation_surface(before_wikitext) != citation_surface(after_wikitext)
        )
    elif family is ReasonFamily.WRITING:
        touched = strip_markup(before_wikitext) != strip_markup(after_wikitext)
    else:
        # STRUCTURAL points at an action taken somewhere other than this draft,
        # and UNKNOWN points nowhere we are willing to name. Neither can be
        # answered from a diff of this page, so we report False rather than
        # invent a target area.
        touched = False

    return AttemptSignal(
        refs_before=refs_before,
        refs_after=refs_after,
        new_domains=new_domains,
        bytes_changed=bytes_changed,
        similarity=similarity,
        touched_target_area=touched,
        editor_comments=_edit_summaries(revisions_between),
    )


def _edit_summaries(revisions: Iterable[Revision]) -> tuple[str, ...]:
    """Non-empty edit summaries, oldest first.

    Blank summaries are dropped: an empty string is not something a reviewer can
    read, and keeping them would make the tuple length look like an edit count.
    """
    ordered = sorted(revisions, key=lambda r: (r.timestamp, r.revid))
    return tuple(r.comment.strip() for r in ordered if r.comment and r.comment.strip())
