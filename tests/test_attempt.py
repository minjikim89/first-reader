"""AttemptSignal computation, on two real-shaped AfC drafts.

The fixtures below are a before/after pair of the same draft. BEFORE is a
promotional, thinly-cited stub of the kind that gets declined `adv`+`nn`. AFTER
is what a contributor typically produces on resubmission: prose rewritten to
sound neutral, three more citations, one genuinely new publisher.

No network. Everything here is computed from the two strings.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from first_reader.analysis.attempt import (
    citation_surface,
    count_cite_templates,
    count_ref_tags,
    count_refs,
    detect_attempt,
    extract_domains,
    prose_similarity,
    strip_markup,
)
from first_reader.models import Revision

BEFORE = """{{AfC submission|d|adv|u=Contributor|ns=118|decliner=Reviewer1|declinets=20260301120000|ts=20260228090000|reason2=nn}}
{{Infobox company
| name = Northwind Analytics
| founded = 2019
| website = {{URL|https://northwind-analytics.example}}
}}

'''Northwind Analytics''' is a ''revolutionary'' data platform company that is
transforming how enterprises unlock the power of their data. Founded in 2019,
the company has quickly become a leader in the space.<ref>{{cite web
|url=https://northwind-analytics.example/about |title=About Us
|publisher=Northwind Analytics |access-date=2026-02-01}}</ref>

== History ==
The company was founded by two former engineers who saw an opportunity to
disrupt the analytics market. Their innovative approach has been widely
praised.<ref>https://blog.northwind-analytics.example/our-story</ref>

== Products ==
Northwind offers a best-in-class suite of solutions.

== References ==
{{reflist}}

[[Category:Companies established in 2019]]
"""

AFTER = """{{AfC submission|d|nn|u=Contributor|ns=118|decliner=Reviewer2|declinets=20260420120000|ts=20260415090000}}
{{Infobox company
| name = Northwind Analytics
| founded = 2019
| website = {{URL|https://northwind-analytics.example}}
}}

'''Northwind Analytics''' is a software company headquartered in Leeds, England.
It was founded in 2019 and develops data analysis tools for enterprise
customers.<ref>{{cite news |url=https://www.thetimes.co.uk/article/leeds-tech
|title=Leeds technology sector grows |work=The Times |date=2025-06-11}}</ref>

== History ==
The company was established in 2019 by two engineers. It raised a Series A
round in 2023.<ref>{{cite news |url=https://www.ft.com/content/northwind-series-a
|title=Northwind raises Series A |work=Financial Times |date=2023-09-02}}</ref>
It employed 40 people as of 2025.<ref>{{cite web
|url=https://www.bbc.co.uk/news/business-leeds |title=Leeds employers
|publisher=BBC News |date=2025-11-30}}</ref>

== Products ==
The company publishes a reporting tool and a data pipeline
service.<ref>{{cite web |url=https://northwind-analytics.example/products
|title=Products |publisher=Northwind Analytics}}</ref>

== References ==
{{reflist}}

[[Category:Companies established in 2019]]
[[Category:Software companies of England]]
"""

# A resubmission that adds sources without touching a word of the prose.
SOURCES_ONLY = AFTER.replace(
    "== References ==",
    "<ref name=extra>{{cite book |title=Yorkshire Business Directory "
    "|url=https://archive.org/details/yorkshire-directory}}</ref>\n\n== References ==",
)


def _revs(*comments: str) -> list[Revision]:
    base = datetime(2026, 3, 5, tzinfo=timezone.utc)
    return [
        Revision(
            revid=500 + i,
            parentid=499 + i,
            user="Contributor",
            timestamp=base + timedelta(days=i),
            comment=c,
            size=4000 + i * 100,
        )
        for i, c in enumerate(comments)
    ]


class TestCountingReferences:
    def test_both_citation_styles_are_caught(self):
        text = "<ref>Plain text ref</ref> <ref>{{cite web|url=http://a.test}}</ref> {{cite book|title=B}}"
        assert count_ref_tags(text) == 2
        assert count_cite_templates(text) == 2
        assert count_refs(text) == 3, "the templated ref must not be counted twice"

    def test_self_closing_named_ref_counts(self):
        assert count_ref_tags('<ref name="a" />') == 1
        assert count_refs('<ref name="a" />') == 1

    def test_closing_tag_is_not_a_reference(self):
        assert count_ref_tags("</ref>") == 0

    def test_case_is_ignored(self):
        # One bare ref plus one standalone cite template: two citations either
        # way, but only if the scanners are case-insensitive.
        assert count_refs("<REF>x</REF> {{Cite Web|url=http://a.test}}") == 2
        assert count_refs("<REF>{{Cite Web|url=http://a.test}}</REF>") == 1

    def test_fixture_counts(self):
        assert count_refs(BEFORE) == 2
        assert count_refs(AFTER) == 4


class TestDomains:
    def test_hosts_only_www_stripped_port_dropped(self):
        text = "[https://www.Example.com:8443/a/b label] and https://sub.example.org/x?y=1"
        assert extract_domains(text) == {"example.com", "sub.example.org"}

    def test_template_url_parameters_are_found(self):
        assert "bbc.co.uk" in extract_domains("{{cite web |url=https://www.bbc.co.uk/news/x }}")

    def test_no_quality_judgment_is_applied(self):
        """A self-published blog is reported exactly like a national newspaper."""
        found = extract_domains(BEFORE)
        assert "blog.northwind-analytics.example" in found
        assert "northwind-analytics.example" in found


class TestStripMarkup:
    def test_templates_refs_and_categories_are_removed(self):
        out = strip_markup(BEFORE)
        assert "cite web" not in out
        assert "Infobox" not in out
        assert "Category" not in out
        assert "reflist" not in out
        assert "revolutionary" in out, "prose must survive"

    def test_links_reduce_to_their_visible_text(self):
        assert strip_markup("[[Leeds|the city]] and [[York]]") == "the city and York"

    def test_nested_templates_do_not_leak(self):
        assert strip_markup("a {{outer|{{inner|x}}|y}} b") == "a b"

    def test_headings_lose_their_markers(self):
        assert strip_markup("== History ==\ntext") == "History text"


class TestSimilarity:
    def test_identical_text_is_one(self):
        assert prose_similarity(BEFORE, BEFORE) == 1.0

    def test_adding_citations_without_touching_prose_stays_near_one(self):
        """Markup churn must not masquerade as a rewrite."""
        assert prose_similarity(AFTER, SOURCES_ONLY) == pytest.approx(1.0)

    def test_a_real_rewrite_scores_low(self):
        assert prose_similarity(BEFORE, AFTER) < 0.7

    def test_two_empty_drafts_are_identical(self):
        assert prose_similarity("", "") == 1.0


class TestCitationSurface:
    def test_prose_edits_do_not_change_the_citation_surface(self):
        edited = AFTER.replace("headquartered in Leeds, England", "based in Leeds")
        assert citation_surface(edited) == citation_surface(AFTER)

    def test_adding_a_reference_does_change_it(self):
        assert citation_surface(SOURCES_ONLY) != citation_surface(AFTER)


class TestDetectAttempt:
    def test_measurements_against_a_source_existence_reason(self):
        signal = detect_attempt(
            BEFORE, AFTER, _revs("rewrote lead", "added refs"), "nn"
        )
        assert signal.refs_before == 2
        assert signal.refs_after == 4
        assert signal.added_sources == 2
        assert signal.new_domains == ("bbc.co.uk", "ft.com", "thetimes.co.uk")
        assert "northwind-analytics.example" not in signal.new_domains, "already present before"
        assert signal.bytes_changed == len(AFTER.encode()) - len(BEFORE.encode())
        assert signal.bytes_changed > 0
        assert 0.0 <= signal.similarity < 0.7
        assert signal.touched_target_area is True
        assert signal.editor_comments == ("rewrote lead", "added refs")

    def test_writing_reason_looks_at_prose_not_citations(self):
        signal = detect_attempt(BEFORE, AFTER, [], "adv")
        assert signal.touched_target_area is True

        untouched_prose = detect_attempt(AFTER, SOURCES_ONLY, [], "adv")
        assert untouched_prose.touched_target_area is False, (
            "adding a citation is not an attempt at a prose problem"
        )
        assert untouched_prose.added_sources == 1

    def test_source_reason_sees_the_citation_only_edit(self):
        signal = detect_attempt(AFTER, SOURCES_ONLY, [], "nn")
        assert signal.touched_target_area is True
        assert signal.new_domains == ("archive.org",)

    def test_structural_reason_is_never_claimed_to_be_addressed(self):
        signal = detect_attempt(BEFORE, AFTER, [], "exists")
        assert signal.touched_target_area is False

    def test_unknown_reason_is_never_claimed_to_be_addressed(self):
        for code in ("cv", "reason", "banana"):
            signal = detect_attempt(BEFORE, AFTER, [], code)
            assert signal.touched_target_area is False, code

    def test_aliases_resolve_to_the_same_target_area(self):
        assert (
            detect_attempt(AFTER, SOURCES_ONLY, [], "nosource").touched_target_area
            is detect_attempt(AFTER, SOURCES_ONLY, [], "ns").touched_target_area
        )

    def test_no_change_at_all(self):
        signal = detect_attempt(BEFORE, BEFORE, [], "nn")
        assert signal.bytes_changed == 0
        assert signal.similarity == 1.0
        assert signal.new_domains == ()
        assert signal.touched_target_area is False
        assert signal.added_sources == 0
        assert signal.substantive_rewrite is False

    def test_substantive_rewrite_is_the_models_definition(self):
        signal = detect_attempt(BEFORE, AFTER, [], "nn")
        assert signal.substantive_rewrite is (
            signal.similarity < 0.7 and abs(signal.bytes_changed) > 500
        )

    def test_edit_summaries_are_ordered_and_blanks_dropped(self):
        revs = _revs("first", "", "   ", "second")
        signal = detect_attempt(BEFORE, AFTER, list(reversed(revs)), "nn")
        assert signal.editor_comments == ("first", "second")

    def test_removing_sources_never_reports_negative_added_sources(self):
        signal = detect_attempt(AFTER, BEFORE, [], "nn")
        assert signal.refs_before == 4
        assert signal.refs_after == 2
        assert signal.added_sources == 0
        assert signal.bytes_changed < 0
