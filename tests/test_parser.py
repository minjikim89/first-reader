"""Offline unit tests for the AfC template parser.

Every fixture in this file is a verbatim template taken from live English
Wikipedia drafts (sampled 2026-09 from Category:Pending AfC submissions and
Category:Declined AfC submissions). No network access.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from first_reader.models import ReasonFamily
from first_reader.wiki.parser import (
    SubmissionStatus,
    iter_afc_templates,
    parse_declines,
    parse_timestamp,
)

UTC = timezone.utc

# Draft:0→1 Doctrine for AI, exactly as it appears on the page: newest decline
# first, each template followed by the HTML comment AFCH writes.
DOCTRINE_STACK = """{{AfC submission|d|nn|u=Contributor 001|ns=118|decliner=Reviewer 003|declinets=20260708144901|ts=20260708144743}} <!-- Do not remove this line! -->
{{AfC submission|d|v|u=Contributor 001|ns=118|decliner=Reviewer 002|declinets=20260629205711|reason2=ai|small=yes|ts=20260629172235}} <!-- Do not remove this line! -->
{{AfC submission|d|ai|u=Contributor 001|ns=118|decliner=Reviewer 001|declinets=20260627045213|reason2=nn|small=yes|ts=20260627032430}} <!-- Do not remove this line! -->

{{AfC comment|1=You'll need to '''completely''' rewrite the article from scratch. [[WP:LLM]]}}

----

{{Short description|Doctrine for Artificial Intelligence}}

== 0→1 Doctrine for AI ==

The 0→1 Doctrine is a proposed pre-execution computing architecture.
"""


def ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=UTC)


class TestTimestamps:
    def test_parses_fourteen_digit_utc(self):
        assert parse_timestamp("20260627045213") == ts("20260627045213")

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            "   ",
            "2026-06-27T04:52:13Z",
            "not a timestamp",
            "20261345999999",  # syntactically fine, calendar-invalid
        ],
    )
    def test_returns_none_instead_of_raising(self, value):
        assert parse_timestamp(value) is None

    def test_accepts_twelve_digits_as_seconds_zero(self):
        # Draft:4fnnann -- ts lost its seconds. The sibling declinets on the
        # same template reads 20260723144903, which confirms the reading.
        assert parse_timestamp("202607231436") == ts("20260723143600")

    def test_accepts_a_pasted_signature_timestamp(self):
        # Draft:Michael Metzger -- a reviewer pasted their signature into ts.
        assert parse_timestamp("16:59, 14 August 2026 (UTC)") == ts("20260814165900")

    def test_signature_with_an_unknown_month_is_rejected(self):
        assert parse_timestamp("16:59, 14 Foo 2026 (UTC)") is None


@pytest.fixture(scope="module")
def result():
    """The reference draft, parsed once."""
    return parse_declines(DOCTRINE_STACK)


class TestDoctrineStack:
    """The reference draft. These values are asserted against the live page."""

    def test_three_declines(self, result):
        assert len(result.declines) == 3
        assert result.stats.declines_skipped == 0

    def test_ordered_oldest_first(self, result):
        assert [d.declined_at for d in result.declines] == sorted(
            d.declined_at for d in result.declines
        )

    def test_first_decline(self, result):
        d = result.declines[0]
        assert d.code == "ai"
        assert d.secondary_code == "nn"
        assert d.decliner == "Reviewer 001"
        assert d.submitted_at == ts("20260627032430")
        assert d.declined_at == ts("20260627045213")
        assert d.family is ReasonFamily.WRITING
        assert d.fixable_by_rewrite is True

    def test_second_decline(self, result):
        d = result.declines[1]
        assert d.code == "v"
        assert d.secondary_code == "ai"
        assert d.decliner == "Reviewer 002"
        assert d.family is ReasonFamily.SOURCE_EXISTENCE
        assert d.fixable_by_rewrite is False

    def test_third_decline(self, result):
        d = result.declines[2]
        assert d.code == "nn"
        assert d.secondary_code is None
        assert d.decliner == "Reviewer 003"

    def test_afc_comment_is_not_a_submission(self, result):
        assert result.stats.templates_found == 3

    def test_raw_is_the_whole_template(self, result):
        assert result.declines[0].raw.startswith("{{AfC submission|d|ai|")
        assert result.declines[0].raw.endswith("}}")
        assert "<!--" not in result.declines[0].raw


class TestStatusValues:
    """Positional 1, per Template:AfC submission/doc templatedata."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("{{AfC submission||ns=118|u=Contributor 108|ts=20260823213730}}", SubmissionStatus.PENDING),
            ("{{AfC submission|T}}", SubmissionStatus.DRAFT),
            ("{{AfC submission|t|ts=}}", SubmissionStatus.DRAFT),
            (
                "{{AfC submission|d|nn|u=A|ns=118|decliner=B|declinets=20260708144901|ts=20260708144743}}",
                SubmissionStatus.DECLINED,
            ),
            (
                "{{AfC submission|R|u=A|ns=118|reviewer=B|reviewts=20260813185457|ts=20260813185322}}",
                SubmissionStatus.REVIEWING,
            ),
            (
                # `r` is *reviewing*; a rejection is `d` plus reject=yes.
                "{{AfC submission|d|n|u=Contributor 106|ns=118|decliner=Reviewer 105|declinets=20260430193543|reject=yes|small=yes|ts=20260218013338}}",
                SubmissionStatus.REJECTED,
            ),
            ("{{AfC submission|zz|ts=20260218013338}}", SubmissionStatus.UNKNOWN),
        ],
    )
    def test_status(self, text, expected):
        assert iter_afc_templates(text)[0].status is expected

    def test_pending_template_sets_is_pending(self):
        text = (
            "{{AfC submission||ns=118|u=A|ts=20260823213730}}\n"
            "{{AfC submission|d|nn|u=A|ns=118|decliner=B|declinets=20260708144901|ts=20260708144743}}"
        )
        result = parse_declines(text)
        assert result.is_pending is True
        assert len(result.declines) == 1

    def test_rejection_still_counts_as_a_decline(self):
        text = "{{AfC submission|d|n|u=Contributor 106|ns=118|decliner=Reviewer 105|declinets=20260430193543|reject=yes|ts=20260218013338}}"
        result = parse_declines(text)
        assert len(result.declines) == 1
        assert result.stats.rejected_found == 1


class TestParameterOrder:
    """Named and positional arguments are interleaved in real pages."""

    def test_named_before_positional_reason_code(self):
        # Draft:4fnnann: `u=` comes before the reason code `ns`, so positions
        # must be counted over the non-named arguments only.
        text = "{{AfC submission|d|u=Contributor 104|ns|ns=118|ts=202607231436|decliner=Reviewer 102|declinets=20260723144903|reason2=music}}"
        template = iter_afc_templates(text)[0]
        assert template.status is SubmissionStatus.DECLINED
        assert template.arg(2) == "ns"
        assert template.named["reason2"] == "music"

        d = parse_declines(text).declines[0]
        assert d.code == "ns"
        assert d.secondary_code == "music"
        assert d.submitted_at == ts("20260723143600")

    def test_unparseable_ts_is_skipped_and_counted(self):
        # Draft:1GLOBAL -- ts is empty; there is no instant to recover.
        text = "{{AfC submission|d|corp|ns=118|decliner=Reviewer 004|declinets=20260817152619|ts=}}"
        result = parse_declines(text)
        assert result.declines == []
        assert result.stats.skip_reasons["bad_ts"] == 1

    def test_explicit_numbered_parameter(self):
        text = "{{AfC submission|1=d|2=nn|u=A|ns=118|decliner=B|declinets=20260708144901|ts=20260708144743}}"
        result = parse_declines(text)
        assert len(result.declines) == 1
        assert result.declines[0].code == "nn"

    def test_third_positional_is_detail_not_a_second_reason(self):
        # `exists|Ashenda` -- "Ashenda" is the conflicting article title.
        text = "{{AfC submission|d|exists|Ashenda|u=Contributor 105|ns=118|decliner=Reviewer 101|declinets=20260817012145|ts=20260815042655}}"
        result = parse_declines(text)
        d = result.declines[0]
        assert d.code == "exists"
        assert d.secondary_code is None  # NOT "Ashenda"
        assert d.family is ReasonFamily.STRUCTURAL

    def test_reason2_is_the_secondary_code(self):
        text = "{{AfC submission|d|ns|u=A|ns=118|decliner=Reviewer 102|declinets=20260624033939|reason2=music|ts=20260624032817}}"
        d = parse_declines(text).declines[0]
        assert d.code == "ns"
        assert d.secondary_code == "music"

    def test_small_and_demo_do_not_change_parsing(self):
        text = "{{AFC submission|d|v|u=Contributor 101|ns=118|demo=|decliner=Reviewer 104|declinets=20260511050057|small=yes|ts=20260210231741}}"
        d = parse_declines(text).declines[0]
        assert d.code == "v"
        assert d.decliner == "Reviewer 104"

    def test_uppercase_template_name(self):
        text = "{{AFC submission|d|corp|u=A|ns=118|decliner=B|declinets=20260405065741|ts=20260203110405}}"
        assert len(parse_declines(text).declines) == 1


class TestFreeTextDetails:
    def test_detail_containing_wikilinks_and_pipes(self):
        text = (
            "{{AfC submission|d|reason|lease read [[WP:REFB]] and [[WP:CITE|how to cite]]. "
            "I appreciate your first attempt.|u=Contributor 002|ns=118|decliner=Reviewer 004|"
            "declinets=20260819175000|small=yes|ts=20260819170000}}"
        )
        d = parse_declines(text).declines[0]
        assert d.code == "reason"
        assert d.decliner == "Reviewer 004"
        assert d.submitted_at == ts("20260819170000")

    def test_detail_containing_a_nested_template(self):
        # Draft:2San -- the detail parameter transcludes {{tl|citation needed}}.
        text = (
            "{{AFC submission|d|reason|Your edits have improved the article. "
            "I have marked several claims with {{tl|citation needed}} or {{tl|elaborate}}.|"
            "u=Editorino123|ns=118|decliner=Reviewer 003|declinets=20260819175912|ts=20260816170602}}"
        )
        templates = iter_afc_templates(text)
        assert len(templates) == 1  # the nested {{tl|...}} is not a submission
        d = parse_declines(text).declines[0]
        assert d.code == "reason"
        assert d.decliner == "Reviewer 003"


class TestSkipsAreCountedNotSwallowed:
    def test_missing_decliner(self):
        text = "{{AfC submission|d|v|u=Contributor 102|ns=118|small=yes|ts=20260821152456}}"
        result = parse_declines(text)
        assert result.declines == []
        assert result.stats.declines_found == 1
        assert result.stats.declines_skipped == 1
        assert result.stats.skip_reasons == {"missing_decliner": 1}
        assert result.stats.had_failure is True
        assert result.stats.skip_rate == 1.0

    def test_missing_declinets(self):
        text = "{{AfC submission|d|ai|u=Contributor 107|ns=118|decliner=Reviewer 105|small=yes|ts=20260821152456}}"
        result = parse_declines(text)
        assert result.stats.skip_reasons == {"bad_declinets": 1}

    def test_bare_decline_with_no_metadata(self):
        text = "{{AFC submission|d|small=yes|ts=20251201002517}}"
        result = parse_declines(text)
        assert result.stats.declines_skipped == 1
        assert result.stats.skip_reasons == {"missing_decliner": 1}

    def test_good_and_bad_in_one_stack(self):
        text = (
            "{{AfC submission|d|nn|u=A|ns=118|decliner=Reviewer 003|declinets=20260708144901|ts=20260708144743}}\n"
            "{{AfC submission|d|v|u=A|ns=118|small=yes|ts=20260821152456}}\n"
        )
        result = parse_declines(text)
        assert len(result.declines) == 1
        assert result.stats.declines_found == 2
        assert result.stats.declines_skipped == 1
        assert result.stats.skip_rate == 0.5

    def test_empty_reason_code_is_kept_but_flagged(self):
        # The decline happened and we know who and when; only the code is
        # absent. models.py resolves an unknown code to UNKNOWN, never a guess.
        text = "{{AfC submission|d||u=Contributor 103|ns=118|decliner=Reviewer 004|declinets=20260819191212|ts=20260817045355}}"
        result = parse_declines(text)
        assert len(result.declines) == 1
        assert result.declines[0].code == ""
        assert result.declines[0].family is ReasonFamily.UNKNOWN
        assert result.declines[0].fixable_by_rewrite is None
        assert result.stats.empty_code == 1

    def test_unmapped_code_is_counted(self):
        # `cv` is a real, current AFCH code that analysis/reasons.py deliberately
        # refuses to classify (its remedy is split across two families), so it
        # stays out of REASON_FAMILY. Do not use a code like `prof` here: those
        # were unmapped only while REASON_FAMILY was a partial seed.
        text = "{{AfC submission|d|cv|u=A|ns=118|decliner=B|declinets=20260819191212|ts=20260817045355}}"
        result = parse_declines(text)
        assert result.declines[0].family is ReasonFamily.UNKNOWN
        assert result.stats.unmapped_codes == {"cv": 1}


class TestNoTemplates:
    def test_empty_page(self):
        result = parse_declines("")
        assert result.declines == []
        assert result.stats.templates_found == 0
        assert result.stats.had_failure is False

    def test_page_without_afc_templates(self):
        result = parse_declines("== Heading ==\nJust prose with {{Infobox person}}.")
        assert result.declines == []
        assert result.stats.templates_found == 0

    def test_unbalanced_braces_do_not_raise(self):
        result = parse_declines("{{AfC submission|d|nn|u=A|decliner=B|declinets=20260708144901")
        assert result.declines == []

    def test_afc_submission_draft_subpage_is_a_different_template(self):
        result = parse_declines("{{AfC submission/draft}}")
        assert result.stats.templates_found == 0


class TestInertRegions:
    def test_template_inside_an_html_comment_is_ignored(self):
        text = (
            "<!-- {{AfC submission|d|nn|u=A|ns=118|decliner=B|declinets=20260708144901|ts=20260708144743}} -->\n"
            "{{AfC submission|d|ai|u=A|ns=118|decliner=C|declinets=20260627045213|ts=20260627032430}}"
        )
        result = parse_declines(text)
        assert len(result.declines) == 1
        assert result.declines[0].decliner == "C"

    def test_template_inside_nowiki_is_ignored(self):
        text = "<nowiki>{{AfC submission|d|nn|u=A|decliner=B|declinets=20260708144901|ts=20260708144743}}</nowiki>"
        assert parse_declines(text).stats.templates_found == 0


class TestOrdering:
    def test_sorted_by_declinets_regardless_of_document_order(self):
        text = (
            "{{AfC submission|d|nn|u=A|ns=118|decliner=Third|declinets=20260708144901|ts=20260708144743}}\n"
            "{{AfC submission|d|v|u=A|ns=118|decliner=First|declinets=20260627045213|ts=20260627032430}}\n"
            "{{AfC submission|d|ai|u=A|ns=118|decliner=Second|declinets=20260629205711|ts=20260629172235}}\n"
        )
        result = parse_declines(text)
        assert [d.decliner for d in result.declines] == ["First", "Second", "Third"]


class TestDatasetRoundTrip:
    """collect.py must be able to write and read back the contract types."""

    def test_history_survives_json(self):
        import json

        from first_reader.models import DraftHistory, Revision
        from first_reader.wiki.collect import history_from_dict, history_to_dict

        original = DraftHistory(
            title="Draft:0→1 Doctrine for AI",
            pageid=83560144,
            declines=parse_declines(DOCTRINE_STACK).declines,
            revisions=[
                Revision(
                    revid=1361228266,
                    parentid=0,
                    user="Contributor 001",
                    timestamp=ts("20260626145800"),
                    comment="Live page",
                    size=5355,
                )
            ],
            is_pending=False,
        )
        restored = history_from_dict(json.loads(json.dumps(history_to_dict(original))))

        assert restored.title == original.title
        assert restored.pageid == original.pageid
        assert restored.is_pending == original.is_pending
        assert restored.declines == original.declines
        assert restored.revisions == original.revisions
        assert restored.latest_decline.decliner == "Reviewer 003"
