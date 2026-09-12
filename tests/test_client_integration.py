"""Integration tests against the live English Wikipedia API.

Run with ``pytest -m integration``. These are excluded from a default run.

They assert against ``Draft:0→1 Doctrine for AI``, a draft declined three
times in mid-2026. If that page is later accepted, moved or blanked, the
history assertions here will need a new reference draft -- the revision ids
asserted below are immutable, so they pin the exact content that was verified.
"""

from __future__ import annotations

from datetime import timezone

import pytest

from first_reader.models import ReasonFamily
from first_reader.wiki.client import WikiClient
from first_reader.wiki.parser import fetch_draft_history, parse_declines

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

DRAFT = "Draft:0→1 Doctrine for AI"
PAGEID = 83560144
FIRST_REVID = 1361228266  # the draft's first revision, 2026-06-26T14:58:00Z

# Reviewer account names are deliberately not asserted. They are public record,
# but the shipped dataset labels them, and a test that pins three volunteers by
# name would undo that in the one file nobody thinks to check. The parse is
# fully determined by the codes and timestamps.
EXPECTED_DECLINES = [
    ("ai", "nn", "20260627032430", "20260627045213"),
    ("v", "ai", "20260629172235", "20260629205711"),
    ("nn", None, "20260708144743", "20260708144901"),
]


@pytest.fixture
async def client(tmp_path):
    # A per-test cache dir, so these tests really exercise the network.
    async with WikiClient(cache_dir=tmp_path / "cache") as c:
        yield c


async def test_get_wikitext(client):
    text = await client.get_wikitext(DRAFT)
    assert text is not None
    assert "{{AfC submission|d|nn|" in text


async def test_get_wikitext_missing_page(client):
    text = await client.get_wikitext("Draft:This page does not exist 8f3a9c21")
    assert text is None


async def test_get_wikitext_batch_is_one_round_trip(client):
    titles = [DRAFT, "Wikipedia:Articles for creation"]
    before = client.requests_made
    texts = await client.get_wikitext_batch(titles)
    assert client.requests_made - before == 1
    assert set(titles) <= set(texts)


async def test_cache_prevents_a_second_request(client):
    await client.get_wikitext(DRAFT)
    before = client.requests_made
    await client.get_wikitext(DRAFT)
    assert client.requests_made == before
    assert client.cache_hits >= 1


async def test_category_members_paginate(client):
    members = await client.get_category_members(
        "Category:Pending AfC submissions", limit=600
    )
    # cmlimit caps at 500, so >500 results proves the continue blob was followed.
    assert len(members) > 500
    assert len({m.title for m in members}) == len(members)
    assert all(m.pageid > 0 for m in members)


async def test_category_name_without_prefix_works(client):
    members = await client.get_category_members("Pending AfC submissions", limit=5)
    assert len(members) == 5


async def test_get_revisions_oldest_first(client):
    revisions = await client.get_revisions(DRAFT)
    assert len(revisions) >= 11
    assert revisions[0].revid == FIRST_REVID
    assert revisions[0].parentid == 0
    assert revisions[0].user  # present, not asserted by name
    assert revisions[0].timestamp.tzinfo is timezone.utc
    timestamps = [r.timestamp for r in revisions]
    assert timestamps == sorted(timestamps)


async def test_get_revisions_missing_page(client):
    assert await client.get_revisions("Draft:This page does not exist 8f3a9c21") == []


async def test_compare_two_revisions(client):
    revisions = await client.get_revisions(DRAFT, limit=3)
    result = await client.compare(
        from_rev=revisions[0].revid, to_rev=revisions[1].revid
    )
    assert result["fromrevid"] == revisions[0].revid
    assert result["torevid"] == revisions[1].revid
    assert "diff-" in result["body"]


async def test_parse_live_draft_matches_expected_declines(client):
    """The acceptance check: three declines, in order, with exact fields."""
    history, stats = await fetch_draft_history(client, DRAFT)

    assert history is not None
    assert history.pageid == PAGEID
    assert history.decline_count == 3
    assert stats.declines_skipped == 0

    actual = [
        (
            d.code,
            d.secondary_code,
            d.submitted_at.strftime("%Y%m%d%H%M%S"),
            d.declined_at.strftime("%Y%m%d%H%M%S"),
        )
        for d in history.declines
    ]
    assert actual == EXPECTED_DECLINES

    assert history.declines[0].family is ReasonFamily.WRITING
    assert history.declines[0].fixable_by_rewrite is True
    assert history.latest_decline.decliner  # present, not asserted by name
    assert len(history.revisions) >= 11


async def test_live_wikitext_parses_identically_offline(client):
    """Whatever the API returns, the parser is pure: same text, same result."""
    text = await client.get_wikitext(DRAFT)
    a = parse_declines(text)
    b = parse_declines(text)
    assert a.declines == b.declines
