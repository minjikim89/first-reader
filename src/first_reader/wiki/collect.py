"""Build ``data/dataset.jsonl``: drafts that have been declined more than once.

Two passes, because they cost different amounts:

1. *Screen* -- enumerate the source categories and pull current wikitext 50
   titles per request, parsing only the template stack. Cheap: ~20 requests
   per 1,000 drafts.
2. *Collect* -- for the drafts that clear ``--min-declines``, fetch the full
   revision history. One request per draft, so this is the expensive half and
   only runs on drafts we are keeping.

Both passes are resumable. Screened titles and collected records are appended
to disk as they are produced, and a re-run skips anything already there.

Run::

    python -m first_reader.wiki.collect --target 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from first_reader.models import Decline, DraftHistory, Revision
from first_reader.wiki.client import WikiClient
from first_reader.wiki.parser import ParseStats, parse_declines

DEFAULT_CATEGORIES = [
    # Primary source: drafts sitting in the review queue right now.
    "Category:Pending AfC submissions",
    # Supplement. The pending category holds ~1,000 drafts of which ~26% have
    # been declined twice or more (~260), which is short of a 300-draft target,
    # so declined-but-not-resubmitted drafts make up the difference. The
    # `is_pending` flag on each record keeps the two distinguishable.
    "Category:Declined AfC submissions",
]

DEFAULT_OUT = Path("data/dataset.jsonl")
DEFAULT_ANONYMIZED = Path("data/dataset.anonymized.jsonl")


# --------------------------------------------------------------- serialisation


def decline_to_dict(decline: Decline) -> dict[str, object]:
    return {
        "code": decline.code,
        "secondary_code": decline.secondary_code,
        "decliner": decline.decliner,
        "submitted_at": decline.submitted_at.isoformat(),
        "declined_at": decline.declined_at.isoformat(),
        "raw": decline.raw,
    }


def decline_from_dict(data: dict) -> Decline:
    return Decline(
        code=data["code"],
        secondary_code=data["secondary_code"],
        decliner=data["decliner"],
        submitted_at=datetime.fromisoformat(data["submitted_at"]),
        declined_at=datetime.fromisoformat(data["declined_at"]),
        raw=data["raw"],
    )


def revision_to_dict(revision: Revision) -> dict[str, object]:
    return {
        "revid": revision.revid,
        "parentid": revision.parentid,
        "user": revision.user,
        "timestamp": revision.timestamp.isoformat(),
        "comment": revision.comment,
        "size": revision.size,
    }


def revision_from_dict(data: dict) -> Revision:
    return Revision(
        revid=data["revid"],
        parentid=data["parentid"],
        user=data["user"],
        timestamp=datetime.fromisoformat(data["timestamp"]),
        comment=data["comment"],
        size=data["size"],
    )


def history_to_dict(history: DraftHistory) -> dict[str, object]:
    return {
        "title": history.title,
        "pageid": history.pageid,
        "is_pending": history.is_pending,
        "declines": [decline_to_dict(d) for d in history.declines],
        "revisions": [revision_to_dict(r) for r in history.revisions],
    }


def history_from_dict(data: dict) -> DraftHistory:
    return DraftHistory(
        title=data["title"],
        pageid=data["pageid"],
        declines=[decline_from_dict(d) for d in data["declines"]],
        revisions=[revision_from_dict(r) for r in data["revisions"]],
        is_pending=data.get("is_pending", False),
    )


def load_dataset(path: Path | str | None = None) -> list[DraftHistory]:
    """Read a collected dataset back into ``DraftHistory`` objects.

    Defaults to the distributed file, not the raw one. The raw collection
    carries real reviewer and contributor handles and is gitignored, so a
    fresh clone has only ``dataset.anonymized.jsonl`` -- and the numbers in
    the README have to be the numbers that clone reproduces. Resolving to the
    raw file when it happens to be present would mean we and a reader are
    measuring different sets without either of us noticing.
    """
    if path is None:
        path = DEFAULT_ANONYMIZED if DEFAULT_ANONYMIZED.exists() else DEFAULT_OUT
    path = Path(path)
    if not path.exists():
        return []
    out: list[DraftHistory] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(history_from_dict(json.loads(line)))
    return out


# ------------------------------------------------------------------- resuming


@dataclass
class ResumeState:
    collected: dict[str, int]  # title -> decline count
    screened: dict[str, int]  # title -> decline count (0 means "not enough")

    @property
    def known(self) -> set[str]:
        return set(self.screened) | set(self.collected)


def _screen_ledger_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".screened.jsonl")


def _stats_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".stats.json")


def load_resume_state(out: Path) -> ResumeState:
    collected: dict[str, int] = {}
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated final line from an interrupted run
                collected[record["title"]] = len(record.get("declines", []))

    screened: dict[str, int] = {}
    ledger = _screen_ledger_path(out)
    if ledger.exists():
        with ledger.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                screened[record["title"]] = record.get("declines", 0)
    return ResumeState(collected=collected, screened=screened)


# ---------------------------------------------------------------- collection


def _write_summary(
    out: Path,
    *,
    collected: int,
    drafts_screened: int,
    drafts_with_parse_failure: int,
    stats: ParseStats,
) -> dict:
    """Accumulate the summary across resumed runs, then write it.

    A resumed run only screens the drafts it has not seen, so a per-run
    failure rate would be computed over a shrinking denominator. Totals are
    carried forward from the previous summary so the rate stays a statement
    about the whole dataset.
    """
    path = _stats_path(out)
    previous: dict = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}

    prior_stats = ParseStats()
    for key, value in (previous.get("parse_stats") or {}).items():
        if isinstance(value, dict):
            getattr(prior_stats, key).update(value)
        elif isinstance(value, int):
            setattr(prior_stats, key, value)
    prior_stats.merge(stats)

    screened = previous.get("drafts_screened", 0) + drafts_screened
    failed = previous.get("drafts_with_parse_failure", 0) + drafts_with_parse_failure
    summary = {
        "collected": collected,
        "drafts_screened": screened,
        "drafts_with_parse_failure": failed,
        "draft_parse_failure_rate": failed / screened if screened else 0.0,
        "template_skip_rate": prior_stats.skip_rate,
        "parse_stats": prior_stats.to_dict(),
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


async def _gather_titles(
    client: WikiClient, categories: Sequence[str], limit_per_category: int | None
) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for category in categories:
        count = 0
        async for member in client.iter_category_members(
            category, limit=limit_per_category
        ):
            if member.title in seen:
                continue
            seen.add(member.title)
            titles.append(member.title)
            count += 1
        print(f"  {category}: {count} pages", file=sys.stderr)
    return titles


def _chunks(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def collect(
    *,
    categories: Sequence[str] = tuple(DEFAULT_CATEGORIES),
    out: Path = DEFAULT_OUT,
    min_declines: int = 2,
    target: int = 300,
    limit_per_category: int | None = None,
    revision_limit: int | None = None,
    use_cache: bool = True,
    concurrency: int = 4,
) -> ParseStats:
    out.parent.mkdir(parents=True, exist_ok=True)
    state = load_resume_state(out)
    ledger = _screen_ledger_path(out)

    stats = ParseStats()
    kept = len(state.collected)
    print(
        f"resume: {kept} collected, {len(state.screened)} screened already",
        file=sys.stderr,
    )
    if kept >= target:
        print(f"target {target} already met ({kept} records)", file=sys.stderr)
        return stats

    async with WikiClient(
        cache_dir=Path("data/cache") if use_cache else None,
        max_concurrency=concurrency,
    ) as client:
        print("enumerating categories...", file=sys.stderr)
        titles = await _gather_titles(client, categories, limit_per_category)
        todo = [t for t in titles if t not in state.known]
        print(
            f"{len(titles)} candidates, {len(todo)} not yet screened", file=sys.stderr
        )

        screened = 0
        drafts_with_parse_failure = 0
        drafts_screened_total = 0

        with (
            out.open("a", encoding="utf-8") as out_fh,
            ledger.open("a", encoding="utf-8") as ledger_fh,
        ):
            for batch in _chunks(todo, 50):
                if kept >= target:
                    break

                texts = await client.get_wikitext_batch(batch)
                info = await client.get_page_info(batch)

                selected: list[tuple[str, DraftHistory]] = []
                for title in batch:
                    wikitext = texts.get(title)
                    screened += 1
                    if wikitext is None:
                        stats.skip_reasons["page_missing"] += 1
                        ledger_fh.write(
                            json.dumps({"title": title, "declines": -1}) + "\n"
                        )
                        continue

                    result = parse_declines(wikitext)
                    stats.merge(result.stats)
                    drafts_screened_total += 1
                    if result.stats.had_failure:
                        drafts_with_parse_failure += 1

                    count = len(result.declines)
                    ledger_fh.write(
                        json.dumps({"title": title, "declines": count}) + "\n"
                    )
                    if count < min_declines:
                        continue

                    page = info.get(title, {})
                    selected.append(
                        (
                            title,
                            DraftHistory(
                                title=page.get("title", title),
                                pageid=page.get("pageid", 0),
                                declines=result.declines,
                                revisions=[],
                                is_pending=result.is_pending,
                            ),
                        )
                    )

                # Revision history: one request per kept draft, run in parallel
                # under the client's own concurrency bound.
                room = max(0, target - kept)
                selected = selected[:room] if room else []
                if selected:
                    revisions = await asyncio.gather(
                        *(
                            client.get_revisions(title, limit=revision_limit)
                            for title, _ in selected
                        )
                    )
                    for (_, history), revs in zip(selected, revisions):
                        history.revisions = revs
                        out_fh.write(
                            json.dumps(history_to_dict(history), ensure_ascii=False)
                            + "\n"
                        )
                        kept += 1

                out_fh.flush()
                ledger_fh.flush()
                print(
                    f"  screened {screened}/{len(todo)} | kept {kept}/{target} "
                    f"| requests {client.requests_made} cache-hits {client.cache_hits}",
                    file=sys.stderr,
                )

        summary = _write_summary(
            out,
            collected=kept,
            drafts_screened=drafts_screened_total,
            drafts_with_parse_failure=drafts_with_parse_failure,
            stats=stats,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)

    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="source category (repeatable). Defaults to pending + declined.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-declines", type=int, default=2)
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=None,
        help="cap how many members are enumerated from each category",
    )
    parser.add_argument("--revision-limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    asyncio.run(
        collect(
            categories=args.categories or DEFAULT_CATEGORIES,
            out=args.out,
            min_declines=args.min_declines,
            target=args.target,
            limit_per_category=args.limit_per_category,
            revision_limit=args.revision_limit,
            use_cache=not args.no_cache,
            concurrency=args.concurrency,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
