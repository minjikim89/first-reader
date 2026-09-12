"""Async MediaWiki API client for the English Wikipedia.

Scope: read-only. First Reader never writes to Wikipedia, so nothing here
takes a token or issues a POST that changes state.

Three things this module is careful about:

* Rate limits. Requests are bounded by a semaphore, spaced by a minimum
  interval, tagged with ``maxlag``, and retried with exponential backoff that
  honours ``Retry-After``.
* Caching. Every response is stored as JSON under ``data/cache`` keyed by the
  request parameters, so re-running collection or tests does not re-hit the
  API.
* Continuation. Every list endpoint here paginates via the ``continue``
  blob rather than assuming one page is the whole answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from first_reader.models import Revision

log = logging.getLogger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "FirstReader/0.1 (https://github.com/minjikim89/first-reader)"

# The API caps a multi-title query at 50 for unauthenticated clients.
MAX_TITLES_PER_REQUEST = 50
# rvlimit=500 is the anonymous ceiling for a revision listing.
MAX_REVISIONS_PER_REQUEST = 500
# cmlimit=500 likewise.
MAX_CATEGORY_MEMBERS_PER_REQUEST = 500

DEFAULT_CACHE_DIR = Path("data/cache")

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
# MediaWiki error codes that mean "come back later", not "your request is wrong".
_RETRYABLE_ERROR_CODES = frozenset({"maxlag", "readonly", "ratelimited", "internal_api_error_DBQueryError"})


class WikiAPIError(RuntimeError):
    """The API answered, but with an error we are not going to retry."""

    def __init__(self, code: str, info: str, params: dict[str, Any]) -> None:
        super().__init__(f"{code}: {info}")
        self.code = code
        self.info = info
        self.params = params


@dataclass(frozen=True)
class CategoryMember:
    pageid: int
    ns: int
    title: str


def parse_iso_timestamp(value: str) -> datetime:
    """MediaWiki hands back ``2026-06-26T14:58:00Z``."""
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


class _Throttle:
    """Minimum spacing between request starts, shared across coroutines."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def wait(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            if delay > 0:
                await asyncio.sleep(delay)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval

    async def pause(self, seconds: float) -> None:
        """Push the whole client back, e.g. after a 429."""
        async with self._lock:
            self._next_allowed = max(self._next_allowed, time.monotonic() + seconds)


class WikiClient:
    """Read-only async client for the MediaWiki action API.

    Use as an async context manager::

        async with WikiClient() as client:
            text = await client.get_wikitext("Draft:Example")
    """

    def __init__(
        self,
        api_url: str = API_URL,
        *,
        user_agent: str = USER_AGENT,
        cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
        max_concurrency: int = 4,
        min_interval: float = 0.05,
        max_retries: int = 5,
        timeout: float = 30.0,
        maxlag: int | None = 5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_url = api_url
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.max_retries = max_retries
        self.maxlag = maxlag
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._throttle = _Throttle(min_interval)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip"},
            follow_redirects=True,
        )
        self.cache_hits = 0
        self.requests_made = 0

    async def __aenter__(self) -> WikiClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------ cache

    def _cache_path(self, params: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        blob = json.dumps(params, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(f"{self.api_url}|{blob}".encode()).hexdigest()
        action = str(params.get("action", "query"))
        sub = str(params.get("list") or params.get("prop") or action)
        # Keep the tree shallow but not a single flat directory of 100k files.
        return self.cache_dir / sub.replace("|", "+")[:40] / digest[:2] / f"{digest}.json"

    @staticmethod
    def _read_cache(path: Path) -> dict[str, Any] | None:
        try:
            with path.open(encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_cache(path: Path, payload: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            tmp.replace(path)
        except OSError:  # a cache we cannot write is not a failure
            log.debug("could not write cache file %s", path, exc_info=True)

    # ------------------------------------------------------------- transport

    async def get(self, params: dict[str, Any], *, use_cache: bool = True) -> dict[str, Any]:
        """One API call, cached and rate-limited. Returns the decoded JSON."""
        full = {"format": "json", "formatversion": "2", **params}
        cache_key = dict(full)
        if self.maxlag is not None:
            full["maxlag"] = str(self.maxlag)

        path = self._cache_path(cache_key) if use_cache else None
        if path is not None and path.exists():
            cached = self._read_cache(path)
            if cached is not None:
                self.cache_hits += 1
                return cached

        payload = await self._request_with_retry(full)
        if path is not None:
            self._write_cache(path, payload)
        return payload

    async def _request_with_retry(self, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            backoff = min(2.0**attempt, 30.0)
            try:
                async with self._semaphore:
                    await self._throttle.wait()
                    self.requests_made += 1
                    response = await self._client.get(self.api_url, params=params)
            except (httpx.TransportError, httpx.HTTPError) as exc:
                last_error = exc
                log.warning("transport error (attempt %d): %s", attempt + 1, exc)
                await asyncio.sleep(backoff)
                continue

            if response.status_code in _RETRYABLE_STATUS:
                retry_after = _retry_after_seconds(response) or backoff
                log.warning(
                    "HTTP %s from API, backing off %.1fs (attempt %d)",
                    response.status_code,
                    retry_after,
                    attempt + 1,
                )
                await self._throttle.pause(retry_after)
                await asyncio.sleep(retry_after)
                last_error = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}", request=response.request, response=response
                )
                continue

            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:
                last_error = exc
                await asyncio.sleep(backoff)
                continue

            error = payload.get("error")
            if error:
                code = str(error.get("code", ""))
                if code in _RETRYABLE_ERROR_CODES:
                    wait = float(error.get("lag", 0) or 0) or backoff
                    wait = min(max(wait, 1.0), 30.0)
                    log.warning("API error %s, backing off %.1fs", code, wait)
                    await self._throttle.pause(wait)
                    await asyncio.sleep(wait)
                    last_error = WikiAPIError(code, str(error.get("info", "")), params)
                    continue
                raise WikiAPIError(code, str(error.get("info", "")), params)

            return payload

        raise RuntimeError(f"gave up after {self.max_retries + 1} attempts: {last_error}")

    async def _paginate(
        self, params: dict[str, Any], *, use_cache: bool = True
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield each response page, following the ``continue`` blob."""
        cont: dict[str, Any] = {}
        while True:
            payload = await self.get({**params, **cont}, use_cache=use_cache)
            yield payload
            cont = payload.get("continue") or {}
            if not cont:
                return

    # ------------------------------------------------------------- endpoints

    async def iter_category_members(
        self,
        category: str,
        *,
        member_type: str = "page",
        namespace: int | None = None,
        limit: int | None = None,
        use_cache: bool = True,
    ) -> AsyncIterator[CategoryMember]:
        """Enumerate pages in a category, following pagination.

        ``category`` may be given with or without the ``Category:`` prefix.
        """
        title = category if category.lower().startswith("category:") else f"Category:{category}"
        params: dict[str, Any] = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": title,
            "cmlimit": str(MAX_CATEGORY_MEMBERS_PER_REQUEST),
            "cmtype": member_type,
        }
        if namespace is not None:
            params["cmnamespace"] = str(namespace)

        yielded = 0
        async for payload in self._paginate(params, use_cache=use_cache):
            for member in payload.get("query", {}).get("categorymembers", []):
                yield CategoryMember(
                    pageid=member.get("pageid", 0), ns=member.get("ns", 0), title=member["title"]
                )
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    async def get_category_members(
        self, category: str, *, limit: int | None = None, **kwargs: Any
    ) -> list[CategoryMember]:
        return [m async for m in self.iter_category_members(category, limit=limit, **kwargs)]

    async def get_wikitext(self, title: str, *, use_cache: bool = True) -> str | None:
        """Current wikitext of one page, or None if it is missing."""
        result = await self.get_wikitext_batch([title], use_cache=use_cache)
        return result.get(title)

    async def get_wikitext_batch(
        self, titles: Sequence[str], *, use_cache: bool = True
    ) -> dict[str, str]:
        """Current wikitext for up to 50 titles per API round-trip.

        Keys are the titles the API returns (normalised), plus the titles as
        requested where normalisation changed them, so callers can look up
        either form. Missing or empty pages are simply absent.
        """
        out: dict[str, str] = {}
        for chunk in _chunks(titles, MAX_TITLES_PER_REQUEST):
            params = {
                "action": "query",
                "prop": "revisions",
                "rvslots": "main",
                "rvprop": "content|ids|timestamp",
                "titles": "|".join(chunk),
            }
            async for payload in self._paginate(params, use_cache=use_cache):
                query = payload.get("query", {})
                normalised = {n["to"]: n["from"] for n in query.get("normalized", [])}
                for page in query.get("pages", []):
                    if page.get("missing") or not page.get("revisions"):
                        continue
                    slot = page["revisions"][0].get("slots", {}).get("main", {})
                    content = slot.get("content")
                    if content is None:
                        continue
                    out[page["title"]] = content
                    original = normalised.get(page["title"])
                    if original:
                        out[original] = content
        return out

    async def get_page_info(
        self, titles: Sequence[str], *, use_cache: bool = True
    ) -> dict[str, dict[str, Any]]:
        """pageid / ns / missing for up to 50 titles per round-trip."""
        out: dict[str, dict[str, Any]] = {}
        for chunk in _chunks(titles, MAX_TITLES_PER_REQUEST):
            payload = await self.get(
                {"action": "query", "prop": "info", "titles": "|".join(chunk)},
                use_cache=use_cache,
            )
            query = payload.get("query", {})
            normalised = {n["to"]: n["from"] for n in query.get("normalized", [])}
            for page in query.get("pages", []):
                out[page["title"]] = page
                original = normalised.get(page["title"])
                if original:
                    out[original] = page
        return out

    async def get_revisions(
        self,
        title: str,
        *,
        limit: int | None = None,
        newer_first: bool = False,
        use_cache: bool = True,
    ) -> list[Revision]:
        """Full revision history, oldest first by default.

        ``limit`` caps the number of revisions fetched; None means all of them
        (paginating in blocks of 500).
        """
        params: dict[str, Any] = {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvprop": "ids|timestamp|user|comment|size",
            "rvlimit": str(MAX_REVISIONS_PER_REQUEST),
            "rvdir": "older" if newer_first else "newer",
        }
        revisions: list[Revision] = []
        async for payload in self._paginate(params, use_cache=use_cache):
            for page in payload.get("query", {}).get("pages", []):
                if page.get("missing"):
                    return []
                for rev in page.get("revisions", []):
                    revisions.append(
                        Revision(
                            revid=rev.get("revid", 0),
                            parentid=rev.get("parentid", 0),
                            # Deleted/suppressed fields come back as flags with
                            # the value absent; keep the record, blank the field.
                            user=rev.get("user", ""),
                            timestamp=parse_iso_timestamp(rev["timestamp"]),
                            comment=rev.get("comment", ""),
                            size=rev.get("size", 0),
                        )
                    )
                    if limit is not None and len(revisions) >= limit:
                        return revisions
        return revisions

    async def compare(
        self,
        *,
        from_rev: int | None = None,
        to_rev: int | None = None,
        from_title: str | None = None,
        to_relative: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """``action=compare`` between two revisions.

        Returns the ``compare`` object; ``body`` holds the diff as HTML table
        rows. Either give ``from_rev``/``to_rev``, or ``from_title`` with
        ``to_relative`` (``prev``/``next``/``cur``).
        """
        params: dict[str, Any] = {"action": "compare", "prop": "diff|ids|title|size"}
        if from_rev is not None:
            params["fromrev"] = str(from_rev)
        if from_title is not None:
            params["fromtitle"] = from_title
        if to_rev is not None:
            params["torev"] = str(to_rev)
        if to_relative is not None:
            params["torelative"] = to_relative
        payload = await self.get(params, use_cache=use_cache)
        return payload.get("compare", {})

    async def get_diff_html(self, from_rev: int, to_rev: int, *, use_cache: bool = True) -> str:
        result = await self.compare(from_rev=from_rev, to_rev=to_rev, use_cache=use_cache)
        return result.get("body", "")

    async def get_revision_content(
        self, revids: Sequence[int], *, use_cache: bool = True
    ) -> dict[int, str]:
        """Wikitext of specific revisions, up to 50 per round-trip."""
        out: dict[int, str] = {}
        for chunk in _chunks([str(r) for r in revids], MAX_TITLES_PER_REQUEST):
            payload = await self.get(
                {
                    "action": "query",
                    "prop": "revisions",
                    "rvslots": "main",
                    "rvprop": "content|ids",
                    "revids": "|".join(chunk),
                },
                use_cache=use_cache,
            )
            for page in payload.get("query", {}).get("pages", []):
                for rev in page.get("revisions", []):
                    content = rev.get("slots", {}).get("main", {}).get("content")
                    if content is not None:
                        out[rev["revid"]] = content
        return out


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(1.0, float(raw))
    except ValueError:
        return None


def _chunks(items: Iterable[str], size: int) -> list[list[str]]:
    seq = list(items)
    return [seq[i : i + size] for i in range(0, len(seq), size)]
