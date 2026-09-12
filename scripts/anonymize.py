#!/usr/bin/env python3
"""Replace every account name in the collected dataset with a stable label.

    python scripts/anonymize.py                       # -> data/dataset.anonymized.jsonl
    python scripts/anonymize.py --check OUT           # verify no original name survives
    python scripts/anonymize.py --salt-file PATH      # reuse a salt across runs

Why, when the edit histories are already public
-----------------------------------------------
Every decline in this file is on a public page under a real volunteer's name,
and none of it is secret. What changes is aggregation. Collecting 540 declined
drafts into one file turns information that was scattered across thousands of
page histories into a searchable index of *who gets declined for what* and *who
declines them*. That index did not exist before we made it, and redistributing
it is a different act from the individual edits being public.

Reproducibility does not require it. Anyone can rebuild the original with
`python -m first_reader.wiki.collect`, from the same public API. So the
anonymised file loses nothing a researcher needs and removes an artifact nobody
asked us to create.

What is preserved
-----------------
* Draft titles and page ids -- reproducibility depends on them.
* All timestamps, revision ids, sizes, edit summaries, decline codes.
* Identity *relations*: one person always gets the same label, across every
  draft. Analyses that depend on repeat reviewers -- "did the same reviewer
  decline this twice", "how many distinct reviewers touched this draft" --
  work unchanged on the anonymised file.

What this is not
----------------
This is de-identification, not anonymity, and the difference matters. The salt
is random per run and written to a file that is not distributed; without it the
labels do not resolve. But with the salt, or by re-collecting from the API and
matching on revision ids, the mapping is recoverable. The point is to stop the
distributed file from *being* the index, not to make re-identification
impossible -- against public data with immutable revision ids, nothing can.

Edit summaries are passed through unchanged except for account names appearing
in them. A summary that quotes a name in prose is not detected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DEFAULT_IN = Path("data/dataset.jsonl")
DEFAULT_OUT = Path("data/dataset.anonymized.jsonl")
DEFAULT_SALT = Path("data/anonymize-salt.txt")

BOT = re.compile(r"\bbot\b", re.IGNORECASE)


def load_or_make_salt(path: Path) -> str:
    """Reuse a salt if one exists, otherwise make one and save it.

    Kept out of the distributed file on purpose. `data/.gitignore` excludes it.
    """
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    salt = secrets.token_hex(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(salt + "\n", encoding="utf-8")
    path.chmod(0o600)
    return salt


def _digest(salt: str, name: str) -> str:
    return hashlib.sha256(f"{salt}:{name}".encode()).hexdigest()


# What a scrubbed account position must look like afterwards.
LABEL = re.compile(r"(?:Reviewer|Contributor|Bot) \d{3,}")

# Places an account name appears inside free text. Structured fields are not
# enough: `raw` holds the original wikitext, and a submitter who never left a
# revision (or submitted under a different account) reaches the file only
# through `|u=`. Missing one of these is how the first pass leaked 16 accounts,
# two of them IP addresses.
NAME_IN_TEXT = [
    re.compile(r"\|\s*u\s*=\s*([^|}\n]+)"),
    re.compile(r"\bdecliner\s*=\s*([^|}\n]+)"),
    re.compile(r"\[\[\s*User(?:\s+talk)?\s*:\s*([^\]|/#]+)", re.IGNORECASE),
    re.compile(r"\[\[\s*Special:Contributions/\s*([^\]|#]+)", re.IGNORECASE),
]

# An unregistered editor is identified by their IP. Wikipedia publishes those;
# a dataset of "who got declined for AI" that republishes them is a different
# object, so they are labelled like any other account.
IP_ADDRESS = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    r"|\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}\b"
)


def names_in_text(text: str) -> set[str]:
    """Every account name a piece of free text mentions."""
    found: set[str] = set()
    for pattern in NAME_IN_TEXT:
        for match in pattern.findall(text):
            name = match.strip()
            if name:
                found.add(name)
    found.update(m.strip() for m in IP_ADDRESS.findall(text))
    return found


def build_label_map(records: list[dict], salt: str) -> dict[str, str]:
    """One label per account, globally stable across the whole file.

    A label's role comes from what the account does anywhere in the set: an
    account that declines anything is a Reviewer even on drafts where it only
    edited. Ordering is by salted digest, so the numbering carries no
    information -- not activity, not collection order.

    Names are gathered from structured fields *and* from every piece of free
    text that ships, because scrubbing can only replace what the map knows.
    """
    decliners: set[str] = set()
    editors: set[str] = set()
    for record in records:
        for decline in record.get("declines", []):
            if decline.get("decliner"):
                decliners.add(decline["decliner"])
            if decline.get("raw"):
                editors.update(names_in_text(decline["raw"]))
        for revision in record.get("revisions", []):
            if revision.get("user"):
                editors.add(revision["user"])
            if revision.get("comment"):
                editors.update(names_in_text(revision["comment"]))

    editors -= decliners
    everyone = decliners | editors
    ordered = sorted(everyone, key=lambda n: _digest(salt, n))

    counters: Counter[str] = Counter()
    mapping: dict[str, str] = {}
    for name in ordered:
        if BOT.search(name):
            role = "Bot"
        elif name in decliners:
            role = "Reviewer"
        else:
            role = "Contributor"
        counters[role] += 1
        mapping[name] = f"{role} {counters[role]:03d}"
    return mapping


def _scrub(text: str, mapping: dict[str, str]) -> str:
    """Replace account names in free text, longest first so prefixes lose."""
    for name in sorted(mapping, key=len, reverse=True):
        if name in text:
            text = text.replace(name, mapping[name])
    return text


def anonymize_record(record: dict, mapping: dict[str, str]) -> dict:
    out = dict(record)
    out["declines"] = []
    for decline in record.get("declines", []):
        item = dict(decline)
        if item.get("decliner"):
            item["decliner"] = mapping.get(item["decliner"], item["decliner"])
        if item.get("raw"):
            item["raw"] = _scrub(item["raw"], mapping)
        out["declines"].append(item)

    out["revisions"] = []
    for revision in record.get("revisions", []):
        item = dict(revision)
        if item.get("user"):
            item["user"] = mapping.get(item["user"], item["user"])
        if item.get("comment"):
            item["comment"] = _scrub(item["comment"], mapping)
        out["revisions"].append(item)
    return out


def check(path: Path, originals: set[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Audit the output.

    Returns ``(leaked, title_collisions)``.

    ``leaked`` is any original name still present in a field we claim to scrub:
    decliners, revision users, template raws, edit summaries. It must be empty.

    ``title_collisions`` is separate and is not a bug. Draft titles are kept for
    reproducibility, and some accounts share a name with the draft they wrote --
    a contributor writing an article about themselves is the commonest reason an
    AfC draft exists. For those, the label hides nothing, and saying so is
    better than a clean check that implies otherwise.
    """
    leaked: set[str] = set()
    collisions: list[tuple[str, str]] = []
    unlabelled: set[str] = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        scrubbed = json.dumps(
            {"declines": record.get("declines", []), "revisions": record.get("revisions", [])},
            ensure_ascii=False,
        )
        title = record.get("title", "")
        for name in originals:
            if name in scrubbed:
                leaked.add(name)
            if len(name) >= 4 and name in title:
                collisions.append((name, title))

        # Independent of `originals`. Checking only names we collected is
        # circular: a name the map never learned is absent from `originals`
        # too, so it passes. Here we go the other way and read every account
        # position out of the shipped text, failing anything that is not a
        # label. This is what catches a collection gap.
        for decline in record.get("declines", []):
            unlabelled |= _not_a_label(decline.get("raw") or "")
            unlabelled |= _not_a_label(decline.get("decliner") or "", bare=True)
        for revision in record.get("revisions", []):
            unlabelled |= _not_a_label(revision.get("comment") or "")
            unlabelled |= _not_a_label(revision.get("user") or "", bare=True)

    return sorted(leaked), sorted(set(collisions)), sorted(unlabelled)


def _not_a_label(text: str, *, bare: bool = False) -> set[str]:
    """Account positions in `text` whose value is not one of our labels."""
    if not text:
        return set()
    found = {text.strip()} if bare else names_in_text(text)
    return {n for n in found if n and not LABEL.fullmatch(n)}


def _report_collisions(collisions: list[tuple[str, str]]) -> None:
    """Say plainly where a preserved title still names an account."""
    if not collisions:
        return
    print(f"\nnote   {len(collisions)} draft title(s) contain an account name. Titles are")
    print("       kept for reproducibility, so the label does not hide the account")
    print("       in these cases. Most are contributors writing about themselves.")
    for name, title in collisions[:5]:
        print(f"         {title}  (account {name!r})")
    if len(collisions) > 5:
        print(f"         ... and {len(collisions) - 5} more")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--salt-file", type=Path, default=DEFAULT_SALT)
    parser.add_argument("--check", type=Path, default=None,
                        help="verify an already-written file instead of writing one")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"{args.input} not found -- run `python -m first_reader.wiki.collect` first")
        return 1

    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]

    # AfC accepts submissions from a user sandbox as well as Draft space, and
    # those titles name the account outright. Titles are otherwise preserved
    # for reproducibility, so the only way to keep the promise here is to drop
    # the record. Re-collect with `python -m first_reader.wiki.collect` to get
    # them back.
    dropped = [r for r in records if not r.get("title", "").startswith("Draft:")]
    records = [r for r in records if r.get("title", "").startswith("Draft:")]
    if dropped:
        print(f"drop   {len(dropped)} record(s) whose title names an account:")
        for record in dropped[:8]:
            print(f"         {record.get('title')}")
    salt = load_or_make_salt(args.salt_file)
    mapping = build_label_map(records, salt)

    if args.check:
        leaked, collisions, unlabelled = check(args.check, set(mapping))
        if leaked:
            print(f"{len(leaked)} original name(s) still present in {args.check}:")
            for name in leaked[:20]:
                print(f"  {name}")
        if unlabelled:
            print(f"{len(unlabelled)} account position(s) not carrying a label:")
            for name in unlabelled[:20]:
                print(f"  {name}")
            return 1
        print(f"{args.check}: clean, none of {len(mapping)} original names appear "
              f"in a scrubbed field")
        _report_collisions(collisions)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(anonymize_record(record, mapping), ensure_ascii=False) + "\n")

    roles = Counter(label.split()[0] for label in mapping.values())
    print(f"read   {args.input}  ({len(records)} drafts)")
    print(f"wrote  {args.output}")
    print(f"salt   {args.salt_file}  (not for distribution -- gitignored)")
    print(f"labels {len(mapping)} accounts: " + ", ".join(f"{n} {r}" for r, n in sorted(roles.items())))

    leaked, collisions, unlabelled = check(args.output, set(mapping))
    if leaked:
        print(f"\nFAILED: {len(leaked)} original name(s) survived: {leaked[:10]}")
        return 1
    if unlabelled:
        print(f"\nFAILED: {len(unlabelled)} account position(s) do not carry a label.")
        print("       These were never collected, so scrubbing could not reach them.")
        for name in unlabelled[:20]:
            print(f"         {name!r}")
        if len(unlabelled) > 20:
            print(f"         ... and {len(unlabelled) - 20} more")
        return 1
    print("check  no original account name appears in any scrubbed field")
    print("check  every account position in the output carries a label")
    _report_collisions(collisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
