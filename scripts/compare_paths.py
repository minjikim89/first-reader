#!/usr/bin/env python3
"""Prove the language model changed no decisions.

First Reader's safety argument is that a model reads prose and writes prose and
decides nothing. That is a claim about behaviour, so it should be checkable.

    python scripts/dump_judgments.py                          # deterministic path
    FIRST_READER_MODEL=openrouter python scripts/dump_judgments.py \
        --write --out data/judgments-llm.json                 # prose layer on
    python scripts/compare_paths.py data/judgments.json data/judgments-llm.json

Every judgment field must match. If a run ever differs, the model reached
something it was not supposed to reach, and this script names which draft and
which field so the leak can be found.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Fields the model is allowed to change. Everything else must be identical.
PROSE_FIELDS = {"comment", "interpretation", "closing_text"}


def load(path: str) -> dict[str, dict]:
    data = json.loads(Path(path).read_text())
    return {row["title"]: row for row in data}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2

    left, right = load(argv[1]), load(argv[2])

    only_left = sorted(left.keys() - right.keys())
    only_right = sorted(right.keys() - left.keys())
    shared = sorted(left.keys() & right.keys())

    print(f"{argv[1]}: {len(left)} drafts")
    print(f"{argv[2]}: {len(right)} drafts")
    print(f"compared: {len(shared)}\n")

    for title in only_left:
        print(f"  only in {argv[1]}: {title}")
    for title in only_right:
        print(f"  only in {argv[2]}: {title}")

    mismatches: list[tuple[str, str, object, object]] = []
    prose_differs = 0

    for title in shared:
        a, b = left[title], right[title]
        for key in sorted(set(a) | set(b)):
            if key in PROSE_FIELDS:
                if a.get(key) != b.get(key):
                    prose_differs += 1
                continue
            if a.get(key) != b.get(key):
                mismatches.append((title, key, a.get(key), b.get(key)))

    print(f"prose fields that differ: {prose_differs}  (expected -- that is the model's job)")

    if mismatches:
        print(f"\nDECISION FIELDS THAT DIFFER: {len(mismatches)}  <-- the model reached something it should not have")
        for title, key, av, bv in mismatches[:40]:
            print(f"  {title}\n    {key}: {av!r}  !=  {bv!r}")
        if len(mismatches) > 40:
            print(f"  ... and {len(mismatches) - 40} more")
        return 1

    print("\ndecision fields that differ: 0")
    print("Every judgment, blocker ordering, family and escalation state is identical.")
    print("The model changed the words and nothing else.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
