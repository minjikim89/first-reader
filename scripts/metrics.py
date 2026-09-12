#!/usr/bin/env python3
"""Recompute every headline number in the README from the shipped dataset.

Run this to check the claims rather than taking them on trust:

    python scripts/metrics.py

Each metric prints the exact definition it uses. Three different readings of
"repeated an unfixable reason" are reported side by side because they differ
by ten percentage points, and quoting one without saying which is how a claim
becomes indefensible.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from first_reader.analysis import build_blockers  # noqa: E402
from first_reader.models import REASON_FAMILY, ReasonFamily  # noqa: E402
from first_reader.wiki.collect import load_dataset  # noqa: E402


def main() -> int:
    drafts = list(load_dataset())
    n = len(drafts)
    if not n:
        print("dataset is empty -- run `python -m first_reader.wiki.collect` first")
        return 1

    declines = sum(d.decline_count for d in drafts)
    print(f"dataset: {n} drafts, {declines} declines, "
          f"{sum(len(d.revisions) for d in drafts)} revisions")
    print("  (every draft in this set has been declined at least twice)\n")

    # -- family distribution ------------------------------------------------
    fam: Counter[str] = Counter()
    for d in drafts:
        for dec in d.declines:
            fam[dec.family.value] += 1
            if dec.secondary_code:
                secondary = REASON_FAMILY.get(dec.secondary_code, ReasonFamily.UNKNOWN)
                fam[secondary.value] += 1
    total_mentions = sum(fam.values())

    print(f"reason families, counting primary and secondary codes "
          f"({total_mentions} mentions)")
    for name, count in fam.most_common():
        print(f"  {name:<18} {count:>5}  {count / total_mentions * 100:5.1f}%")

    # -- what is standing in the way ----------------------------------------
    top_unfixable = 0
    top_unfixable_repeated = 0
    any_unfixable_repeated = 0
    any_repeated = 0

    for d in drafts:
        blockers = build_blockers(d)
        if not blockers:
            continue
        top = blockers[0]
        if top.fixable_by_rewrite is False:
            top_unfixable += 1
            if top.repeat_count >= 2:
                top_unfixable_repeated += 1
        if any(b.fixable_by_rewrite is False and b.repeat_count >= 2 for b in blockers):
            any_unfixable_repeated += 1
        if any(b.repeat_count >= 2 for b in blockers):
            any_repeated += 1

    def pct(k: int) -> str:
        return f"{k:>4}/{n}  {k / n * 100:5.1f}%"

    print("\nwhat is standing in the way (one row per draft)")
    print(f"  top blocker cannot be fixed by rewriting      {pct(top_unfixable)}")
    print(f"    ...and that same reason was given 2+ times  {pct(top_unfixable_repeated)}")
    print("\nlooser readings, reported so the headline cannot be mistaken for them")
    print(f"  any unfixable reason given 2+ times           {pct(any_unfixable_repeated)}")
    print(f"  any reason at all given 2+ times              {pct(any_repeated)}")

    # -- classification coverage --------------------------------------------
    unknown_codes: Counter[str] = Counter()
    for d in drafts:
        for dec in d.declines:
            if dec.family is ReasonFamily.UNKNOWN:
                unknown_codes[dec.code or "(empty)"] += 1
    unknown_total = sum(unknown_codes.values())

    print(f"\nunclassified: {unknown_total}/{declines} declines "
          f"({unknown_total / declines * 100:.1f}%)")
    for code, count in unknown_codes.most_common():
        print(f"  {code:<14} {count:>4}")
    print("\n  `reason` is free text with no fixed meaning, and n/j/e are reject "
          "codes\n  that block resubmission outright, so neither can be assigned a "
          "family.\n  Everything else here is a code we looked at and declined to "
          "classify.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
