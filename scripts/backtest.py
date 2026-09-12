#!/usr/bin/env python3
"""Run the agent's pipeline over the whole collected dataset and report what breaks.

    python scripts/backtest.py                  # card build + escalation + overlap
    python scripts/backtest.py --agent 20       # also run the full agent on 20 drafts
    python scripts/backtest.py --attempts 30    # also measure attempts live on 30
    python scripts/backtest.py --json out.json  # machine-readable

Three numbers come out of this, and the first is the one that decides whether
any of the others mean anything.

**Card build failure rate.** How many of the collected drafts the pipeline
cannot turn into a review card at all. A demo is worth its failure rate.

**Escalation distribution.** Which evidence state each draft lands in, and what
share reach a reviewer. Run cold, with an empty policy, which is the honest
starting condition -- precedent accumulates as the pass proceeds, exactly as it
would in use.

**Next-decline overlap.** On drafts declined three or more times: show the
pipeline only the first N declines, take the blocker it puts on top, and check
it against the reason the next decline actually gave.

That last one is an overlap rate and nothing more. It does **not** say a
reviewer would have decided differently had they seen a different ordering.
That counterfactual is not observable in this data and is not claimed here. The
naive baseline -- just repeat the most recent primary code -- is reported
alongside, because an overlap rate without a baseline is not a result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from first_reader.agent import tools as t  # noqa: E402
from first_reader.agent.escalation import classify, judge, pattern_key  # noqa: E402
from first_reader.agent.interpret import free_text_reason, resolve_interpreter  # noqa: E402
from first_reader.agent.policy import PolicyStore  # noqa: E402
from first_reader.analysis import build_blockers, canonical_code  # noqa: E402
from first_reader.models import DraftHistory, ReviewCard  # noqa: E402
from first_reader.wiki.collect import load_dataset  # noqa: E402


# --- card build -------------------------------------------------------------


@dataclass
class BuildReport:
    total: int = 0
    built: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    states: Counter[str] = field(default_factory=Counter)
    escalated: int = 0
    empty_blockers: list[str] = field(default_factory=list)
    readings: int = 0
    calls: int = 0
    fell_back: int = 0

    @property
    def failure_rate(self) -> float:
        return len(self.failures) / self.total if self.total else 0.0


def build_pass(
    drafts: list[DraftHistory], policy: PolicyStore, interpreter: Any | None = None
) -> BuildReport:
    """Build a card for every draft and judge it, cold, in dataset order.

    With an ``interpreter``, free-text reasons are read and the note is written
    by the model -- the path a reviewer actually gets. Without one, both fall
    back and the pass is free. Both have to get through all 540.
    """
    report = BuildReport(total=len(drafts))

    for history in drafts:
        try:
            blockers = build_blockers(history, None)
            card = ReviewCard(
                draft=history,
                blockers=blockers,
                draft_comment="",
                needs_human=False,
                escalation_reason="",
            )
            interpretation = None
            if interpreter is not None:
                interpretation = t.interpret_card(card, interpreter)
                report.readings += len(interpretation.families)
                report.calls += 1 + sum(
                    1 for d in history.declines if free_text_reason(d) is not None
                )
            card.draft_comment = t.compose_comment(
                card, interpretation=interpretation, interpreter=interpreter
            )
            if interpreter is not None and "Carried forward for" in card.draft_comment:
                report.fell_back += 1
        except Exception as exc:  # noqa: BLE001 - the point is to count these
            report.failures.append((history.title, f"{type(exc).__name__}: {exc}"))
            continue

        report.built += 1
        if not card.blockers:
            report.empty_blockers.append(history.title)
            continue

        key = pattern_key(card)
        verdict = judge(card, precedent_seen=policy.precedent_seen(key))
        policy.observe(key)
        report.states[verdict.state.value] += 1
        if verdict.escalate:
            report.escalated += 1

    return report


# --- next-decline overlap ---------------------------------------------------


@dataclass
class OverlapReport:
    windows: int = 0
    top_hits_primary: int = 0
    top_hits_either: int = 0
    baseline_hits_primary: int = 0
    baseline_hits_either: int = 0
    carried_contains: int = 0
    latest_only_contains: int = 0
    revived: int = 0
    drafts: int = 0

    def rate(self, hits: int) -> float:
        return hits / self.windows if self.windows else 0.0


def _truncate(history: DraftHistory, keep: int) -> DraftHistory:
    """The draft as it looked after its first ``keep`` declines."""
    declines = sorted(history.declines, key=lambda d: (d.declined_at, d.submitted_at))[:keep]
    cutoff = declines[-1].declined_at
    return DraftHistory(
        title=history.title,
        pageid=history.pageid,
        declines=declines,
        revisions=[r for r in history.revisions if r.timestamp <= cutoff],
        is_pending=False,
    )


def overlap_pass(drafts: list[DraftHistory]) -> OverlapReport:
    """Predict decline N+1's reason from declines 1..N, and score the overlap."""
    report = OverlapReport()

    for history in drafts:
        declines = sorted(history.declines, key=lambda d: (d.declined_at, d.submitted_at))
        if len(declines) < 3:
            continue
        report.drafts += 1

        for keep in range(2, len(declines)):
            actual = declines[keep]
            actual_primary = canonical_code(actual.code)
            actual_either = {
                c for c in (canonical_code(actual.code), canonical_code(actual.secondary_code)) if c
            }
            if not actual_primary:
                continue

            blockers = build_blockers(_truncate(history, keep), None)
            if not blockers:
                continue

            report.windows += 1
            predicted = blockers[0].code
            baseline = canonical_code(declines[keep - 1].code)

            report.top_hits_primary += predicted == actual_primary
            report.top_hits_either += predicted in actual_either
            report.baseline_hits_primary += baseline == actual_primary
            report.baseline_hits_either += baseline in actual_either

            # What First Reader actually does differently: it carries every
            # reason ever raised, not only the ones on the most recent decline.
            carried = {b.code for b in blockers}
            latest_only = {
                c
                for c in (
                    canonical_code(declines[keep - 1].code),
                    canonical_code(declines[keep - 1].secondary_code),
                )
                if c
            }
            report.carried_contains += actual_primary in carried
            report.latest_only_contains += actual_primary in latest_only
            # The case the product is named for: a reason went quiet on the most
            # recent decline and came back as the next one's primary reason.
            report.revived += actual_primary in carried and actual_primary not in latest_only

    return report


# --- optional live passes ---------------------------------------------------


def attempt_pass(drafts: list[DraftHistory], sample: int) -> dict[str, Any]:
    """Measure attempts live on a sample and re-judge, to size the offline gap.

    The offline pass leaves every attempt unmeasured, which sends a third of the
    set to a reviewer under ``attempt_not_measured``. That share is an artifact
    of not looking, not a property of the drafts. This measures how it moves
    once we do look.
    """
    measured = 0
    empty_window = 0
    failed = 0
    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    escalated_before = 0
    escalated_after = 0
    started = time.time()

    for history in drafts[:sample]:
        cold = ReviewCard(
            draft=history,
            blockers=build_blockers(history, None),
            draft_comment="",
            needs_human=False,
            escalation_reason="",
        )
        if not cold.blockers:
            continue
        verdict = judge(cold, precedent_seen=True)
        before[verdict.state.value] += 1
        escalated_before += verdict.escalate

        try:
            attempts = t.measure_attempts(history)
        except Exception:  # noqa: BLE001
            failed += 1
            continue
        if attempts:
            measured += 1
        else:
            empty_window += 1

        warm = ReviewCard(
            draft=history,
            blockers=build_blockers(history, attempts),
            draft_comment="",
            needs_human=False,
            escalation_reason="",
        )
        verdict = judge(warm, precedent_seen=True)
        after[verdict.state.value] += 1
        escalated_after += verdict.escalate

    return {
        "sampled": min(sample, len(drafts)),
        "measured": measured,
        "no_window": empty_window,
        "failed": failed,
        "states_unmeasured": dict(before),
        "states_measured": dict(after),
        "escalated_unmeasured": escalated_before,
        "escalated_measured": escalated_after,
        "seconds": round(time.time() - started, 1),
    }


def agent_pass(
    titles: list[str], sample: int, policy: PolicyStore, queue_dir: Path
) -> dict[str, Any]:
    """Run the whole agent -- event loop, tools, gates -- on real drafts."""
    from first_reader.agent.core import run_review

    ok = 0
    escalated = 0
    errors: list[tuple[str, str]] = []
    started = time.time()

    for title in titles[:sample]:
        outcome = run_review(title, policy=policy, queue_dir=queue_dir)
        if outcome.ok:
            ok += 1
            escalated += outcome.escalated
        else:
            errors.append((title, outcome.error or "?"))

    return {
        "ran": min(sample, len(titles)),
        "ok": ok,
        "escalated": escalated,
        "errors": errors,
        "seconds": round(time.time() - started, 1),
    }


# --- entry point ------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=None,
                        help="default: data/dataset.jsonl, else the anonymised copy")
    parser.add_argument("--limit", type=int, default=0, help="only the first N drafts")
    parser.add_argument("--agent", type=int, default=0, help="run the full agent on N drafts")
    parser.add_argument("--attempts", type=int, default=0, help="measure attempts live on N drafts")
    parser.add_argument("--write", action="store_true",
                        help="read free text and write notes with the model (costs money)")
    parser.add_argument("--queue", type=Path, default=Path("data/queue"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    dataset_path = args.dataset or t.default_dataset()
    drafts = load_dataset(dataset_path)
    if args.limit:
        drafts = drafts[: args.limit]
    if not drafts:
        print(f"{dataset_path} is empty -- run `python -m first_reader.wiki.collect` first")
        return 1

    policy = PolicyStore(Path("data/backtest-policy.json"), autosave=False)
    interpreter = resolve_interpreter() if args.write else None
    if args.write and not interpreter.available:
        print("--write asked for, but no model is configured; falling back")
        interpreter = None
    started = time.time()
    build = build_pass(drafts, policy, interpreter)
    elapsed = round(time.time() - started, 1)
    overlap = overlap_pass(drafts)

    print(f"dataset: {build.total} drafts from {dataset_path}\n")

    print("CARD BUILD")
    print(f"  built            {build.built}/{build.total}  ({100 * (1 - build.failure_rate):.1f}%)")
    print(f"  failed           {len(build.failures)}  ({100 * build.failure_rate:.2f}%)")
    print(f"  no blockers      {len(build.empty_blockers)}  (built, but nothing standing)")
    for title, err in build.failures[:10]:
        print(f"    {title}: {err}")
    if len(build.failures) > 10:
        print(f"    ... and {len(build.failures) - 10} more")
    if interpreter is not None:
        print(f"  model            {interpreter.model_id}")
        print(f"  free-text read   {build.readings} readings")
        print(f"  llm calls        {build.calls}  "
              f"({build.calls / build.total:.2f} per draft)")
        print(f"  note fell back   {build.fell_back}  to the template")
        print(f"  elapsed          {elapsed}s")
    else:
        print(f"  path             template + no interpretation ({elapsed}s)")

    judged = sum(build.states.values())
    print("\nESCALATION, run cold over the whole set")
    print(f"  judged           {judged}")
    print(f"  to a reviewer    {build.escalated}  ({100 * build.escalated / judged:.1f}%)" if judged else "")
    for state, count in build.states.most_common():
        print(f"    {state:<24} {count:>4}  ({100 * count / judged:.1f}%)")

    print("\nNEXT-DECLINE OVERLAP  (drafts declined 3+ times)")
    print(f"  drafts           {overlap.drafts}")
    print(f"  windows          {overlap.windows}   (one per 'declines 1..N -> decline N+1')")
    if overlap.windows:
        print("\n  a) which single reason goes on top")
        print(f"     top blocker == next primary            "
              f"{overlap.top_hits_primary:>4}  ({100 * overlap.rate(overlap.top_hits_primary):.1f}%)")
        print(f"     top blocker in next primary/secondary  "
              f"{overlap.top_hits_either:>4}  ({100 * overlap.rate(overlap.top_hits_either):.1f}%)")
        print(f"     baseline: repeat last primary          "
              f"{overlap.baseline_hits_primary:>4}  ({100 * overlap.rate(overlap.baseline_hits_primary):.1f}%)")
        print(f"     baseline in next primary/secondary     "
              f"{overlap.baseline_hits_either:>4}  ({100 * overlap.rate(overlap.baseline_hits_either):.1f}%)")
        if overlap.top_hits_primary == overlap.baseline_hits_primary:
            print("     These are identical by construction, not by coincidence:")
            print("     build_blockers puts the most recent decline's primary reason")
            print("     first, so on this metric the predictor IS the baseline. The")
            print("     number says something about AfC (how often the next decline")
            print("     repeats the last reason), not about First Reader.")

        print("\n  b) whether the next reason was on the card at all")
        print("     (this is the difference First Reader actually makes: it carries")
        print("      every reason ever raised, not only the most recent decline's)")
        print(f"     carried forward set contains next primary "
              f"{overlap.carried_contains:>4}  ({100 * overlap.rate(overlap.carried_contains):.1f}%)")
        print(f"     most recent decline alone contains it     "
              f"{overlap.latest_only_contains:>4}  ({100 * overlap.rate(overlap.latest_only_contains):.1f}%)")
        print(f"     gap: reason had gone quiet and came back  "
              f"{overlap.revived:>4}  ({100 * overlap.rate(overlap.revived):.1f}%)")
    print("\n  All of the above is overlap, not effect. It does not say a reviewer")
    print("  would have decided differently given a different ordering; that")
    print("  counterfactual is not observable in this data and is not claimed.")

    payload: dict[str, Any] = {
        "dataset": str(dataset_path),
        "drafts": build.total,
        "card_build": {
            "built": build.built,
            "failed": len(build.failures),
            "failure_rate": build.failure_rate,
            "no_blockers": len(build.empty_blockers),
            "llm_calls": build.calls,
            "readings": build.readings,
            "fell_back_to_template": build.fell_back,
            "failures": build.failures,
        },
        "escalation": {
            "judged": judged,
            "escalated": build.escalated,
            "states": dict(build.states),
        },
        "overlap": {
            "drafts": overlap.drafts,
            "windows": overlap.windows,
            "top_primary": overlap.rate(overlap.top_hits_primary),
            "top_either": overlap.rate(overlap.top_hits_either),
            "baseline_primary": overlap.rate(overlap.baseline_hits_primary),
            "baseline_either": overlap.rate(overlap.baseline_hits_either),
            "carried_contains_next": overlap.rate(overlap.carried_contains),
            "latest_only_contains_next": overlap.rate(overlap.latest_only_contains),
            "revived": overlap.rate(overlap.revived),
        },
    }

    if args.attempts:
        print("\nATTEMPT MEASUREMENT, live API")
        result = attempt_pass(drafts, args.attempts)
        print(f"  sampled {result['sampled']}, measured {result['measured']}, "
              f"no window {result['no_window']}, failed {result['failed']} "
              f"({result['seconds']}s)")
        print(f"  escalated without measuring: {result['escalated_unmeasured']}"
              f"  -> after measuring: {result['escalated_measured']}")
        print(f"    unmeasured states: {result['states_unmeasured']}")
        print(f"    measured states:   {result['states_measured']}")
        payload["attempts"] = result

    if args.agent:
        print("\nFULL AGENT on real drafts")
        result = agent_pass([d.title for d in drafts], args.agent, policy, args.queue)
        print(f"  ran {result['ran']}, ok {result['ok']}, escalated {result['escalated']} "
              f"({result['seconds']}s)")
        for title, err in result["errors"][:5]:
            print(f"    {title}: {err}")
        payload["agent"] = result

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
