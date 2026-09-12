#!/usr/bin/env python3
"""Dump one judgment record per draft, so two runs can be compared field by field.

    python scripts/dump_judgments.py                        # deterministic path
    FIRST_READER_MODEL=openrouter \
        python scripts/dump_judgments.py --write --out data/judgments-llm.json
    python scripts/compare_paths.py data/judgments.json data/judgments-llm.json

The safety argument of this project is that a language model reads prose and
writes prose and decides nothing. That is a claim about behaviour, so it is
checkable: run the same dataset down both paths and diff the records. Only
``comment`` and ``interpretation`` may differ. Anything else differing means the
model reached a decision it was not supposed to reach, and ``compare_paths.py``
names the draft and the field.

The pass mirrors ``scripts/backtest.py``'s ``build_pass`` -- same calls, same
order, same cold policy -- because a proof that compared a different pipeline
than the backtest reports on would prove nothing about the backtest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from first_reader.agent import tools as t  # noqa: E402
from first_reader.agent.escalation import judge, pattern_key  # noqa: E402
from first_reader.agent.interpret import resolve_interpreter  # noqa: E402
from first_reader.agent.policy import PolicyStore  # noqa: E402
from first_reader.analysis import build_blockers  # noqa: E402
from first_reader.models import Blocker, ReviewCard  # noqa: E402
from first_reader.wiki.collect import load_dataset  # noqa: E402


def blocker_record(blocker: Blocker) -> dict[str, Any]:
    """Everything about a blocker that the model must not be able to move."""
    attempt = blocker.attempt
    return {
        "code": blocker.code,
        "family": blocker.family.value,
        "repeat_count": blocker.repeat_count,
        "fixable_by_rewrite": blocker.fixable_by_rewrite,
        "first_seen": blocker.first_seen.isoformat(),
        "carried_from": list(blocker.carried_from),
        "attempt": None if attempt is None else {
            "refs_before": attempt.refs_before,
            "refs_after": attempt.refs_after,
            "added_sources": attempt.added_sources,
            "new_domains": list(attempt.new_domains),
            "bytes_changed": attempt.bytes_changed,
            "similarity": attempt.similarity,
            "touched_target_area": attempt.touched_target_area,
            "substantive_rewrite": attempt.substantive_rewrite,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=None,
                        help="default: data/dataset.jsonl, else the anonymised copy")
    parser.add_argument("--limit", type=int, default=0, help="only the first N drafts")
    parser.add_argument("--write", action="store_true",
                        help="read free text and write notes with the model (costs money)")
    parser.add_argument("--out", type=Path, default=Path("data/judgments.json"))
    args = parser.parse_args(argv)

    dataset_path = args.dataset or t.default_dataset()
    drafts = load_dataset(dataset_path)
    if args.limit:
        drafts = drafts[: args.limit]
    if not drafts:
        print(f"{dataset_path} is empty -- run `python -m first_reader.wiki.collect` first")
        return 1

    interpreter = resolve_interpreter() if args.write else None
    if args.write and not interpreter.available:
        print("--write asked for, but no model is configured; nothing would be proved")
        return 1

    # Cold, non-persisting, and walked in dataset order: precedent accumulates
    # identically on both paths, so escalation stays comparable.
    policy = PolicyStore(Path("data/backtest-policy.json"), autosave=False)

    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    started = time.time()

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
            interpretation = t.interpret_card(card, interpreter) if interpreter else None
            card.draft_comment = t.compose_comment(
                card, interpretation=interpretation, interpreter=interpreter
            )
        except Exception as exc:  # noqa: BLE001 - counted, never swallowed
            failures.append((history.title, f"{type(exc).__name__}: {exc}"))
            continue

        row: dict[str, Any] = {
            "title": history.title,
            "pageid": history.pageid,
            "decline_count": len(history.declines),
            "blockers": [blocker_record(b) for b in card.blockers],
            "wasted_rewrite": card.wasted_rewrite,
            # Prose. The model is allowed to change these and nothing else.
            "comment": card.draft_comment,
            "interpretation": interpretation.to_dict() if interpretation else None,
        }

        if card.blockers:
            key = pattern_key(card)
            verdict = judge(card, precedent_seen=policy.precedent_seen(key))
            policy.observe(key)
            row.update({
                "pattern_key": key,
                "escalate": verdict.escalate,
                "state": verdict.state.value,
                "advantage": verdict.advantage,
                "cost": verdict.cost,
                "recoverability": verdict.recoverability,
                "score": verdict.score,
                "interruption_cost": verdict.interruption_cost,
            })

        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True))

    path = "model" if interpreter else "deterministic"
    model_id = f" ({interpreter.model_id})" if interpreter else ""
    print(f"{args.out}: {len(rows)} judgments from {dataset_path}, {path} path{model_id}")
    print(f"elapsed {time.time() - started:.1f}s")
    if failures:
        print(f"build failures: {len(failures)}")
        for title, err in failures[:10]:
            print(f"  {title}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
