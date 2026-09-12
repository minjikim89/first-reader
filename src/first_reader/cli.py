"""Command line entry point.

    first-reader review "Draft:0→1 Doctrine for AI"   # one draft, end to end
    first-reader demo                                 # every fixture, no network
    first-reader rules                                # what it has learned so far

The same three things a reviewer gets on the screen come out here: what is
standing in the way, whether rewriting can clear it, and the note they would
post. Nothing is sent anywhere -- printing a prepared comment is the whole of
what this command does with it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .agent import PolicyStore, demo, run_review
from .agent import tools as t
from .agent._model import resolve_model
from .agent.policy import DEFAULT_POLICY_PATH
from .models import Blocker, ReviewCard

RULE = "─" * 72


def _fixable(blocker: Blocker) -> str:
    if blocker.fixable_by_rewrite is None:
        return "unknown — this code is not in our map"
    if blocker.fixable_by_rewrite:
        return "yes — this reason is about how the draft is written"
    return "NO — this depends on whether sources exist, not on the prose"


def _render(card: ReviewCard, judgment: dict[str, Any] | None) -> str:
    out: list[str] = [RULE, card.draft.title, RULE]

    if not card.blockers:
        out.append("\nNothing standing. No prior decline is still open.")
        return "\n".join(out)

    top = card.blockers[0]
    out.append(f"\nSTILL BLOCKING   {top.code}   (given {top.repeat_count}x)")
    out.append(f"Can rewriting clear it?   {_fixable(top)}")

    if card.wasted_rewrite:
        out.append(
            "\n!! This contributor substantively rewrote the draft against a reason\n"
            "   rewriting cannot fix. Saying so is the most useful thing this\n"
            "   review can carry."
        )

    attempt = top.attempt
    if attempt is None:
        out.append("\nWhat changed since the last decline: not measured.")
    else:
        out.append("\nWHAT THE CONTRIBUTOR DID SINCE   (measured from the diff, not judged)")
        out.append(f"  references      {attempt.refs_before} -> {attempt.refs_after}")
        out.append(f"  text rewritten  {round((1 - attempt.similarity) * 100)}%")
        out.append(f"  bytes           {attempt.bytes_changed:+,}")
        out.append(
            f"  touched the part the reason pointed at: "
            f"{'yes' if attempt.touched_target_area else 'no'}"
        )
        out.append(
            "  Whether those sources are independent, reliable or substantial is\n"
            "  the review itself, and First Reader does not decide it."
        )

    if len(card.blockers) > 1:
        out.append("\nALSO CARRIED FORWARD")
        for b in card.blockers[1:]:
            carried = f", raised by {len(b.carried_from)} reviewer(s)" if b.carried_from else ""
            out.append(f"  {b.code:<10} {b.family.value}  given {b.repeat_count}x{carried}")

    if judgment:
        verdict = "sending this to a reviewer" if judgment.get("escalate") else "handled alone"
        out.append(f"\nESCALATION   {verdict}")
        out.append(f"  state           {judgment.get('state')}")
        out.append(
            f"  advantage {judgment.get('advantage')} x cost {judgment.get('cost')} "
            f"x (1 - recoverability {judgment.get('recoverability')}) "
            f"= {judgment.get('score')}"
        )
        out.append(f"  threshold       {judgment.get('threshold')}")

    out.append("\nPREPARED COMMENT   (edit it — it goes out in your name, not ours)")
    out.append(RULE)
    out.append(card.draft_comment)
    out.append(RULE)
    return "\n".join(out)


def cmd_review(args: argparse.Namespace) -> int:
    model = resolve_model(args.model) if args.model else None
    outcome = run_review(args.title, model=model, queue_dir=args.queue_dir)

    if args.json:
        from .agent._serde import card_to_dict

        print(json.dumps({
            "title": outcome.title,
            "card": card_to_dict(outcome.card, outcome.interpretation) if outcome.card else None,
            "judgment": outcome.judgment,
            "queued_at": outcome.queued_at,
            "denied_calls": outcome.denied_calls,
            "error": outcome.error,
        }, indent=2, ensure_ascii=False))
        return 0 if outcome.ok else 1

    if outcome.error:
        print(f"{outcome.title}: {outcome.error}", file=sys.stderr)
        return 1
    if outcome.card is None:
        print(f"{outcome.title}: no card was produced ({outcome.stop_reason})", file=sys.stderr)
        return 1

    print(_render(outcome.card, outcome.judgment))

    for denial in outcome.denied_calls:
        print(f"\nrefused {denial['tool']}: {denial['reason']}")
    if outcome.queued_at:
        print(f"\nfiled for a reviewer at {outcome.queued_at}")
    if outcome.backstopped:
        print("(filed by the backstop: the judgment said escalate and the model did not file it)")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    print(demo(queue_dir=args.queue_dir))
    return 0


def cmd_rules(args: argparse.Namespace) -> int:
    store = PolicyStore(args.policy, autosave=False)
    if not store.rules:
        print("No rules yet. Cold start escalates everything, and autonomy is")
        print("earned from precedent: five agreements to relax, one reversal to tighten.")
        return 0
    for rule in store.rules:
        print(rule.render())
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="first-reader",
        description="Carry Wikipedia AfC decline context forward. It does not review.",
    )
    parser.add_argument("--queue-dir", type=Path, default=t.DEFAULT_QUEUE_DIR,
                        help=f"where escalated cards are filed (default: {t.DEFAULT_QUEUE_DIR})")
    sub = parser.add_subparsers(dest="command", required=True)

    p_review = sub.add_parser("review", help="run one draft end to end")
    p_review.add_argument("title", help='e.g. "Draft:0→1 Doctrine for AI"')
    p_review.add_argument("--model", default=None,
                          help="auto, openrouter, openai, anthropic, bedrock, ollama, offline")
    p_review.add_argument("--json", action="store_true", help="machine-readable output")
    p_review.set_defaults(func=cmd_review)

    p_demo = sub.add_parser("demo", help="run every fixture draft, no network")
    p_demo.set_defaults(func=cmd_demo)

    p_rules = sub.add_parser("rules", help="show the rules learned so far")
    p_rules.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    p_rules.set_defaults(func=cmd_rules)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
