# Where the data on screen comes from

Wikipedia AfC reviewers are volunteers under their real names, and their review
history is public. That makes it easy to build a convincing demo out of real
handles, and it is the wrong thing to do: a person appears in a repository and a
demo video attached to a review they did not make, on a draft they never saw.

So, plainly:

**Every account name anywhere in this repository's fixtures is a neutral label.**
`Reviewer A`, `Reviewer B`, `Contributor`, `AfC bot`. No real handle appears in
`web/fixtures/`, in `src/first_reader/agent/_stubs.py`, or on the reviewer screen.

## The one real card

`web/fixtures/cards.json`, first card: **Draft:0→1 Doctrine for AI**. It is a
real draft with a public history, and these fields are the real ones:

| field | value |
|---|---|
| pageid | 83560144 |
| decline 1 | `ai` + `nn`, submitted 20260627032430, declined 20260627045213 |
| decline 2 | `v` + `ai`, submitted 20260629172235, declined 20260629205711 |
| decline 3 | `nn`, submitted 20260708144743, declined 20260708144901 |
| revisions | real ids, timestamps, sizes and edit summaries (11 of them) |

The contributor's own words shown on the card — *"made changes re wrote
organically"*, *"have re- written completely"* — are their real edit summaries.
The decliners are labelled Reviewer A, B and C in the order they appear.

Two things on that card are **not** measured:

- **Diff-derived numbers.** `refs_before`, `refs_after`, `new_domains` and
  `similarity` are estimates. Nothing in this repository computed them from the
  wikitext. `bytes_changed` is real, because revision sizes are.
- **Queue state.** `is_pending` is true because the demo shows a pending queue.
  The real draft is not currently awaiting review.

## The other eight cards

Invented — titles, histories, timestamps, everything. Any resemblance to a real
draft is coincidence. Same for all five fixtures in
`src/first_reader/agent/_stubs.py`, which exist to give the escalation judgment
one draft per evidence state.

## The collected dataset

`data/dataset.jsonl` is 540 real declined drafts collected from the public API.
It is **not distributed**. What ships is `data/dataset.anonymized.jsonl` — 534
drafts, in which every account name is a stable label (`Reviewer 132`,
`Contributor 918`, `Bot 004`) produced by `scripts/anonymize.py`.

The six missing drafts are the ones whose *title* was `User:<account>/sandbox`.
A label cannot hide an account the title itself names, so those records are
dropped rather than shipped half-scrubbed. `scripts/anonymize.py --check` prints
each one.

Every edit in that file is public, and none of it is secret. What changes is
aggregation: collecting 540 declined drafts into one file turns information
scattered across thousands of page histories into a searchable index of who
gets declined for what, and who declines them. That index did not exist until
we built it, and redistributing it is a different act from the individual edits
being public.

Nothing analytic is lost. One person always gets one label, across every draft,
so repeat-reviewer analysis works unchanged. Titles, page ids, timestamps,
revision ids, sizes, edit summaries and decline codes are all preserved, so the
file is still fully reproducible against the API.

Every figure this project publishes is computed from the **shipped** file, so
what a reader runs is what a reader reads. The same scripts against the
540-draft raw file move small counts by the six dropped drafts — next-decline
overlap 36.2% → 36.1%, carried-forward 58.7% → 58.6% — and leave the headline
claims unchanged: 67.8% standing on a reason rewriting cannot fix, 19.7%
revived.

Two honest limits:

- **This is de-identification, not anonymity.** The salt is random per run and
  gitignored, so labels do not resolve without it. But revision ids are
  immutable and public, so anyone re-collecting from the API can rebuild the
  mapping. The goal is to stop the distributed file from *being* the index, not
  to make re-identification impossible — against public data with immutable ids,
  nothing can.
- **9 titles still name an account**, because titles are kept for
  reproducibility and some contributors write about themselves — `Draft:<their
  own name>`. Dropping those would remove the drafts this project is most about.
  `scripts/anonymize.py --check` lists every one on each run rather than letting
  a clean check imply otherwise.

To rebuild the raw file yourself: `python -m first_reader.wiki.collect`.

## Where this is stated

- `web/fixtures/cards.json` — a `provenance` block, carried into `cards.js`.
- The reviewer screen — a footnote under the card, permanent and not dismissible.
- `src/first_reader/agent/_stubs.py` — module docstring.
- `scripts/anonymize.py` — the module docstring argues the case, and every run
  prints what it could not hide.
- Here.

## If you add a fixture

Use a label, never a handle. If you take anything from the public record, add it
to the table above and say which of its fields are real.
