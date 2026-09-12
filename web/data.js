/* First Reader - data layer.
 *
 * Everything that knows where cards come from lives in this file. To move off
 * fixtures, change SOURCE below to the API endpoint; the rest of the app only
 * ever sees the shape returned by loadCards().
 *
 * The derived values (fixableByRewrite, addedSources, substantiveRewrite,
 * wastedRewrite, declineCount) are NOT read from the fixture. They are
 * recomputed here with the same rules as first_reader/models.py, so the
 * fixture cannot claim a verdict the model would not reach.
 */

const SOURCE = { kind: 'fixture', url: 'fixtures/cards.json' };

/* Port of REASON_FAMILY in models.py. Codes absent here resolve to UNKNOWN,
 * never to a guess. */
const REASON_FAMILY = {
  nn: 'source_existence', v: 'source_existence', bio: 'source_existence',
  corp: 'source_existence', event: 'source_existence', music: 'source_existence',
  athlete: 'source_existence', film: 'source_existence', web: 'source_existence',
  ai: 'writing', adv: 'writing', npov: 'writing', essay: 'writing',
  resume: 'writing', context: 'writing', lang: 'writing', ilc: 'writing',
  exists: 'structural', dup: 'structural', blank: 'structural',
  test: 'structural', redirect: 'structural',
};

const familyOf = (code) => REASON_FAMILY[code] || 'unknown';

/* Blocker.fixable_by_rewrite / Decline.fixable_by_rewrite. null on UNKNOWN. */
function fixableByRewrite(family) {
  if (family === 'unknown') return null;
  return family === 'writing';
}

/* AttemptSignal.substantive_rewrite */
const substantiveRewrite = (a) => a.similarity < 0.7 && Math.abs(a.bytes_changed) > 500;
/* AttemptSignal.added_sources */
const addedSources = (a) => Math.max(0, a.refs_after - a.refs_before);

/* What each decline code means, in a reviewer's words rather than as a code.
 * `line` is the sentence that becomes the headline of the card. */
const CODE_COPY = {
  nn:     { label: 'Notability',        line: 'The draft does not show that independent people have written about this subject in depth.' },
  v:      { label: 'Verifiability',     line: 'What the draft says cannot be checked against the sources it cites.' },
  bio:    { label: 'Notability, person', line: 'The draft does not show that this person has been written about in depth by people independent of her.' },
  corp:   { label: 'Notability, organisation', line: 'The draft does not show that independent outlets have covered this organisation in depth.' },
  event:  { label: 'Notability, event', line: 'The draft does not show sustained independent coverage of this event.' },
  music:  { label: 'Notability, music', line: 'The draft does not show that independent coverage of this artist or release exists.' },
  athlete:{ label: 'Notability, athlete', line: 'The draft does not show independent coverage of this athlete.' },
  film:   { label: 'Notability, film',  line: 'The draft does not show independent coverage of this film.' },
  web:    { label: 'Notability, website', line: 'The draft does not show independent coverage of this website.' },
  ai:     { label: 'Machine-written prose', line: 'The writing reads as machine-generated and has not been checked against the sources.' },
  adv:    { label: 'Promotional tone',  line: 'The draft reads as promotion for its subject rather than as a description of it.' },
  npov:   { label: 'Not neutral',       line: 'The draft takes a side instead of reporting what sources say.' },
  essay:  { label: 'Written as an essay', line: 'The draft argues a position rather than summarising published material.' },
  resume: { label: 'Written as a CV',   line: 'The draft is laid out as a résumé rather than as an article.' },
  context:{ label: 'Missing context',   line: 'The draft does not say enough for a reader to tell what the subject is.' },
  lang:   { label: 'Language',          line: 'The draft needs a pass in English before it can be reviewed.' },
  ilc:    { label: 'Inline citations',  line: 'Long stretches of the draft carry no citation for what they claim.' },
  exists: { label: 'Already covered',   line: 'A live article already covers this subject.' },
  dup:    { label: 'Duplicate draft',   line: 'Another draft already covers this subject.' },
  blank:  { label: 'Blank submission',  line: 'The submission has no content to review.' },
  test:   { label: 'Test edit',         line: 'The submission looks like a test rather than an attempted article.' },
  redirect:{ label: 'Should be a redirect', line: 'This belongs as a redirect rather than as its own article.' },
};

const codeCopy = (code) => CODE_COPY[code] || {
  label: code, line: 'Declined with a code First Reader has not been taught.',
};

/* The message that carries the whole product: can rewriting clear this? */
const FAMILY_COPY = {
  source_existence: {
    verdict: 'Rewriting will not clear this.',
    body: 'This block is about what has been published in the world, not about the prose. The contributor can rewrite the draft any number of times and the answer will not move.',
    need: 'What moves it: independent, published sources that already exist — or the honest answer that they do not.',
    short: 'not fixable by rewriting',
  },
  writing: {
    verdict: 'Rewriting can clear this.',
    body: 'This block is about the text itself. Editing the draft is the right response, and the contributor is not wasting their time doing it.',
    need: 'What moves it: an edit to the flagged prose, made against the sources already cited.',
    short: 'fixable by rewriting',
  },
  structural: {
    verdict: 'Rewriting is the wrong tool.',
    body: 'This block is not about the quality of the draft at all. Neither better prose nor better sources resolve it.',
    need: 'What moves it: a different action — a merge, a redirect, or a different page.',
    short: 'needs a different action',
  },
  unknown: {
    verdict: 'Not decided: we do not know whether rewriting clears this.',
    body: 'This decline code is not in First Reader’s map, and First Reader does not guess at codes it has not been taught. Nothing is being carried forward about it.',
    need: 'What happens next is your call, made from the draft itself.',
    short: 'not decided',
  },
};

/* Attach derived values without mutating meaning. */
function decorateAttempt(a) {
  if (!a) return null;
  return Object.assign({}, a, {
    addedSources: addedSources(a),
    substantiveRewrite: substantiveRewrite(a),
    rewrittenPct: Math.round((1 - a.similarity) * 100),
  });
}

function decorateBlocker(b) {
  const family = b.family || familyOf(b.code);
  return Object.assign({}, b, {
    family,
    fixableByRewrite: fixableByRewrite(family),
    attempt: decorateAttempt(b.attempt),
    copy: codeCopy(b.code),
    familyCopy: FAMILY_COPY[family],
  });
}

function decorateCard(raw, index) {
  const blockers = (raw.blockers || []).map(decorateBlocker);
  const top = blockers.length ? blockers[0] : null;
  const declines = (raw.draft.declines || []).map((d) => Object.assign({}, d, {
    family: familyOf(d.code),
    copy: codeCopy(d.code),
  }));
  /* ReviewCard.wasted_rewrite: top blocker is provably not fixable by rewriting,
   * and the contributor rewrote substantively anyway. `null` (unknown) never
   * counts as wasted. */
  const wastedRewrite = !!(top && top.attempt && top.fixableByRewrite === false
    && top.attempt.substantiveRewrite);

  return {
    index,
    id: raw.draft.pageid,
    draft: Object.assign({}, raw.draft, { declines, declineCount: declines.length }),
    blockers,
    topBlocker: top,
    otherBlockers: blockers.slice(1),
    draftComment: raw.draft_comment || '',
    needsHuman: raw.needs_human,
    escalationReason: raw.escalation_reason,
    wastedRewrite,
  };
}

async function loadCards() {
  let payload = null;
  try {
    const res = await fetch(SOURCE.url, { cache: 'no-store' });
    if (res.ok) payload = await res.json();
  } catch (err) {
    /* Opened straight off the filesystem: fetch of a local file is blocked by
     * the browser. fixtures/cards.js carries the same payload as a script tag
     * so the page still works on file://. */
  }
  if (!payload && window.__FIRST_READER_FIXTURE__) payload = window.__FIRST_READER_FIXTURE__;
  if (!payload) throw new Error('No card source reachable.');
  return {
    reviewer: payload.reviewer || 'you',
    generatedAt: payload.generated_at || new Date().toISOString(),
    cards: payload.cards.map(decorateCard),
  };
}

window.FirstReaderData = { loadCards, FAMILY_COPY, codeCopy, familyOf };
