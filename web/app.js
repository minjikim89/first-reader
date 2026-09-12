/* First Reader - reviewer screen.
 * One draft at a time. Nothing here decides; it shows what was already decided. */

(function () {
  'use strict';

  const $ = (sel) => document.querySelector(sel);
  const esc = (s) => String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

  const state = {
    cards: [], reviewer: 'you', now: new Date(),
    i: 0, posted: {}, comments: {}, rules: [],
  };

  /* ------------------------------------------------------------ format */

  const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

  function fmtDate(iso) {
    const d = new Date(iso);
    return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
  }

  function ago(iso) {
    const days = Math.round((state.now - new Date(iso)) / 86400000);
    if (days <= 0) return 'today';
    if (days === 1) return 'yesterday';
    if (days < 31) return `${days} days ago`;
    const m = Math.round(days / 30.4);
    return m <= 1 ? 'a month ago' : `${m} months ago`;
  }

  function fmtBytes(n) {
    const sign = n > 0 ? '+' : n < 0 ? '-' : '';
    const a = Math.abs(n);
    return a >= 1024 ? `${sign}${(a / 1024).toFixed(1)} KB` : `${sign}${a} B`;
  }

  const ordinal = (n) => n + (['th','st','nd','rd'][(n % 100 - 20) % 10] || ['th','st','nd','rd'][n % 100] || 'th');

  /* ------------------------------------------------------------ header */

  function renderHead(c) {
    const d = c.draft;
    const title = d.title.startsWith('Draft:')
      ? `<span class="ns">Draft:</span>${esc(d.title.slice(6))}`
      : esc(d.title);
    const last = d.revisions.length ? d.revisions[d.revisions.length - 1] : null;
    const meta = [];
    meta.push(d.declineCount
      ? `<span>${d.declineCount} decline${d.declineCount > 1 ? 's' : ''}</span>`
      : `<span>never declined</span>`);
    if (last) meta.push(`<span>last edited by ${esc(last.user)}, ${ago(last.timestamp)}</span>`);
    if (c.topBlocker && c.topBlocker.attempt) {
      meta.push(`<span>${c.topBlocker.attempt.refs_after} references</span>`);
    }
    meta.push(`<span>pageid ${d.pageid}</span>`);

    const eyebrow = c.needsHuman
      ? `<strong>Surfaced because</strong><span>${esc(c.escalationReason)}</span>`
      : `<strong>Nothing outstanding</strong><span>${esc(c.escalationReason)}</span>`;

    return `<header class="card__head">
      <div class="eyebrow${c.needsHuman ? '' : ' eyebrow--quiet'}">${eyebrow}</div>
      <h1 class="draft-title">${title}</h1>
      <div class="meta">${meta.join('<span class="sep">·</span>')}</div>
    </header>`;
  }

  /* ----------------------------------------------------------- verdict */

  function renderVerdict(c) {
    const b = c.topBlocker;

    if (!b) {
      return `<section class="verdict verdict--none">
        <div class="verdict__tag"><span>Nothing standing</span></div>
        <h2 class="verdict__headline">No prior decision to carry forward.</h2>
        <div class="answer">
          <span class="answer__mark" aria-hidden="true">–</span>
          <div class="answer__body">
            <div class="answer__verdict">This is a first submission.</div>
            <p>There is no decline to repeat and no attempt to measure against one.
               First Reader has nothing to add to this review.</p>
          </div>
        </div>
      </section>`;
    }

    const f = b.familyCopy;
    const mark = { source_existence: '✕', writing: '✓', structural: '⇄', unknown: '?' }[b.family];
    const repeat = b.repeat_count > 1
      ? `<span class="verdict__repeat">${ordinal(b.repeat_count)} time this reason has been given</span>`
      : `<span class="verdict__repeat">raised ${ago(b.first_seen)}</span>`;

    return `<section class="verdict verdict--${b.family}">
      <div class="verdict__tag">
        <span>Still blocking</span>
        <span class="verdict__code">${esc(b.code)}</span>
        ${repeat}
      </div>
      <h2 class="verdict__headline">${esc(b.copy.line)}</h2>
      <div class="answer">
        <span class="answer__mark" aria-hidden="true">${mark}</span>
        <div class="answer__body">
          <div class="answer__verdict">${esc(f.verdict)}</div>
          <p>${esc(f.body)}</p>
          <div class="answer__need">${esc(f.need)}</div>
        </div>
      </div>
    </section>`;
  }

  /* ------------------------------------------------------------- waste */

  function renderWaste(c) {
    if (!c.wastedRewrite) return '';
    const b = c.topBlocker, a = b.attempt;
    return `<div class="waste">
      <span class="waste__mark" aria-hidden="true">◆</span>
      <div>
        <div class="waste__title">This contributor rewrote the draft against a reason rewriting cannot fix.</div>
        <p><b>${a.rewrittenPct}% of the text is new</b> and the draft grew by ${fmtBytes(a.bytes_changed)} since
           <b>${esc(b.code)}</b> was raised on ${fmtDate(b.first_seen)}. That work cannot move this decline.
           Saying so is the single most useful thing this review can do.</p>
      </div>
    </div>`;
  }

  /* -------------------------------------------------- attempt (observed) */

  function renderAttempt(c) {
    const b = c.topBlocker;
    if (!b) return '';

    const head = `<div class="sec__head">
      <h2 class="sec__title">What the contributor did since</h2>
      <span class="sec__aside">measured from the diff — not judged</span>
    </div>`;

    if (!b.attempt) {
      return `<section class="sec">${head}
        <p class="nothing">No attempt signal. Nothing has been submitted against this reason yet,
        so there is nothing to measure.</p></section>`;
    }

    const a = b.attempt;
    const flat = a.addedSources === 0;
    const stats = [
      `<div class="stat${flat ? ' stat--flat' : ''}">
         <div class="stat__k">Sources cited</div>
         <div class="stat__v">${a.refs_before} → ${a.refs_after}
           <small>${flat ? 'none added' : `+${a.addedSources} added`}</small></div>
       </div>`,
      `<div class="stat${a.rewrittenPct >= 30 ? '' : ' stat--flat'}">
         <div class="stat__k">Text rewritten</div>
         <div class="stat__v">${a.rewrittenPct}%</div>
         <div class="bar"><i style="width:${Math.max(2, a.rewrittenPct)}%"></i></div>
       </div>`,
      `<div class="stat">
         <div class="stat__k">Size change</div>
         <div class="stat__v">${fmtBytes(a.bytes_changed)}</div>
       </div>`,
      `<div class="stat${a.touched_target_area ? '' : ' stat--warn'}">
         <div class="stat__k">Flagged area</div>
         <div class="stat__v">${a.touched_target_area ? 'Touched' : 'Not touched'}</div>
       </div>`,
    ].join('');

    const obs = [];
    if (a.new_domains.length) {
      obs.push(`<div>
        <div class="obs__k">Domains cited that were not there before</div>
        <div class="domains">${a.new_domains.map((d) => `<span class="domain">${esc(d)}</span>`).join('')}</div>
      </div>`);
    }
    if (a.editor_comments.length) {
      obs.push(`<div>
        <div class="obs__k">In the contributor's own words</div>
        <ul class="summaries">${a.editor_comments.map((s) => `<li>${esc(s)}</li>`).join('')}</ul>
      </div>`);
    }

    let caveat = '';
    if (b.family === 'source_existence' && a.addedSources > 0) {
      caveat = `<p class="nothing">First Reader does not check whether these sources are independent,
        reliable or substantial. Deciding that <em>is</em> the review, and it is yours to make.</p>`;
    } else if (a.addedSources === 0 && a.rewrittenPct < 10) {
      caveat = `<p class="nothing">Nothing measurable changed: no sources added, ${fmtBytes(a.bytes_changed)} of text,
        and the flagged area was left alone.</p>`;
    }

    return `<section class="sec">${head}
      <div class="stats">${stats}</div>
      ${obs.length ? `<div class="obs">${obs.join('')}</div>` : ''}
      ${caveat}
    </section>`;
  }

  /* ----------------------------------------------------------- history */

  function renderHistory(c) {
    const ds = c.draft.declines;
    if (!ds.length) return '';
    const seen = {};
    const items = ds.map((d) => {
      seen[d.code] = (seen[d.code] || 0) + 1;
      const again = seen[d.code] > 1
        ? `<div class="tl__2nd">Same code as before — ${ordinal(seen[d.code])} time.</div>` : '';
      const sec = d.secondary_code
        ? `<span class="tl__also">also</span><span class="code code--${window.FirstReaderData.familyOf(d.secondary_code)}">${esc(d.secondary_code)}</span>`
        : '';
      return `<li class="tl tl--${d.family}">
        <div class="tl__top">
          <span class="tl__date">${fmtDate(d.declined_at)}</span>
          <span class="code code--${d.family}">${esc(d.code)}</span>${sec}
          <span class="tl__by">declined by <b>${esc(d.decliner)}</b></span>
        </div>
        <div class="tl__line">${esc(d.copy.line)}</div>
        ${again}
      </li>`;
    }).join('');

    return `<section class="sec">
      <div class="sec__head">
        <h2 class="sec__title">Decline history</h2>
        <span class="sec__aside">${ds.length} decision${ds.length > 1 ? 's' : ''} by
          ${new Set(ds.map((d) => d.decliner)).size} reviewer${new Set(ds.map((d) => d.decliner)).size > 1 ? 's' : ''}</span>
      </div>
      <ol class="timeline">${items}</ol>
    </section>`;
  }

  /* -------------------------------------------------- other blockers */

  function renderAlso(c) {
    if (!c.otherBlockers.length) return '';
    const rows = c.otherBlockers.map((b) => {
      const cls = b.fixableByRewrite === true ? 'fix--true'
        : b.family === 'structural' ? 'fix--struct'
        : b.fixableByRewrite === false ? 'fix--false' : 'fix--null';
      const carried = b.carried_from.length
        ? `raised by ${b.carried_from.map(esc).join(', ')}` : '';
      const rep = b.repeat_count > 1 ? `${b.repeat_count}×` : 'once';
      return `<div class="also__row">
        <span class="code code--${b.family}">${esc(b.code)}</span>
        <span class="also__line">${esc(b.copy.line)}</span>
        <span class="also__meta"><b class="${cls}">${esc(b.familyCopy.short)}</b> · given ${rep}${carried ? ' · ' + carried : ''}</span>
      </div>`;
    }).join('');

    return `<section class="sec">
      <div class="sec__head">
        <h2 class="sec__title">Also still standing</h2>
        <span class="sec__aside">below the block above, not instead of it</span>
      </div>
      <div class="also">${rows}</div>
    </section>`;
  }

  /* ------------------------------------------------------------ action */

  function renderAction(c) {
    const posted = state.posted[c.id];
    const talk = c.draft.title.replace(/^Draft:/, 'Draft talk:');

    if (!c.draftComment) {
      return `<section class="sec action">
        <div class="sec__head">
          <h2 class="sec__title">Reviewer action</h2>
          <span class="sec__aside">nothing is posted without you</span>
        </div>
        <p class="nothing">No prepared comment. There is no earlier decision to carry forward,
          so First Reader has nothing to put in your name.</p>
        <div class="actions">
          <button class="btn btn--ghost" data-act="skip" type="button">Next draft</button>
          <span class="spacer"></span>
          <button class="btn btn--ghost" data-act="afch" type="button">Open in AFCH →</button>
        </div>
      </section>`;
    }

    const body = state.comments[c.id] != null ? state.comments[c.id] : c.draftComment;

    const controls = posted
      ? `<div class="actions">
           <span class="done-stamp">✓ Posted to ${esc(talk)} as ${esc(state.reviewer)}</span>
           <span class="spacer"></span>
           <button class="btn btn--ghost" data-act="afch" type="button">Open in AFCH →</button>
         </div>`
      : `<div class="actions">
           <button class="btn btn--primary" data-act="post" type="button">Post as ${esc(state.reviewer)}</button>
           <button class="btn btn--ghost" data-act="wrong" type="button">Wrong thing on top</button>
           <button class="btn btn--ghost" data-act="skip" type="button">Skip</button>
           <span class="spacer"></span>
           <button class="btn btn--ghost" data-act="afch" type="button">Open in AFCH →</button>
         </div>`;

    return `<section class="sec action">
      <div class="sec__head">
        <h2 class="sec__title">Reviewer action</h2>
        <span class="sec__aside">edit it — it goes out in your name</span>
      </div>
      <div class="compose">
        <div class="compose__head">
          <span>${esc(talk)}</span>
          <span>signed <b>${esc(state.reviewer)}</b></span>
        </div>
        <textarea id="comment" spellcheck="false" aria-label="Prepared comment">${esc(body)}</textarea>
      </div>
      ${controls}
      <p class="handoff">First Reader does not accept, decline or tag anything. Those stay in
        <b>AFCH</b>, under your account. This button posts the comment above and nothing else.</p>
    </section>`;
  }

  /* ------------------------------------------------------------ render */

  function renderCard() {
    const c = state.cards[state.i];
    $('#card-root').innerHTML = `<article class="card">
      ${renderHead(c)}
      ${renderVerdict(c)}
      ${renderWaste(c)}
      ${renderAttempt(c)}
      ${renderHistory(c)}
      ${renderAlso(c)}
      ${renderAction(c)}
    </article>`;
    renderPager();
    const ta = $('#comment');
    if (ta) {
      /* the reviewer is about to sign this - never make them scroll to read it */
      const fit = () => { ta.style.height = 'auto'; ta.style.height = ta.scrollHeight + 'px'; };
      fit();
      ta.addEventListener('input', () => { state.comments[c.id] = ta.value; fit(); });
    }
  }

  function renderPager() {
    $('#counter').textContent = `${state.i + 1} / ${state.cards.length}`;
    $('#dots').innerHTML = state.cards.map((c, n) => `<li>
      <button type="button" data-go="${n}" aria-current="${n === state.i}"
        data-done="${!!state.posted[c.id]}"
        aria-label="Draft ${n + 1}: ${esc(c.draft.title)}"><span></span></button></li>`).join('');
    $('#prev').disabled = state.i === 0;
    $('#next').disabled = state.i === state.cards.length - 1;
  }

  function go(n) {
    const next = Math.max(0, Math.min(state.cards.length - 1, n));
    if (next === state.i) return;
    state.i = next;
    history.replaceState(null, '', '#' + (next + 1));
    renderCard();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  /* #3 in the URL opens the third draft, so a reviewer can hand one to
   * someone else by pasting a link. */
  function indexFromHash() {
    const n = parseInt((location.hash || '').replace('#', ''), 10);
    return Number.isFinite(n) ? Math.max(0, Math.min(state.cards.length - 1, n - 1)) : 0;
  }

  /* ------------------------------------------------------------- rules */

  const SEED = [
    { id: 'seed-verdict-first', seen: 4,
      text: 'When a decline is not fixable by rewriting, say so before describing anything about the draft.' },
    { id: 'seed-no-shortcut', seen: 2,
      text: 'Do not open with a guideline shortcut. Contributors who have been declined twice have already read it.' },
  ];

  function ruleFor(c, action) {
    const b = c.topBlocker;
    const code = b ? `<code>${esc(b.code)}</code>` : '';

    if (action === 'wrong') {
      if (!b) return null;
      const other = c.otherBlockers[0];
      return other
        ? { id: `order:${b.code}>${other.code}`,
            text: `Stop putting ${code} on top when <code>${esc(other.code)}</code> is also standing — you reordered this one.` }
        : { id: `order:${b.code}`,
            text: `You disagreed with leading on ${code}. Rank it lower until a second reviewer agrees with it.` };
    }

    if (action === 'skip') {
      if (!b) return null;
      return { id: `defer:${b.code}:${b.repeat_count}`,
        text: `Hold drafts blocked only by ${code} after ${b.repeat_count} decline${b.repeat_count > 1 ? 's' : ''} — you passed on this one.` };
    }

    /* action === 'post' */
    if (!b) {
      return { id: 'no-history-no-note',
        text: 'On first submissions, do not carry anything forward. There is nothing to repeat.' };
    }
    if (c.wastedRewrite) {
      return { id: `waste-first:${b.code}`,
        text: `When ${code} is still standing after a substantive rewrite, lead with “rewriting will not clear this” before anything about tone.` };
    }
    if (b.family === 'unknown') {
      return { id: `no-guess:${b.code}`,
        text: `Never state a rewrite verdict for ${code}. Report what changed and leave the call to the reviewer.` };
    }
    if (b.family === 'structural') {
      return { id: `structural-route:${b.code}`,
        text: `When ${code} is the block, offer the merge-and-redirect route instead of advice about the draft.` };
    }
    if (b.family === 'source_existence') {
      if (b.attempt && b.attempt.addedSources > 0) {
        return { id: `name-the-test:${b.code}`,
          text: `When ${code} repeats and sources were added anyway, name what makes a source count instead of repeating the code.` };
      }
      return { id: `same-result:${b.code}`,
        text: `When ${code} repeats with no new sources, say plainly that resubmitting will produce the same result.` };
    }
    /* writing */
    if (b.attempt && b.attempt.touched_target_area) {
      return { id: `credit-the-edit:${b.code}`,
        text: `When an edit touched what ${code} pointed at, acknowledge that in the first line.` };
    }
    return { id: `point-at-new-text:${b.code}`,
      text: `When ${code} is still standing and the new text carries no citations, point at the new text specifically.` };
  }

  function learn(c, action) {
    const r = ruleFor(c, action);
    if (!r) return;
    const existing = state.rules.find((x) => x.id === r.id);
    if (existing) {
      existing.seen += 1;
      existing.isNew = true;
      state.rules = [existing].concat(state.rules.filter((x) => x !== existing));
    } else {
      state.rules.unshift({ id: r.id, text: r.text, seen: 1, on: true, isNew: true, fresh: true });
    }
    renderRules();
    setTimeout(() => { state.rules.forEach((x) => { x.isNew = false; }); renderRules(); }, 2600);
  }

  function renderRules() {
    const list = $('#rules-list');
    $('#rules-count').textContent = state.rules.length;
    if (!state.rules.length) {
      list.innerHTML = '<li class="rules__empty">Nothing yet. Rules appear as you act on drafts.</li>';
      return;
    }
    list.innerHTML = state.rules.map((r, n) => `<li class="rule${r.isNew ? ' rule--new' : ''}${r.on ? '' : ' rule--off'}">
      <div class="rule__text">${r.text}${r.isNew ? '<span class="rule__tag">new</span>' : ''}</div>
      <div class="rule__foot">
        <span class="rule__seen">learned from <b>${r.seen}</b> of your review${r.seen > 1 ? 's' : ''}${r.fresh ? '' : ' · earlier sessions'}</span>
        <button class="linkbtn" data-rule="${n}" type="button">${r.on ? 'turn off' : 'turn on'}</button>
      </div>
    </li>`).join('');
  }

  /* ------------------------------------------------------------- toast */

  let toastTimer = null;
  function toast(html) {
    const t = $('#toast');
    t.innerHTML = html;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, 4200);
  }

  /* ------------------------------------------------------------ events */

  function onAction(act) {
    const c = state.cards[state.i];
    if (act === 'post') {
      state.posted[c.id] = true;
      learn(c, 'post');
      toast(`<b>Comment posted to ${esc(c.draft.title.replace(/^Draft:/, 'Draft talk:'))} as ${esc(state.reviewer)}.</b>
             <small>Demo build running on fixtures — nothing was sent to Wikipedia.</small>`);
      renderCard();
      setTimeout(() => go(state.i + 1), 900);
    } else if (act === 'wrong') {
      learn(c, 'wrong');
      toast(`<b>Noted.</b><small>The ordering rule on the right now reflects this. No verdict changed.</small>`);
    } else if (act === 'skip') {
      learn(c, 'skip');
      go(state.i + 1);
    } else if (act === 'afch') {
      toast(`<b>AFCH handles the accept/decline.</b>
             <small>First Reader stops here on purpose: it never writes a decision to Wikipedia.</small>`);
    }
  }

  document.addEventListener('click', (e) => {
    const go_ = e.target.closest('[data-go]');
    if (go_) { go(Number(go_.dataset.go)); return; }
    const act = e.target.closest('[data-act]');
    if (act) { onAction(act.dataset.act); return; }
    const rule = e.target.closest('[data-rule]');
    if (rule) { const r = state.rules[Number(rule.dataset.rule)]; r.on = !r.on; renderRules(); return; }
  });

  $('#prev').addEventListener('click', () => go(state.i - 1));
  $('#next').addEventListener('click', () => go(state.i + 1));
  $('#reset').addEventListener('click', () => {
    state.posted = {}; state.comments = {};
    state.rules = SEED.map((r) => Object.assign({}, r, { on: true }));
    state.i = 0; renderCard(); renderRules();
    toast('<b>Session reset.</b><small>Back to the first draft, with only the rules carried in from earlier sessions.</small>');
  });

  document.addEventListener('keydown', (e) => {
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'textarea' || tag === 'input') {
      if (e.key === 'Escape') e.target.blur();
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'ArrowRight' || e.key === 'j') { e.preventDefault(); go(state.i + 1); }
    else if (e.key === 'ArrowLeft' || e.key === 'k') { e.preventDefault(); go(state.i - 1); }
  });

  /* -------------------------------------------------------------- boot */

  $('#card-root').innerHTML = '<div class="loading">Loading review queue…</div>';

  window.FirstReaderData.loadCards().then((payload) => {
    state.cards = payload.cards;
    state.reviewer = payload.reviewer;
    state.now = new Date(payload.generatedAt);
    state.rules = SEED.map((r) => Object.assign({}, r, { on: true }));
    state.i = indexFromHash();
    renderCard();
    renderRules();
    window.addEventListener('hashchange', () => {
      const n = indexFromHash();
      if (n !== state.i) { state.i = n; renderCard(); }
    });
  }).catch((err) => {
    $('#card-root').innerHTML = `<div class="loading">Could not load the queue.<br>${esc(err.message)}</div>`;
  });
})();
