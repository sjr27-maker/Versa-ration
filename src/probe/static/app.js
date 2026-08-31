/* probe — calm UI client.
 *
 * Ported from the "Probe UI - Calm.dc.html" design canvas: the layout,
 * the sage/cream theme, the three.js knowledge-graph background and the
 * full-screen branch-generation overlay are all the design's; the
 * mocked submit()/_mockAnswer() are replaced with real calls to the
 * Starlette API in probe/webserver.py.
 *
 * "Branching working dynamically" here means: the generation overlay is
 * driven by real SessionLoop node-progress events streamed over SSE,
 * the branch statements / options in the inspector are the live
 * disambiguation record, and clicking an option resolves its branch
 * through the same handle_turn path the CLI uses.
 */

const $ = (id) => document.getElementById(id);

const state = {
  sessionId: null,
  learner: 'alice-chen',
  stub: false,          // real Gemini; no in-UI toggle
  turnIndex: 0,
  turns: [],            // {q, a, trace, options:[{id,text}], error}
  pendingOptions: [],   // options awaiting a click on the latest turn
  busy: false,
  panel: null,          // null | 'learners' | 'session' | 'story' | 'evidence'
  learnerId: null,
  priorSessions: [],
  learners: [],
  facts: [],
  evidence: [],
  consolidateNote: '',
  panelNote: '',
};

/* node name -> the phase label the overlay shows while it runs */
const PHASE_LABEL = {
  EmbedAndSearchFacts: 'searching memory',
  ConfirmFactMatch: 'checking recall',
  AssessAndBranch: 'assessing ambiguity',
  DisambiguationOptions: 'generating branches',
  FinalAnswer: 'composing answer',
  WriteLearnerFact: 'writing memory',
  BaselineTeach: 'answering',
};

/* ─────────────────────────── rendering ─────────────────────────── */

function render() {
  const empty = state.turns.length === 0 && !state.busy;
  $('emptyHero').hidden = !empty;

  $('turnDotSep').hidden = empty;
  $('turnChip').hidden = empty;
  $('turnChip').textContent = 'turn ' + String(state.turns.length).padStart(2, '0');
  $('thinkingBadge').hidden = !state.busy;
  $('learnerEcho').textContent = state.learner;
  $('draft').placeholder = empty
    ? "e.g. What's the difference between dy/dx and d/dx?"
    : 'Ask a follow-up…';

  $('draft').disabled = state.busy;
  $('sendBtn').disabled = state.busy;
  $('learnerChip').disabled = state.sessionId !== null;
  $('btnLearners').classList.toggle('active', state.panel === 'learners');
  $('btnSession').classList.toggle('active', state.panel === 'session');
  $('btnStory').classList.toggle('active', state.panel === 'story');
  $('btnEvidence').classList.toggle('active', state.panel === 'evidence');
  $('promptHint').textContent = state.busy ? 'thinking…' : 'enter ↵ to send';

  const inner = $('streamInner');
  // wipe everything except the empty hero node
  [...inner.querySelectorAll('.turn, .opt-row')].forEach((n) => n.remove());

  state.turns.forEach((t, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'turn';

    if (t.q) {
      const qRow = document.createElement('div');
      qRow.className = 'q-row';
      const qb = document.createElement('div');
      qb.className = 'q-bubble';
      qb.textContent = t.q;
      qRow.appendChild(qb);
      wrap.appendChild(qRow);
    }

    if (t.error) {
      const row = document.createElement('div');
      row.className = 'a-row';
      row.innerHTML = '<div class="p-avatar">p</div>';
      const body = document.createElement('div');
      body.className = 'a-body';
      const panel = document.createElement('div');
      panel.className = 'a-panel';
      panel.innerHTML = '<span class="err-note">' + escapeHtml(t.error) + '</span>';
      body.appendChild(panel);
      row.appendChild(body);
      wrap.appendChild(row);
    } else if (t.a) {
      const row = document.createElement('div');
      row.className = 'a-row';
      row.innerHTML = '<div class="p-avatar">p</div>';
      const body = document.createElement('div');
      body.className = 'a-body';
      const panel = document.createElement('div');
      panel.className = 'a-panel probe-panel';
      panel.textContent = t.a;
      body.appendChild(panel);
      if (t.trace) {
        const tr = document.createElement('div');
        tr.className = 'trace-line';
        tr.innerHTML = t.trace;
        body.appendChild(tr);
      }
      row.appendChild(body);
      wrap.appendChild(row);
    }
    inner.appendChild(wrap);

    // options belong to the last turn only, and only while still pending
    const isLast = i === state.turns.length - 1;
    if (isLast && state.pendingOptions.length && !state.busy) {
      const optRow = document.createElement('div');
      optRow.className = 'opt-row';
      state.pendingOptions.forEach((o) => {
        const b = document.createElement('button');
        b.className = 'opt-btn';
        b.textContent = o.text;
        b.onclick = () => submit(o.text, o.id);
        optRow.appendChild(b);
      });
      inner.appendChild(optRow);
    }
  });

  $('stream').scrollTop = $('stream').scrollHeight;
  renderDrawer();
}

/* ────────────── left drawer: learners / sessions / learner memory ────────── */

const DRAWER_TITLE = {
  learners: 'LEARNERS',
  session: 'SESSIONS',
  story: 'LEARNER MEMORY',
  evidence: 'VERIFICATION EVIDENCE',
};

function openPanel(name) {
  state.panel = state.panel === name ? null : name;
  if (state.panel === 'learners') loadLearners();
  if (state.panel === 'session') loadPriorSessions();
  if (state.panel === 'story') loadFacts();
  if (state.panel === 'evidence') loadEvidence();
  render();
}

function renderDrawer() {
  const open = state.panel !== null;
  $('drawer').hidden = !open;
  $('drawer').classList.toggle('wide', open && state.panel === 'evidence');
  if (!open) return;
  $('drawerTitle').textContent = DRAWER_TITLE[state.panel] || 'PANEL';
  const body = $('drawerBody');
  if (state.panel === 'learners') body.innerHTML = learnersHtml();
  else if (state.panel === 'session') body.innerHTML = sessionsHtml();
  else if (state.panel === 'story') body.innerHTML = storyHtml();
  else if (state.panel === 'evidence') body.innerHTML = evidenceHtml();
  wireDrawer(body);
}

function wireDrawer(body) {
  body.querySelectorAll('[data-resume]').forEach((b) => {
    b.onclick = () => resumeSession(b.getAttribute('data-resume'));
  });
  body.querySelectorAll('[data-pick-learner]').forEach((el) => {
    el.onclick = () => selectLearner(el.getAttribute('data-pick-learner'));
  });
  const copy = $('copySess');
  if (copy) {
    copy.onclick = () => {
      navigator.clipboard && navigator.clipboard.writeText(state.sessionId);
      copy.textContent = 'copied';
      setTimeout(() => copy && (copy.textContent = 'copy id'), 1200);
    };
  }
  const con = $('consolidateBtn');
  if (con) con.onclick = consolidate;
  const mk = $('makeLearnerBtn');
  if (mk) mk.onclick = createLearner;
}

/* --- learners --- */

function learnersHtml() {
  const rows = state.learners
    .map((l) => {
      const name = l.label || l.id;
      const cur = l.id === state.learnerId ? ' current' : '';
      return (
        '<div class="learner-row' + cur + '" data-pick-learner="' + l.id + '">' +
        '<span>' + escapeHtml(name) +
        (l.display_name ? ' <span class="meta">(' + escapeHtml(l.display_name) + ')</span>' : '') +
        '</span><span class="meta">' + l.session_count +
        ' session' + (l.session_count === 1 ? '' : 's') + '</span></div>'
      );
    })
    .join('');
  return (
    '<div class="insp-sec"><div class="label">EXISTING</div>' +
    (rows || '<div class="muted">No learners yet.</div>') +
    '</div>' +
    '<div class="insp-sec"><div class="label">NEW LEARNER</div>' +
    '<input class="mini-input" id="newLearnerLabel" placeholder="label (e.g. alice-chen)" autocomplete="off">' +
    '<input class="mini-input" id="newLearnerDisplay" placeholder="display name (optional)" autocomplete="off">' +
    '<button class="btn" id="makeLearnerBtn" style="margin-top:8px">Create learner</button>' +
    (state.panelNote
      ? '<div class="' + (state.panelNote[0] === '!' ? 'err-note' : 'ok-note') +
        '" style="margin-top:8px">' + escapeHtml(state.panelNote.replace(/^!/, '')) + '</div>'
      : '<div class="muted" style="margin-top:6px">Picking or creating a ' +
        'learner sets who the next new session belongs to. A session in ' +
        'progress keeps its own learner until you start a new one.</div>') +
    '</div>'
  );
}

async function loadLearners() {
  try {
    const res = await fetch('/api/learners');
    if (res.ok) {
      state.learners = (await res.json()).learners || [];
      if (state.panel === 'learners') renderDrawer();
    }
  } catch (_e) {
    /* non-fatal */
  }
}

function selectLearner(id) {
  const l = state.learners.find((x) => x.id === id);
  if (!l) return;
  // Only warn if there's actual chat to lose.
  if (
    state.turns.length &&
    l.id !== state.learnerId &&
    !confirm('Switch to ' + (l.label || l.id) + '? This clears the current chat (the session stays saved).')
  ) {
    return;
  }
  state.learner = l.label || l.id;
  state.learnerId = l.id;
  state.sessionId = null;
  state.turns = [];
  state.pendingOptions = [];
  state.turnIndex = 0;
  state.priorSessions = [];
  state.facts = [];
  state.consolidateNote = '';
  state.panelNote = '';
  $('learnerChip').value = state.learner;
  // jump straight to that learner's sessions — that's what a click means
  state.panel = 'session';
  loadPriorSessions();
  loadFacts();
  render();
}

async function createLearner() {
  const label = ($('newLearnerLabel').value || '').trim();
  const display_name = ($('newLearnerDisplay').value || '').trim();
  if (!label && !display_name) { state.panelNote = '!give a label or display name'; renderDrawer(); return; }
  try {
    const res = await fetch('/api/learners', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ label, display_name }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'create failed');
    state.panelNote = 'created ' + (data.label || data.id);
    await loadLearners();
    selectLearner(data.id);
  } catch (e) {
    state.panelNote = '!' + e.message;
    renderDrawer();
  }
}

/* --- sessions (resume + consolidate) --- */

function sessionsHtml() {
  const current = state.sessionId
    ? '<div class="insp-sec"><div class="label">CURRENT</div>' +
      '<div class="field">' + state.sessionId + '</div>' +
      '<div style="display:flex;gap:8px;margin-top:8px;align-items:center">' +
      '<button class="btn ghost" id="copySess">copy id</button>' +
      '<span class="muted">' + escapeHtml(state.learner) + '</span></div></div>'
    : '<div class="insp-sec"><div class="label">CURRENT</div>' +
      '<div class="muted">No active session — send a message to start one.</div></div>';

  const rows = state.priorSessions
    .map((s) => {
      const when = new Date(s.created_at).toLocaleString([], {
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
      });
      const isCur = s.session_id === state.sessionId;
      return (
        '<div class="sess-row' + (isCur ? ' active' : '') + '"><span class="when">' + when + '</span>' +
        '<span class="turns">' + s.turn_count + ' turn' + (s.turn_count === 1 ? '' : 's') + '</span>' +
        (isCur ? '<span class="turns">current</span>'
               : '<button class="btn ghost" data-resume="' + s.session_id + '">resume</button>') +
        '</div>'
      );
    })
    .join('');

  const end = state.sessionId
    ? '<div class="insp-sec"><div class="label">END</div>' +
      '<button class="btn" id="consolidateBtn">End session &amp; consolidate</button>' +
      (state.consolidateNote
        ? '<div class="' + (state.consolidateNote[0] === '!' ? 'err-note' : 'ok-note') +
          '" style="margin-top:8px">' + escapeHtml(state.consolidateNote.replace(/^!/, '')) + '</div>'
        : '<div class="muted" style="margin-top:6px">Labels this session’s facts ' +
          'and compares them against this learner’s thinking-style candidates ' +
          '(memory.py steps 6–8).</div>') +
      '</div>'
    : '';

  return (
    current +
    '<div class="insp-sec"><div class="label">' +
    (state.learnerId ? 'SESSIONS FOR ' + escapeHtml(state.learner).toUpperCase() : 'SESSIONS') +
    '</div>' +
    (state.learnerId
      ? rows || '<div class="muted">No sessions for this learner yet.</div>'
      : '<div class="muted">Pick a learner first (Learners panel) or start a session.</div>') +
    '</div>' +
    end
  );
}

async function loadPriorSessions() {
  if (!state.learnerId) { state.priorSessions = []; if (state.panel === 'session') renderDrawer(); return; }
  try {
    const res = await fetch('/api/learners/' + state.learnerId + '/sessions');
    if (res.ok) {
      state.priorSessions = (await res.json()).sessions || [];
      if (state.panel === 'session') renderDrawer();
    }
  } catch (_e) {
    /* non-fatal */
  }
}

/* --- story (learner memory facts) --- */

function storyHtml() {
  const who = state.learner || 'this learner';
  if (!state.learnerId) {
    return '<div class="muted">No learner selected — pick one from the Learners panel, or start a session.</div>';
  }
  if (!state.facts.length) {
    return (
      '<div class="muted">Learner: ' + escapeHtml(who) + ' — every resolved turn ' +
      'this memory layer has on record, in order, across every session.<br><br>' +
      'No facts recorded yet — this learner has no minimal_branch turns that ' +
      'resolved something (a BASELINE session, or one that only ever raised ' +
      'options), or the memory layer wasn’t configured for those sessions.</div>'
    );
  }
  return (
    '<div class="muted" style="margin-bottom:12px">Learner: ' + escapeHtml(who) +
    ' — every resolved turn, in order, across every session.</div>' +
    state.facts
      .map((f) => {
        const when = new Date(f.created_at).toLocaleString([], {
          month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
        });
        const branch = f.fact_type === 'branch_resolution';
        return (
          '<div class="story-card">' +
          '<div class="kind">' + (branch ? 'BRANCH RESOLUTION' : 'DIRECT ANSWER') + '</div>' +
          '<div class="meta">session ' + f.session_id.slice(0, 8) + ' · turn ' + f.turn_index + ' · ' + when + '</div>' +
          '<div class="sit">' + escapeHtml(who) + ' ' +
          (branch ? 'asked something that turned out to be unclear: ' : 'asked or did this: ') +
          '<em>' + escapeHtml(f.situation) + '</em></div>' +
          '<div class="res">' + (branch ? 'Resolved like this: ' : 'Answered like this: ') +
          escapeHtml(f.resolution) + '</div></div>'
        );
      })
      .join('')
  );
}

async function loadFacts() {
  if (!state.learnerId) { state.facts = []; if (state.panel === 'story') renderDrawer(); return; }
  try {
    const res = await fetch('/api/learners/' + state.learnerId + '/facts');
    if (res.ok) {
      state.facts = (await res.json()).facts || [];
      if (state.panel === 'story') renderDrawer();
    }
  } catch (_e) {
    /* non-fatal */
  }
}

/* --- verification evidence (evidence_records) --- */

const EV_PART_LABEL = {
  part_1_within_session: 'PART 1 · within-session adaptation',
  part_2_cross_session: 'PART 2 · cross-session memory',
  part_3_thinking_style: 'PART 3 · thinking-style mechanism',
};

function evidenceHtml() {
  const staged = state.evidence.filter((r) => r.source_type === 'staged_verification');
  const note =
    '<div class="ev-note"><b>staged_verification</b> rows are deliberate ' +
    'scripted runs. They can show a <b>mechanism functions</b> — never ' +
    'that the system adapted to a real student. A confirmed thinking ' +
    'style and multi-session organic adaptation need real elapsed usage, ' +
    'not a testing pass.</div>';
  if (!state.evidence.length) {
    return note + '<div class="muted">No verification findings recorded yet.</div>';
  }
  const cards = state.evidence
    .map((r) => {
      const when = new Date(r.created_at).toLocaleString([], {
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
      });
      const srcCls = r.source_type === 'staged_verification' ? 'staged' : 'organic';
      const partLabel = EV_PART_LABEL[r.part] || r.part;
      let bodyStr = '';
      try { bodyStr = JSON.stringify(r.body, null, 2); } catch (_e) { bodyStr = String(r.body); }
      return (
        '<div class="ev-card">' +
        '<span class="ev-src ' + srcCls + '">' + escapeHtml(r.source_type) + '</span>' +
        '<div class="meta">' + escapeHtml(partLabel) + ' · ' + when + '</div>' +
        '<div class="ev-title">' + escapeHtml(r.title) + '</div>' +
        '<div class="ev-summary">' + escapeHtml(r.summary) + '</div>' +
        '<details><summary>evidence (' + bodyStr.length + ' chars)</summary>' +
        '<pre>' + escapeHtml(bodyStr) + '</pre></details>' +
        '</div>'
      );
    })
    .join('');
  return (
    note +
    '<div class="muted" style="margin-bottom:12px">' +
    state.evidence.length + ' finding' + (state.evidence.length === 1 ? '' : 's') +
    ' · ' + staged.length + ' staged</div>' +
    cards
  );
}

async function loadEvidence() {
  try {
    const res = await fetch('/api/evidence');
    if (res.ok) {
      state.evidence = (await res.json()).records || [];
      if (state.panel === 'evidence') renderDrawer();
    }
  } catch (_e) {
    /* non-fatal */
  }
}

function historyToTurns(history) {
  const turns = [];
  for (const h of history || []) {
    if (h.role === 'student') {
      turns.push({ q: h.text, a: '', trace: '' });
    } else if (turns.length) {
      turns[turns.length - 1].a = h.text;
    }
  }
  return turns;
}

async function resumeSession(sid) {
  if (state.busy) return;
  try {
    const res = await fetch('/api/session/' + sid);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'resume failed');
    state.sessionId = data.session_id;
    state.learner = (data.learner && data.learner.label) || state.learner;
    state.learnerId = data.learner && data.learner.id;
    state.turnIndex = data.turn_index;
    state.turns = historyToTurns(data.turns);
    state.pendingOptions = data.pending_options || [];
    state.consolidateNote = '';
    $('learnerChip').value = state.learner;
    render();
    loadPriorSessions();
    loadFacts();
  } catch (e) {
    state.consolidateNote = '!' + e.message;
    renderDrawer();
  }
}

async function consolidate() {
  if (!state.sessionId) return;
  const btn = $('consolidateBtn');
  if (btn) btn.disabled = true;
  state.consolidateNote = 'consolidating…';
  renderDrawer();
  try {
    const res = await fetch('/api/session/' + state.sessionId + '/consolidate', {
      method: 'POST',
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'consolidate failed');
    state.consolidateNote = data.consolidated
      ? 'Thinking-style candidate ' + data.candidate_id.slice(0, 8) +
        ': “' + data.path_summary + '” (count ' + data.confirmation_count +
        ', ' + data.status + ')'
      : 'Nothing to consolidate — this session resolved nothing (or ran on stub with no facts written).';
  } catch (e) {
    state.consolidateNote = '!' + e.message;
  }
  renderDrawer();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function traceHtml(g) {
  if (!g) return '';
  const parts = [
    'plan = <b>' + (g.branching_skipped_by_memory ? 'memory' : g.branched ? 'branch' : 'direct') + '</b>',
    '<span class="sep">|</span>',
    g.total_call_count + ' LLM calls · ' + (g.duration_ms / 1000).toFixed(1) + 's',
  ];
  if (g.guardrail_fired) parts.push('<span class="sep">|</span>', '<b>guardrail fired</b>');
  return parts.join(' ');
}

/* ─────────────────────────── API ──────────────────────────────── */

async function createSession() {
  const res = await fetch('/api/session', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ learner: state.learner, stub: state.stub }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'session create failed');
  state.sessionId = data.session_id;
  state.learnerId = data.learner && data.learner.id;
  state.turnIndex = data.turn_index;
  state.stub = data.stub;
  state.consolidateNote = '';
  loadPriorSessions();
}

async function submit(text, optionId) {
  text = (text || '').trim();
  if (!text || state.busy) return;

  if (!state.sessionId) {
    state.learner = ($('learnerChip').value || 'alice-chen').trim();
    try {
      await createSession();
    } catch (e) {
      flashError(e.message);
      return;
    }
  }

  state.busy = true;
  if (!optionId) $('draft').value = '';
  // clear the just-clicked option set immediately
  state.pendingOptions = [];
  state.turns.push({ q: text, a: '', trace: '', options: [] });
  render();

  openOverlay(text);

  let genCalls = 0;
  let finished = false;
  try {
    const res = await fetch('/api/session/' + state.sessionId + '/turn', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text, option_id: optionId || null }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || 'turn failed (' + res.status + ')');
    }

    for await (const evt of readSSE(res)) {
      if (evt.phase === 'node') {
        genCalls += 1;
        $('genPhase').textContent = PHASE_LABEL[evt.node] || evt.node;
        $('genCalls').textContent = String(genCalls).padStart(2, '0');
      } else if (evt.phase === 'error') {
        finished = true;
        const last = state.turns[state.turns.length - 1];
        last.error = evt.error;
        break;
      } else if (evt.phase === 'done') {
        finished = true;
        const last = state.turns[state.turns.length - 1];
        state.turnIndex = evt.next_turn_index;
        const g = evt.diagnostics;
        if (g) g.branched = evt.branched;
        last.a = evt.message; // a real answer, or "Which of these did you mean?"
        last.branched = evt.branched;
        last.trace = g ? traceHtml(g) : '';
        state.pendingOptions = evt.branched ? evt.pending_options || [] : [];
        break;
      }
    }
  } catch (e) {
    const last = state.turns[state.turns.length - 1];
    last.error = e.message;
  } finally {
    if (!finished) {
      const last = state.turns[state.turns.length - 1];
      if (!last.a && !last.error) last.error = 'stream ended unexpectedly';
    }
    state.busy = false;
    closeOverlay();
    render();
  }
}

async function* readSSE(res) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n\n')) !== -1) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const line = chunk.split('\n').find((l) => l.startsWith('data: '));
      if (line) {
        try {
          yield JSON.parse(line.slice(6));
        } catch (_e) {
          /* ignore keepalive / malformed */
        }
      }
    }
  }
}

function flashError(msg) {
  state.turns.push({ q: '', a: '', error: msg });
  render();
}

/* ─────────────────────── generation overlay ───────────────────── */

let genStop = null;

function openOverlay(query) {
  $('genQuery').textContent = query;
  $('genPhase').textContent = 'root → level 1';
  $('genCalls').textContent = '00';
  $('genOverlay').hidden = false;
  $('probeBgCanvas').classList.add('thinking');
  startGenScene();
}

function closeOverlay() {
  $('genOverlay').hidden = true;
  $('probeBgCanvas').classList.remove('thinking');
  if (genStop) {
    try { genStop(); } catch (_e) {}
    genStop = null;
  }
}

/* ───────────────────────── three.js scenes ────────────────────── */

let THREE_MOD = null;
async function three() {
  if (!THREE_MOD) {
    THREE_MOD = await import('https://unpkg.com/three@0.160.0/build/three.module.js');
  }
  return THREE_MOD;
}

async function startBackground() {
  const canvas = $('probeBgCanvas');
  let THREE;
  try {
    THREE = await three();
  } catch (_e) {
    return; // offline: the CSS bloom still carries the look
  }
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100);
  camera.position.set(0, 0, 11);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);

  const TEAL = new THREE.Color(0x0f766e);
  const ORANGE = new THREE.Color(0xc2410c);
  const INK = new THREE.Color(0x1e293b);

  const graph = new THREE.Group();
  graph.position.set(0, 0.2, 0);
  scene.add(graph);

  const N = 90, radius = 3.4;
  const nodes = [];
  for (let i = 0; i < N; i++) {
    const y = 1 - (i / (N - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    nodes.push(new THREE.Vector3(Math.cos(theta) * r * radius, y * radius, Math.sin(theta) * r * radius));
  }
  const edges = [];
  const seen = new Set();
  nodes.forEach((n, i) => {
    const d = nodes.map((m, j) => ({ j, d: i === j ? Infinity : n.distanceTo(m) }));
    d.sort((a, b) => a.d - b.d);
    for (let k = 0; k < 3; k++) {
      const j = d[k].j;
      const key = i < j ? i + '-' + j : j + '-' + i;
      if (!seen.has(key)) { seen.add(key); edges.push([i, j]); }
    }
  });
  const nodeMats = nodes.map((p, i) => {
    const accent = i % 11 === 0;
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(accent ? 0.09 : 0.055, 10, 10),
      new THREE.MeshBasicMaterial({ color: accent ? ORANGE : INK, transparent: true, opacity: accent ? 0.85 : 0.55 })
    );
    m.position.copy(p);
    m.userData = { baseOpa: m.material.opacity, phase: Math.random() * Math.PI * 2 };
    graph.add(m);
    return m;
  });
  const ep = new Float32Array(edges.length * 6);
  edges.forEach(([a, b], i) => {
    ep.set([nodes[a].x, nodes[a].y, nodes[a].z, nodes[b].x, nodes[b].y, nodes[b].z], i * 6);
  });
  const eg = new THREE.BufferGeometry();
  eg.setAttribute('position', new THREE.BufferAttribute(ep, 3));
  const edgeLines = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: TEAL, transparent: true, opacity: 0.28 }));
  graph.add(edgeLines);

  const pulses = [];
  for (let i = 0; i < 24; i++) {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(0.075, 10, 10),
      new THREE.MeshBasicMaterial({ color: ORANGE, transparent: true, opacity: 0 })
    );
    pulses.push({ m, e: (Math.random() * edges.length) | 0, t: Math.random(), s: 0.3 + Math.random() * 0.4 });
    graph.add(m);
  }
  const cage = new THREE.Mesh(
    new THREE.IcosahedronGeometry(radius * 1.08, 2),
    new THREE.MeshBasicMaterial({ color: TEAL, wireframe: true, transparent: true, opacity: 0.08 })
  );
  graph.add(cage);

  let mx = 0, my = 0;
  const onMove = (e) => { mx = e.clientX / innerWidth - 0.5; my = e.clientY / innerHeight - 0.5; };
  const onResize = () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  };
  addEventListener('mousemove', onMove);
  addEventListener('resize', onResize);

  let stopped = false;
  const clock = new THREE.Clock();
  let think = 0;
  (function loop() {
    if (stopped) return;
    const t = clock.getElapsedTime();
    const thinking = canvas.classList.contains('thinking');
    think += ((thinking ? 1 : 0) - think) * 0.03;
    graph.rotation.y = t * 0.06;
    graph.rotation.x = Math.sin(t * 0.1) * 0.15;
    nodeMats.forEach((m) => {
      m.material.opacity = m.userData.baseOpa * (0.75 + 0.25 * Math.sin(t * 0.6 + m.userData.phase));
    });
    edgeLines.material.opacity = 0.22 + think * 0.18;
    pulses.forEach((p) => {
      p.t += p.s * 0.016 * (0.5 + think * 1.5);
      if (p.t >= 1) { p.t = 0; p.e = (Math.random() * edges.length) | 0; }
      const [a, b] = edges[p.e];
      p.m.position.lerpVectors(nodes[a], nodes[b], p.t);
      p.m.material.opacity = (0.4 + think * 0.6) * Math.sin(p.t * Math.PI);
    });
    cage.rotation.x = -t * 0.04;
    cage.rotation.y = t * 0.03;
    camera.position.x += (mx * 0.5 - camera.position.x) * 0.025;
    camera.position.y += (-my * 0.35 - camera.position.y) * 0.025;
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  })();

  addEventListener('beforeunload', () => {
    stopped = true;
    removeEventListener('mousemove', onMove);
    removeEventListener('resize', onResize);
  });
}

async function startGenScene() {
  const canvas = $('probeGenCanvas');
  let THREE;
  try {
    THREE = await three();
  } catch (_e) {
    return;
  }
  const SAGE = new THREE.Color(0x7ea068);
  const SAGE_LIGHT = new THREE.Color(0xb8d494);
  const ORANGE = new THREE.Color(0xc2410c);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 100);
  camera.position.set(0, 0.4, 10);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);

  const nodes = [], edges = [];
  const build = (pos, dir, depth, maxDepth, fanout, parent) => {
    const idx = nodes.length;
    nodes.push({ pos: pos.clone(), depth });
    if (parent >= 0) edges.push({ a: parent, b: idx, depth });
    if (depth >= maxDepth) return;
    const n = fanout[depth];
    for (let i = 0; i < n; i++) {
      const spread = (i - (n - 1) / 2) * (0.85 - depth * 0.15);
      const newDir = dir.clone()
        .applyAxisAngle(new THREE.Vector3(0, 1, 0), spread * 0.6)
        .applyAxisAngle(new THREE.Vector3(1, 0, 0), (Math.random() - 0.5) * 0.35)
        .applyAxisAngle(new THREE.Vector3(0, 0, 1), spread * 0.9)
        .normalize();
      const len = 2.2 - depth * 0.5;
      build(pos.clone().add(newDir.clone().multiplyScalar(len)), newDir, depth + 1, maxDepth, fanout, idx);
    }
  };
  build(new THREE.Vector3(-3.6, 0, 0), new THREE.Vector3(1, 0, 0), 0, 3, [4, 3, 2], -1);

  const group = new THREE.Group();
  scene.add(group);
  const nodeMeshes = nodes.map((nd, i) => {
    const isRoot = i === 0, isLeaf = nd.depth === 3;
    const size = isRoot ? 0.18 : isLeaf ? 0.09 : 0.14 - nd.depth * 0.02;
    const color = isRoot ? ORANGE : isLeaf ? SAGE_LIGHT : SAGE;
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(size, 16, 16),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0 })
    );
    m.position.copy(nd.pos);
    m.userData = { depth: nd.depth, appearAt: nd.depth * 0.55 };
    group.add(m);
    const halo = new THREE.Mesh(
      new THREE.SphereGeometry(size * 2.4, 16, 16),
      new THREE.MeshBasicMaterial({ color: isRoot ? ORANGE : SAGE_LIGHT, transparent: true, opacity: 0, depthWrite: false })
    );
    halo.position.copy(nd.pos);
    group.add(halo);
    m.userData.halo = halo;
    return m;
  });
  const edgeMeshes = edges.map((e) => {
    const from = nodes[e.a].pos, to = nodes[e.b].pos;
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([from, from.clone()]),
      new THREE.LineBasicMaterial({ color: SAGE, transparent: true, opacity: 0 })
    );
    line.userData = { from, to, depth: e.depth, appearAt: e.depth * 0.55 + 0.15 };
    group.add(line);
    return line;
  });
  const pulses = [];
  for (let i = 0; i < 10; i++) {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(0.08, 10, 10),
      new THREE.MeshBasicMaterial({ color: ORANGE, transparent: true, opacity: 0 })
    );
    pulses.push({ m, e: (Math.random() * edges.length) | 0, t: Math.random(), s: 0.5 + Math.random() * 0.6 });
    group.add(m);
  }

  const onResize = () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  };
  addEventListener('resize', onResize);

  let stopped = false;
  genStop = () => {
    stopped = true;
    removeEventListener('resize', onResize);
    try { renderer.dispose(); } catch (_e) {}
  };

  const start = performance.now();
  (function loop() {
    if (stopped) return;
    const t = (performance.now() - start) / 1000;
    nodeMeshes.forEach((m) => {
      const age = t - m.userData.appearAt;
      if (age <= 0) { m.material.opacity = 0; m.userData.halo.material.opacity = 0; m.scale.setScalar(0.01); return; }
      const ease = 1 - Math.pow(1 - Math.min(1, age / 0.55), 3);
      const bounce = 1 + Math.sin(ease * Math.PI) * 0.15 * (1 - ease);
      m.scale.setScalar(ease * bounce);
      const targetOpa = m.userData.depth === 0 ? 1 : 0.9 - m.userData.depth * 0.05;
      m.material.opacity = targetOpa * ease;
      m.userData.halo.material.opacity = 0.22 * ease * (0.7 + 0.3 * Math.sin(t * 2 + m.userData.depth));
      m.userData.halo.scale.setScalar(ease * (1 + Math.sin(t * 1.5 + m.userData.depth) * 0.05));
    });
    edgeMeshes.forEach((line) => {
      const { from, to, depth, appearAt } = line.userData;
      const age = t - appearAt;
      if (age <= 0) { line.material.opacity = 0; return; }
      const ease = Math.min(1, age / 0.4);
      line.geometry.dispose();
      line.geometry = new THREE.BufferGeometry().setFromPoints([from, from.clone().lerp(to, ease)]);
      line.material.opacity = (0.6 - depth * 0.1) * ease;
    });
    const pulseOn = Math.min(1, Math.max(0, (t - 1.2) / 0.5));
    pulses.forEach((p) => {
      p.t += p.s * 0.016;
      if (p.t >= 1) { p.t = 0; p.e = (Math.random() * edges.length) | 0; }
      const e = edges[p.e];
      if (!e) return;
      p.m.position.lerpVectors(nodes[e.a].pos, nodes[e.b].pos, p.t);
      p.m.material.opacity = pulseOn * Math.sin(p.t * Math.PI) * 0.9;
    });
    group.rotation.y = Math.sin(t * 0.4) * 0.35 + t * 0.05;
    group.rotation.x = Math.sin(t * 0.3) * 0.08;
    camera.position.z = 10 - Math.min(t, 3) * 0.4;
    camera.lookAt(0.5, 0, 0);
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  })();
}

/* ─────────────────────────── wiring ───────────────────────────── */

$('sendBtn').onclick = () => submit($('draft').value);
$('draft').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') submit($('draft').value);
});
$('learnerChip').addEventListener('input', () => {
  if (state.sessionId) return; // locked once a session exists
  state.learner = ($('learnerChip').value || '').trim() || 'alice-chen';
  // typing a fresh name here means "some other learner" — drop the
  // resolved id so the Sessions/Story panels don't show a stale one
  // until the next session pins it down
  state.learnerId = null;
  state.priorSessions = [];
  state.facts = [];
  $('learnerEcho').textContent = state.learner;
});
$('btnLearners').onclick = () => openPanel('learners');
$('btnSession').onclick = () => openPanel('session');
$('btnStory').onclick = () => openPanel('story');
$('btnEvidence').onclick = () => openPanel('evidence');
$('drawerClose').onclick = () => {
  state.panel = null;
  render();
};
$('btnNew').onclick = () => {
  if (state.busy) return;
  state.sessionId = null;
  state.turnIndex = 0;
  state.turns = [];
  state.pendingOptions = [];
  state.priorSessions = [];
  state.consolidateNote = '';
  $('draft').value = '';
  render();
};

startBackground();
render();
