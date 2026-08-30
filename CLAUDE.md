# probe

## Setup

Copy `.env.example` to `.env` and fill in:

- `DATABASE_URL` — Postgres connection string (dev DB and the test suite
  share this by default; running `pytest` wipes and rebuilds it from
  migrations).
- `GEMINI_API_KEY` — required for any real (non-stub) LLM call. Get one
  at https://aistudio.google.com/apikey. Every `probe` command that
  calls an LLM (`chat`, `seed-graph`) accepts `--stub` to run against
  `StubLLMClient` instead, which needs no key and costs nothing.

`GEMINI_MODEL_FAST` / `GEMINI_MODEL_CAPABLE` / `GEMINI_MODEL_BEST` are
optional overrides for the tier→model mapping in `model_config.py` —
only needed if the defaults there go stale (Gemini preview model ids
shift over time).

Run `probe web` for the local Streamlit UI (`src/probe/webui/`) —
setup, running sessions, and the learner portrait, no auth, local
only. It's additive: every `probe` CLI command (`chat`, `seed-graph`,
`review-revisions`, `portrait`) still works exactly as before.

## Invariants

### 1. The hypothesis store is append-only

The `HypothesisStore` (and any store that persists reasoning state) must
never delete rows. Concretely:

- No `delete` / `remove` methods on the store class.
- No `DELETE` SQL anywhere in the store module or its migrations.
- Retirement is modeled by moving rows to the `archived` tier, not by
  removing them. Archived hypotheses can be brought back with
  `resurrect()`.
- Evidence is appended, never mutated in place. `reweight()` records new
  probability/confidence *and* appends the evidence ref that justified
  the update — it does not overwrite prior evidence.

Why: probe's whole premise is auditing how beliefs evolved over time.
Deleting a hypothesis or overwriting its evidence destroys the record
we're trying to build. If something "shouldn't be there anymore," archive
it; the trail matters more than tidiness.

### 2. Every node call is persisted to `node_calls`

Every invocation of a Node's primary async method (`Node.run(...)`) must
be recorded to the `node_calls` table with:

- `node_name`
- `session_id`
- `turn_index`
- `input_json`  (kwargs to `run()`, JSON-serialized; non-serializable
  dependencies like `HypothesisStore` are excluded)
- `output_json` (return value, JSON-serialized)
- `timestamp`

This is enforced by routing every node call through
`SessionLoop._call_node()` rather than invoking `node.run(...)` directly.
Do not call `node.run(...)` from production code paths outside the loop.

Why: the audit trail is the product. If a node executed and we don't
have its inputs+outputs on disk, we can't reconstruct how a belief
changed — which defeats the point of building probe in the first place.

### 3. Value-function terms are individually disable-able

Every term in the `ValueFunction` (learning_value, information_value,
long_term_value, time_cost, cognitive_cost, frustration_risk) must be
independently toggle-able via `ValueFunctionConfig` without any code
changes. A disabled term contributes 0 to `score()` and skips its LLM
calls entirely. The full six-term breakdown must survive on every
`ActionScore` even when some terms are zero, so ablation runs are
comparable side-by-side in `node_calls`.

Why: this codebase is a research artifact whose main question is which
terms actually matter. If turning one off requires editing code, we
can't run apples-to-apples ablations. Keep the config knob, keep the
breakdown, don't collapse to a single float.

### 4. The concept graph is append-only

`ConceptGraph` must never delete rows. Concretely:

- No `delete` / `remove` methods on the class.
- No `DELETE` SQL anywhere in the `concept_graph` module or its
  migrations.
- Verified by the same AST-based check used for invariant 1 (scan for
  delete/remove-prefixed function names and DELETE inside string
  literals, excluding docstrings).

This is a separate invariant from #1, not a restatement of it: the
rationale is different, so it gets its own entry rather than being
folded into the hypothesis-store rule.

Why: the concept graph is seeded once (`probe seed-graph`) and frozen —
it isn't a record of evolving belief the way hypotheses are. The reason
it can't be deleted from is referential, not auditability: `LearnerOverlay`
holds an FK to `concept_nodes.id`. Removing a concept out from under it
would either orphan overlay rows or require a cascade that silently
destroys learner state. `ON DELETE RESTRICT` on that FK enforces this at
the schema level; no delete method enforces it at the code level.

### 5. World-model revisions are append-only

`WorldModelRevisionStore` must never delete rows. Concretely:

- No `delete` / `remove` methods on the class.
- No `DELETE` SQL anywhere in the `revision` module or its migrations.
- Resolution is modeled by moving a revision's `status` from `pending`
  to `approved` or `rejected` via UPDATE, not by removing it. A
  rejected revision stays on record as a rejected claim, not as if it
  had never been proposed.
- `approve()` never overwrites `proposed_change` — it records the
  human-confirmed structured edit separately, as
  `applied_field_updates`, on the same row. The original free-text
  claim and the edit that was actually applied both remain visible.

This is a separate invariant from #1 and #4, not a restatement of
either: the rationale is again distinct, so it gets its own entry.

Why: a `WorldModelRevision` is a claim about the concept graph, evidence-
backed the same way a `Hypothesis` is a claim about the learner — and
probe's premise (invariant 1) is auditing how *all* beliefs evolved, not
just learner-facing ones. Deleting a resolved revision would erase the
record of what was proposed, what a human decided about it, and why —
exactly the trail that makes a rejected-but-plausible claim or an
approved edit's justification recoverable later.

### 6. The branch store is append-only

`BranchStore` (`branches`/`branch_generations`) must never delete rows.
Concretely:

- No `delete` / `remove` methods on the class.
- No `DELETE` SQL anywhere in the `branches` module or its migrations.
- Resolution is modeled by moving a branch's `status` from `open` to
  `matched`, `unmatched`, or `superseded` via UPDATE, not by removing
  it. A generation that turned out to predict nothing right stays on
  record as `unmatched`, not as if it had never been generated.
- Verified by the same AST-based check used for invariants 1 and 4.

This is a separate invariant from #1, not a restatement of it: the
rationale is different, so it gets its own entry rather than being
folded into the hypothesis-store rule.

Why: `HypothesisGenerator` rebuilds a speculative prediction tree every
turn and discards most of it — but "discard" means retiring branches to
a terminal status, the same way a hypothesis retires to `archived`
rather than disappearing, not erasing the record that a generation
happened and what it predicted. Crucially, this store does **not**
write into `HypothesisStore`: a branch match is a single-turn,
episodic signal, and probe's premise (invariant 1) is auditing
*confirmed*, evidence-backed belief — promoting a branch into a real
`Hypothesis` on one coincidental match would let episodic noise
corrupt that durable record. Only a future consolidation step (not
built yet — deliberately deferred until real match data exists to
define "a pattern that repeats") would ever bridge the two; until
then, the wall between episodic branches and durable hypotheses is
itself part of what this invariant protects.

### 7. Turn diagnostics and hypothesis tier changes are append-only

`TurnDiagnosticsStore` (`turn_diagnostics`) and
`HypothesisStore.list_tier_changes`'s backing table
(`hypothesis_tier_changes`) must never delete rows. Concretely:

- No `delete` / `remove` methods on either class.
- No `DELETE` SQL anywhere in `diagnostics.py`, or in `store.py`'s
  `hypothesis_tier_changes` writes, or in either one's migrations.
- `turn_diagnostics` has one row per `handle_turn()` call, written
  once, never updated afterward — a turn's recorded diagnostics
  (call counts, whether the guardrail fired, warnings, whether Teach
  failed) are a historical fact the instant that turn finishes.
- `hypothesis_tier_changes` only ever grows via `retier()`'s INSERT on
  a real transition — never mutated or pruned.
- Verified by the same AST-based check used for invariants 1, 4, and 6.

This is a separate invariant from the others, not a restatement of
any of them: `turn_diagnostics` and `hypothesis_tier_changes` are
audit trail in the same spirit as `node_calls` (invariant 2), but they
didn't exist when that invariant was written and are new stores in
their own right, so they get their own entry rather than being
silently folded into invariant 2's wording after the fact.

Why: both exist specifically so the web UI can *read* what already
happened instead of recomputing it — a per-turn call-count breakdown,
whether `MAX_CALLS_PER_TURN` fired, a hypothesis's tier history. If
either could be edited or pruned, "zero business logic in the UI"
would quietly stop being true: a UI that can't trust the record to be
complete has to start re-deriving things itself, which is exactly the
failure mode this whole feature was built to avoid.

### 8. The options store is append-only

`OptionStore` (`options`) must never delete rows. Concretely:

- No `delete` / `remove` methods on the class.
- No `DELETE` SQL anywhere in the `options` module or its migrations.
- Resolution is modeled by moving an option's `status` from `open` to
  `selected` or `superseded` via UPDATE, not by removing it. An option
  the student never clicked stays on record as `superseded`, not as if
  it had never been offered.
- Verified by the same AST-based check used for invariants 1, 4, 6,
  and 7.

This is a separate invariant from invariant 6, not a restatement of
it: options and branches are different tables with different
rationale, so it gets its own entry rather than being folded into the
branch-store wording.

Why: an option is the record of what was actually offered to the
student and which claim they did or didn't affirm — the evidence trail
for `Branch.evidence_satisfied`. Deleting a superseded option would
erase proof that a specific, unambiguous choice was on the table and
not taken, which is exactly the kind of fact CLAUDE.md invariant 1's
"audit how beliefs evolved" premise depends on, extended to the
options channel: an option is not written into `HypothesisStore`
either (same wall as invariant 6 describes for branches) — it only
ever flips `evidence_satisfied` on the branch it maps to.

### 9. The disambiguation store is append-only

`DisambiguationStore` (`disambiguation_turns`/`disambiguation_branches`/
`disambiguation_options`) must never delete rows. Concretely:

- No `delete` / `remove` methods on the class.
- No `DELETE` SQL anywhere in the `disambiguate` module or its
  migrations.
- Resolution is modeled by moving a branch's or option's `status` from
  `open` to `matched`/`selected` or `superseded` via UPDATE, not by
  removing it — same transitions as invariants 6 and 8, on this mode's
  own tables.
- `disambiguation_turns` is written once per `AssessAndBranch` call,
  unconditionally, whether or not `needs_branches` fires — a turn
  judged unambiguous is still a queryable row with zero branches, not
  a gap.
- Verified by the same AST-based check used for invariants 1, 4, 6, 7,
  and 8.

This is a separate invariant from invariants 6 and 8, not a
restatement of either: `ReasoningMode.DISAMBIGUATE` (see ablation.py)
is a wholly separate reasoning architecture from the branch tree/
options system those invariants describe, running against its own
parallel tables rather than the `branches`/`options` tables — see
disambiguate.py's module docstring for why those tables couldn't be
literally reused (`options.branch_id`'s FK to `branches(id)`) and,
more importantly, why they shouldn't be: the existing tree-based
system still depends on every column `branches` already has, and this
new mode must not require altering it.

Why: same rationale as invariants 6 and 8, extended to this mode — a
distinct reading the student typed past, or a branch set generated for
a message later judged unambiguous, is still a fact about how this
mode reasoned about that turn. Deleting any of it would erase the same
kind of "what was actually offered, and what happened to it" trail
invariant 8 protects, just for a different architecture.
