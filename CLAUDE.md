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
