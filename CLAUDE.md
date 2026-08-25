# probe

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
