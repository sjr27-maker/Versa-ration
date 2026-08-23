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
