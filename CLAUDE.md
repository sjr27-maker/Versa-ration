# probe

## Invariants

### The hypothesis store is append-only

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
