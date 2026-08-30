-- probe: disambiguation_turns / disambiguation_branches — the minimal
-- three-call disambiguation mode's own append-only storage (see
-- disambiguate.py's module docstring). A new, separate mode, not a
-- reshaping of the existing branches/branch_generations tables: those
-- stay exactly as they are, still backing the full tree-based system.
-- disambiguation_branches is the flat counterpart to `branches` —
-- no parent_id, no depth, no requires_evidence — reusing that table's
-- column *shape* (id, session_id, turn_index, statement, status) at
-- the model level (models.py's DisambiguationBranch) without touching
-- the original table, which this mode's own migration must not alter:
-- the full system's HypothesisGenerator still depends on every column
-- branches already has.
--
-- disambiguation_options reuses the existing `option_status` enum type
-- (open/selected/superseded) and the `Option` pydantic model as-is
-- (see models.py) — it is a parallel table, not the same `options`
-- table, only because `options.branch_id` has a hard FK to
-- `branches(id)`: a disambiguation branch lives in a different table,
-- so it cannot satisfy that constraint. The click-resolve *mechanism*
-- (mark an option selected, mark its branch matched, supersede the
-- rest) is reused verbatim in Python; only the two tables it reads and
-- writes are new.
--
-- Append-only, same pattern as branches/options (CLAUDE.md invariant
-- 6/8, and this table's own new entry): no delete method, no DELETE
-- SQL, status moves open -> matched/superseded via UPDATE only.

CREATE TABLE disambiguation_turns (
    id                      UUID PRIMARY KEY,
    session_id              UUID NOT NULL REFERENCES sessions (id) ON DELETE RESTRICT,
    turn_index              INT NOT NULL,
    needs_branches          BOOLEAN NOT NULL,
    turn_had_direct_answer  BOOLEAN NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_disambiguation_turns_session
    ON disambiguation_turns (session_id, turn_index);

CREATE TABLE disambiguation_branches (
    id                       UUID PRIMARY KEY,
    disambiguation_turn_id   UUID NOT NULL REFERENCES disambiguation_turns (id) ON DELETE RESTRICT,
    session_id               UUID NOT NULL REFERENCES sessions (id) ON DELETE RESTRICT,
    turn_index               INT NOT NULL,
    statement                TEXT NOT NULL,
    status                   branch_status NOT NULL DEFAULT 'open',
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_disambiguation_branches_turn
    ON disambiguation_branches (disambiguation_turn_id, status);

CREATE TABLE disambiguation_options (
    id                      UUID PRIMARY KEY,
    branch_id               UUID NOT NULL REFERENCES disambiguation_branches (id) ON DELETE RESTRICT,
    generation_id           UUID NOT NULL REFERENCES disambiguation_turns (id) ON DELETE RESTRICT,
    session_id              UUID NOT NULL REFERENCES sessions (id) ON DELETE RESTRICT,
    turn_index              INT NOT NULL,
    text                    TEXT NOT NULL,
    status                  option_status NOT NULL DEFAULT 'open',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_disambiguation_options_generation
    ON disambiguation_options (generation_id, status);
CREATE INDEX idx_disambiguation_options_branch
    ON disambiguation_options (branch_id);
