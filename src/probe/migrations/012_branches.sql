-- probe: branches — a speculative, regenerated-each-turn prediction
-- tree, distinct from the durable hypotheses table.
--
-- HypothesisGenerator rebuilds this tree every turn: depth 0 is always
-- 3-5 candidate "intents" for why the student sent their last message;
-- deeper depths condition on their parent and get progressively more
-- specific, stopping wherever a branch stops being worth expanding
-- (see should_expand_branch in hypothesis_generator.py). Every branch
-- — leaf or not — carries its own predicted_next_turn, so whatever
-- depth it stops at, it stays checkable against the student's real
-- next message.
--
-- Append-only, same pattern as world_model_revisions and every other
-- store in this project (CLAUDE.md invariant 6): no delete method, no
-- DELETE SQL, status moves open -> matched/unmatched/superseded via
-- UPDATE only.
--
-- This is deliberately NOT wired into hypotheses/evidence_refs in this
-- migration: a branch match does not write into HypothesisStore. See
-- CLAUDE.md invariant 6's rationale.

CREATE TYPE branch_status AS ENUM ('open', 'matched', 'unmatched', 'superseded');

CREATE TABLE branch_generations (
    id          UUID PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES sessions (id) ON DELETE RESTRICT,
    turn_index  INT NOT NULL,
    root_count  INT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_branch_generations_session
    ON branch_generations (session_id, turn_index);

CREATE TABLE branches (
    id                    UUID PRIMARY KEY,
    parent_id             UUID REFERENCES branches (id) ON DELETE RESTRICT,
    generation_id         UUID NOT NULL REFERENCES branch_generations (id) ON DELETE RESTRICT,
    session_id            UUID NOT NULL REFERENCES sessions (id) ON DELETE RESTRICT,
    turn_index            INT NOT NULL,
    depth                 INT NOT NULL,
    depth_label           TEXT NOT NULL,
    statement             TEXT NOT NULL,
    predicted_next_turn   TEXT NOT NULL,
    plausibility          DOUBLE PRECISION NOT NULL
                          CHECK (plausibility >= 0.0 AND plausibility <= 1.0),
    is_leaf               BOOLEAN NOT NULL,
    status                branch_status NOT NULL DEFAULT 'open',
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_branches_generation_status_leaf
    ON branches (generation_id, status, is_leaf);
CREATE INDEX idx_branches_parent ON branches (parent_id);
