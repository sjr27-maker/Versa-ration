-- probe: turn_diagnostics — one row per handle_turn() call, closing
-- the gap between what's already computed in loop.py each turn (call
-- counts, the MAX_CALLS_PER_TURN guardrail, entropy_bits, warnings)
-- and what's actually queryable afterward. Before this, the guardrail
-- firing and warning conditions were only ever logged, never
-- persisted — the web UI's Diagnostics panel needs to read these, not
-- re-derive them, per this feature's "zero business logic in the UI"
-- constraint.
--
-- Append-only: no delete method, no DELETE SQL, same pattern as every
-- other store in this project (CLAUDE.md invariant 1's rule extended
-- here — a turn's diagnostics are a historical fact once recorded,
-- same as a node_calls row).
--
-- teach_failed is its own column, not folded into warnings: downstream
-- analysis (branch match rate, portrait stats, call-count aggregates)
-- needs to filter these turns out by column, not by string-matching a
-- JSONB warnings list.

CREATE TABLE turn_diagnostics (
    id                 UUID PRIMARY KEY,
    session_id         UUID NOT NULL REFERENCES sessions (id) ON DELETE RESTRICT,
    turn_index         INT NOT NULL,
    node_call_counts   JSONB NOT NULL DEFAULT '{}',
    total_call_count   INT NOT NULL DEFAULT 0,
    guardrail_fired    BOOLEAN NOT NULL DEFAULT FALSE,
    entropy_bits       DOUBLE PRECISION,
    duration_ms        DOUBLE PRECISION NOT NULL,
    warnings           JSONB NOT NULL DEFAULT '[]',
    teach_failed       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, turn_index)
);

CREATE INDEX idx_turn_diagnostics_session ON turn_diagnostics (session_id, turn_index);
