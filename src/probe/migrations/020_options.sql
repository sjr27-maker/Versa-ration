-- probe: options — the clickable-button evidence channel. Generated
-- by GenerateOptions from live branches with an unsatisfied
-- requires_evidence (migration 019), one option per branch by
-- construction (GenerateOptions rejects and regenerates any mapping
-- that isn't a clean 1:1). A click is an unambiguous fact about which
-- branch's evidence requirement is now true — no interpretation layer,
-- unlike free text.
--
-- Append-only, same pattern as branches/branch_generations (CLAUDE.md
-- invariant 6, and its own entry in CLAUDE.md for this table): no
-- delete method, no DELETE SQL, status moves open ->
-- selected/superseded via UPDATE only. Regenerated every turn from the
-- current branch set and superseded the same way branches are (see
-- HypothesisGenerator.resolve()) — never carried forward.

CREATE TYPE option_status AS ENUM ('open', 'selected', 'superseded');

CREATE TABLE options (
    id             UUID PRIMARY KEY,
    branch_id      UUID NOT NULL REFERENCES branches (id) ON DELETE RESTRICT,
    generation_id  UUID NOT NULL REFERENCES branch_generations (id) ON DELETE RESTRICT,
    session_id     UUID NOT NULL REFERENCES sessions (id) ON DELETE RESTRICT,
    turn_index     INT NOT NULL,
    text           TEXT NOT NULL,
    status         option_status NOT NULL DEFAULT 'open',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_options_generation_status ON options (generation_id, status);
CREATE INDEX idx_options_branch ON options (branch_id);
