-- probe: turn_diagnostics gains the memory layer's own visibility
-- columns (CLAUDE.md invariant 7 already covers this table's
-- append-only-ness; this just adds more nullable-with-default columns
-- to it, same pattern as migrations 021/023/027/028).
--
-- Required by the memory layer's own design constraint: a turn where
-- branching was skipped because a past fact resolved it must be
-- visible and auditable per turn, never a silent shortcut.

ALTER TABLE turn_diagnostics
    ADD COLUMN memory_match_found BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN memory_match_confirmed_resolution BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN branching_skipped_by_memory BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN matched_fact_id UUID REFERENCES learner_facts (id) ON DELETE RESTRICT;
