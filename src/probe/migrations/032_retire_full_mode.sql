-- probe: retire the full Diagnose/Infer/Update/Replan/Plan reasoning
-- path and the original tree-based branch system, leaving minimal_branch
-- (ReasoningMode.DISAMBIGUATE, now the default) and the plain-LLM
-- BASELINE as the only two architectures a session can run under.
--
-- This is an additive drop-migration on purpose (see the deletion
-- pass's design notes): migrations 001-031 are left byte-for-byte
-- intact so the replay chain stays linear and every historical ALTER
-- still applies to a table that exists at that point in the sequence.
-- Everything the retired subsystems created is torn down here, at the
-- end, in one place.
--
-- DELIBERATELY KEPT (still load-bearing for migration 029's
-- disambiguation tables and memory.py):
--   * the `branch_status` enum type (created in 012) -
--     disambiguation_branches.status is of this type
--   * the `option_status` enum type (created in 020) -
--     disambiguation_options.status is of this type
--   * `learners`, `sessions`, `turns`, `node_calls`, `turn_diagnostics`
--     (trimmed below), `disambiguation_*`, `learner_facts`,
--     `thinking_style_candidates`
--
-- Append-only note: CLAUDE.md invariants 1/4/5/6/7/8 forbade DELETE SQL
-- *in those stores' own modules and migrations* while those stores
-- existed. The stores and their invariants are being retired together;
-- this DROP is schema teardown of a removed subsystem, not a store
-- mutating its own rows.

-- --- original tree-based branch system (012, 018, 019, 022) ----------
DROP TABLE IF EXISTS options CASCADE;
DROP TABLE IF EXISTS branches CASCADE;
DROP TABLE IF EXISTS branch_generations CASCADE;

-- --- world-model revisions (006) -----------------------------------
DROP TABLE IF EXISTS world_model_revision_evidence CASCADE;
DROP TABLE IF EXISTS world_model_revisions CASCADE;
DROP TYPE  IF EXISTS revision_status;

-- --- concept graph + learner overlay (004, 005, 009, 010) ----------
DROP TABLE IF EXISTS learner_overlay CASCADE;
DROP TABLE IF EXISTS hypothesis_concepts CASCADE;
DROP TABLE IF EXISTS concept_prerequisites CASCADE;
DROP TABLE IF EXISTS concept_nodes CASCADE;
DROP TABLE IF EXISTS concept_graphs CASCADE;
DROP TYPE  IF EXISTS overlay_state;

-- --- hypothesis store (001, 003, 005, 014) ------------------------
DROP TABLE IF EXISTS hypothesis_tier_changes CASCADE;
DROP TABLE IF EXISTS evidence_refs CASCADE;
DROP TABLE IF EXISTS hypotheses CASCADE;
DROP TYPE  IF EXISTS hypothesis_layer;
DROP TYPE  IF EXISTS hypothesis_tier;
DROP TYPE  IF EXISTS evidence_polarity;

-- --- sessions.concept_graph_id (011, 013) -------------------------
-- minimal_branch and BASELINE have no concept graph at all.
ALTER TABLE sessions DROP COLUMN IF EXISTS concept_graph_id;

-- --- turn_diagnostics: drop the full-path-only columns ------------
-- (016, 021, 023, 024, 027, 028). The disambiguation path never wrote
-- any of these; every column kept below is one _record_disambiguation_
-- diagnostics / _handle_bypass_turn actually sets.
ALTER TABLE turn_diagnostics
    DROP COLUMN IF EXISTS inferred_topic,
    DROP COLUMN IF EXISTS topic_seeded_new,
    DROP COLUMN IF EXISTS options_missed,
    DROP COLUMN IF EXISTS current_belief_unsupported,
    DROP COLUMN IF EXISTS generation_skipped_reason,
    DROP COLUMN IF EXISTS explicit_request_present,
    DROP COLUMN IF EXISTS explicit_request_what,
    DROP COLUMN IF EXISTS explicit_request_unaddressed,
    DROP COLUMN IF EXISTS prior_reference_detected,
    DROP COLUMN IF EXISTS prior_reference_unaddressed;
