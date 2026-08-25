-- probe: world concept graph + learner overlay.
--
-- concept_nodes/concept_prerequisites are the world graph: seeded once
-- (see `probe seed-graph`) and frozen — ConceptGraph has no delete
-- method, matching the append-only spirit of CLAUDE.md invariant 1,
-- because LearnerOverlay (and future consumers) reference concept ids
-- and a removed concept would break that FK silently if we let it
-- cascade. ON DELETE RESTRICT everywhere, never CASCADE, same reasoning
-- as migration 003.
--
-- Prerequisites are a proper edge table, not an array column on
-- concept_nodes: prerequisites_of/all_prerequisites_of need graph
-- traversal, which an array can't support without unpacking it anyway.
--
-- learner_overlay is different in kind: it's *current-state* tracking,
-- not a claim-with-evidence trail, so set_state upserts rather than
-- appending. It is NOT a Hypothesis and must not be routed through
-- HypothesisStore.

CREATE TYPE overlay_state AS ENUM (
    'known',
    'partial',
    'unknown',
    'blocked'
);

CREATE TABLE concept_nodes (
    id                     TEXT PRIMARY KEY,
    name                   TEXT NOT NULL,
    common_misconceptions  TEXT[] NOT NULL DEFAULT '{}',
    representations        TEXT[] NOT NULL DEFAULT '{}',
    diagnostic_questions   TEXT[] NOT NULL DEFAULT '{}',
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE concept_prerequisites (
    concept_id       TEXT NOT NULL REFERENCES concept_nodes (id) ON DELETE RESTRICT,
    prerequisite_id  TEXT NOT NULL REFERENCES concept_nodes (id) ON DELETE RESTRICT,
    PRIMARY KEY (concept_id, prerequisite_id)
);

CREATE INDEX idx_concept_prerequisites_concept
    ON concept_prerequisites (concept_id);
CREATE INDEX idx_concept_prerequisites_prereq
    ON concept_prerequisites (prerequisite_id);

CREATE TABLE learner_overlay (
    learner_id   UUID NOT NULL,
    concept_id   TEXT NOT NULL REFERENCES concept_nodes (id) ON DELETE RESTRICT,
    state        overlay_state NOT NULL,
    confidence   DOUBLE PRECISION NOT NULL
                 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (learner_id, concept_id)
);

CREATE INDEX idx_learner_overlay_learner ON learner_overlay (learner_id);
