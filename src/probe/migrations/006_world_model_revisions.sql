-- probe: world-model revisions.
--
-- A WorldModelRevision is a claim about the concept graph, evidence-backed
-- the same way a Hypothesis is a claim about the learner — but it is
-- never auto-applied. `status` starts 'pending' and only a human review
-- (`probe review-revisions`, WorldModelRevisionStore.approve/reject)
-- moves it to 'approved' or 'rejected'. `applied_field_updates` records
-- the exact structured edit a human confirmed at approval time — it is
-- NOT derived automatically from `proposed_change` (free text).
--
-- No delete method on WorldModelRevisionStore, no DELETE SQL here:
-- status transitions are UPDATEs, same pattern as HypothesisStore's
-- reweight/retier (CLAUDE.md invariant 1's "no delete" applies to
-- removal, not mutation).

CREATE TYPE revision_status AS ENUM (
    'pending',
    'approved',
    'rejected'
);

CREATE TABLE world_model_revisions (
    id                     UUID PRIMARY KEY,
    concept_id             TEXT NOT NULL REFERENCES concept_nodes (id) ON DELETE RESTRICT,
    proposed_change        TEXT NOT NULL,
    confidence             DOUBLE PRECISION NOT NULL
                           CHECK (confidence >= 0.0 AND confidence <= 1.0),
    status                 revision_status NOT NULL DEFAULT 'pending',
    applied_field_updates  JSONB,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at            TIMESTAMPTZ
);

CREATE INDEX idx_world_model_revisions_concept_status
    ON world_model_revisions (concept_id, status);

-- Reuses evidence_polarity (migration 001) so a revision's evidence
-- looks structurally like a hypothesis's, but lives in its own table:
-- evidence_refs.hypothesis_id is NOT NULL and specific to Hypothesis,
-- so a revision's evidence is not shoehorned in there.
CREATE TABLE world_model_revision_evidence (
    id           UUID PRIMARY KEY,
    revision_id  UUID NOT NULL REFERENCES world_model_revisions (id) ON DELETE RESTRICT,
    turn_id      UUID NOT NULL REFERENCES turns (id) ON DELETE RESTRICT,
    polarity     evidence_polarity NOT NULL,
    timestamp    TIMESTAMPTZ NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_world_model_revision_evidence_revision
    ON world_model_revision_evidence (revision_id);
