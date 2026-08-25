-- probe: hypothesis <-> concept linkage.
--
-- Hypothesis (migration 001) is deliberately left unchanged — this is a
-- separate join table rather than a concept_id column on hypotheses,
-- because a hypothesis can pertain to zero, one, or several concepts,
-- and because it keeps no migration risk on the already-shipped,
-- already-tested Hypothesis model.
--
-- Append-only, same spirit as the rest of the reasoning-state schema:
-- asserting a link is idempotent (ON CONFLICT DO NOTHING in
-- HypothesisStore.link_concept), and there is no unlink method.

CREATE TABLE hypothesis_concepts (
    hypothesis_id  UUID NOT NULL REFERENCES hypotheses (id) ON DELETE RESTRICT,
    concept_id     TEXT NOT NULL REFERENCES concept_nodes (id) ON DELETE RESTRICT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (hypothesis_id, concept_id)
);

CREATE INDEX idx_hypothesis_concepts_concept
    ON hypothesis_concepts (concept_id);
