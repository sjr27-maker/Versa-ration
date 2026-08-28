-- probe: two additions needed to show a per-turn hypothesis delta in
-- the web UI ("what rose, what fell, what resurrected from dormant")
-- without the UI computing anything itself.
--
-- 1. evidence_refs gains the *resulting* probability/confidence from
--    the reweight() call that created it — the "after" value. Nullable
--    because existing evidence predates this and has no way to recover
--    it; same honesty as list_by_learner's documented
--    zero-evidence-is-unattributable gap. Going forward, reweight()
--    always writes both.
--
-- 2. hypothesis_tier_changes is a new, tiny append-only log — tier
--    transitions (retier()/resurrect()) currently leave no trace at
--    all, so "resurrected from dormant" has nothing to read. No delete
--    method, no DELETE SQL, same append-only pattern as every other
--    store in this project (CLAUDE.md invariant 1's rule extended
--    here, not a new numbered invariant — this is a natural extension
--    of hypotheses' own audit trail, not a separate concern the way
--    branches/revisions were).

ALTER TABLE evidence_refs
    ADD COLUMN resulting_probability DOUBLE PRECISION
        CHECK (resulting_probability IS NULL
               OR (resulting_probability >= 0.0 AND resulting_probability <= 1.0)),
    ADD COLUMN resulting_confidence DOUBLE PRECISION
        CHECK (resulting_confidence IS NULL
               OR (resulting_confidence >= 0.0 AND resulting_confidence <= 1.0));

CREATE TABLE hypothesis_tier_changes (
    id             UUID PRIMARY KEY,
    hypothesis_id  UUID NOT NULL REFERENCES hypotheses (id) ON DELETE RESTRICT,
    old_tier       hypothesis_tier NOT NULL,
    new_tier       hypothesis_tier NOT NULL,
    changed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_hypothesis_tier_changes_hypothesis
    ON hypothesis_tier_changes (hypothesis_id, changed_at);
