-- probe: learner identity.
--
-- Identity-level fields only, all nullable beyond id/created_at, with
-- no behavioral logic attached to any of them. What a learner knows,
-- believes, or is working toward is NOT tracked here — that lives in
-- HypothesisStore (evidence-backed claims) or LearnerOverlay (current
-- concept-mastery state). This table is just "which conversations
-- belong to the same person."

CREATE TABLE learners (
    id                  UUID PRIMARY KEY,
    label               TEXT,
    display_name        TEXT,
    preferred_register  TEXT,
    timezone            TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Unique only when a label is actually given: `probe chat --learner
-- <label>` resolves a label to an existing learner (resume) or creates
-- one (new), and that lookup only works if labels don't collide.
-- Unlabeled learners can coexist freely (NULL is never considered
-- equal to NULL by a unique index).
CREATE UNIQUE INDEX idx_learners_label ON learners (label) WHERE label IS NOT NULL;
