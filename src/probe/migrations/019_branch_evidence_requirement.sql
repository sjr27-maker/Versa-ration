-- probe: branches gain requires_evidence / evidence_satisfied — a
-- branch becomes a claim with an entry condition, not just a forecast.
-- Written by the same GENERATE:INTENT/GENERATE:EXPAND call that
-- creates the branch (see hypothesis_generator.py); null means the
-- branch needs nothing further and expands on plausibility alone as
-- before. should_expand_branch's fourth gate blocks expansion of a
-- branch with requires_evidence set and evidence_satisfied still
-- false — held at its current depth, not pruned or superseded.
--
-- evidence_satisfied flips true via a direct option click (see the
-- new options table, migration 020) or a typed message CheckEvidence
-- judges to establish it. It never resets once true.

ALTER TABLE branches
    ADD COLUMN requires_evidence TEXT,
    ADD COLUMN evidence_satisfied BOOLEAN NOT NULL DEFAULT FALSE;
