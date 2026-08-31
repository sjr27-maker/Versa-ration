-- probe: evidence_records -- a durable, labeled log of verification
-- findings. Each row records what a check found, and CRUCIALLY under
-- what circumstances it was produced: `source_type` distinguishes a
-- staged mechanism test (deliberate scripted sessions run to exercise
-- code paths) from evidence gathered during a real, organic student
-- session. The two must never be read as the same kind of proof -- a
-- staged run can show a mechanism functions; only real elapsed usage
-- can support a claim that the system adapted to a student.
--
-- `body` is free-form JSONB: the structured verbatim evidence for one
-- finding (branches offered, facts written, turn_diagnostics fields,
-- responses, consolidation output, etc.). `summary` is the one-line
-- human read, and for staged runs must use "mechanism verified"
-- language, never "adapted to the student" -- that discipline is
-- enforced by the writer, not the schema, but this column exists so
-- the distinction is visible at a glance on the Evidence page.
--
-- Append-only, same discipline as every other store in this project
-- (CLAUDE.md invariant 11): no delete method, no DELETE SQL. A finding,
-- once recorded, is a historical fact about what a check saw at that
-- moment -- superseded findings are added alongside, never removed.

CREATE TYPE evidence_source_type AS ENUM ('staged_verification', 'organic_session');

CREATE TABLE evidence_records (
    id           UUID PRIMARY KEY,
    source_type  evidence_source_type NOT NULL,
    part         TEXT NOT NULL,
    title        TEXT NOT NULL,
    summary      TEXT NOT NULL,
    body         JSONB NOT NULL,
    learner_id   UUID REFERENCES learners (id) ON DELETE RESTRICT,
    session_id   UUID REFERENCES sessions (id) ON DELETE RESTRICT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_evidence_records_source
    ON evidence_records (source_type, created_at);
CREATE INDEX idx_evidence_records_part
    ON evidence_records (part, created_at);
