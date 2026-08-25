-- probe: sessions.learner_id — a session's identity, set once at
-- creation, not per-turn state.
--
-- ON DELETE RESTRICT (never CASCADE), same reasoning as every other FK
-- in this schema: sessions are meant to be append-only (see
-- TranscriptStore's docstring), so it must be impossible to remove a
-- learner a session still points to.
--
-- Backfill: the dev DB had pre-existing session row(s) from before
-- learner identity existed. Rather than deleting them (sessions/turns
-- are append-only by the same convention as the rest of this schema)
-- or leaving learner_id nullable (which would silently make "which
-- learner is this" optional everywhere downstream, defeating the point
-- of adding it), every such session is backfilled onto one placeholder
-- learner before the NOT NULL constraint is added. On a fresh schema
-- (the normal case — tests rebuild from scratch) there are zero
-- sessions and this is a no-op.

ALTER TABLE sessions ADD COLUMN learner_id UUID;

DO $$
DECLARE
    placeholder_id UUID;
BEGIN
    IF EXISTS (SELECT 1 FROM sessions WHERE learner_id IS NULL) THEN
        INSERT INTO learners (id, label)
        VALUES (gen_random_uuid(), 'legacy-unlabeled')
        RETURNING id INTO placeholder_id;

        UPDATE sessions SET learner_id = placeholder_id WHERE learner_id IS NULL;
    END IF;
END $$;

ALTER TABLE sessions
    ALTER COLUMN learner_id SET NOT NULL;

ALTER TABLE sessions
    ADD CONSTRAINT sessions_learner_id_fkey
    FOREIGN KEY (learner_id)
    REFERENCES learners (id)
    ON DELETE RESTRICT;

CREATE INDEX idx_sessions_learner ON sessions (learner_id);
