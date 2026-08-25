-- probe: sessions.concept_graph_id — the topic/graph this session is
-- teaching, set once at creation via `probe chat --topic`, same as
-- learner_id. ON DELETE RESTRICT: a graph can't be removed out from
-- under sessions that reference it (ConceptGraph has no delete method
-- anyway, so this should never be reachable in practice — belt and
-- suspenders, same as every other FK in this schema).
--
-- Backfill: the dev DB's pre-existing session(s) predate concept graphs
-- entirely. They're attached to the same 'legacy-ungrouped' placeholder
-- graph the pre-existing concept_nodes were backfilled onto in the
-- previous migration, rather than a second, disconnected placeholder —
-- reused by topic lookup if it's already there, created fresh
-- otherwise (a session-only backfill with no matching concept_nodes
-- backfill is possible in principle, e.g. a from-scratch dev DB that
-- only ever ran `probe chat` before this feature existed).

ALTER TABLE sessions ADD COLUMN concept_graph_id UUID;

DO $$
DECLARE
    placeholder_id UUID;
BEGIN
    IF EXISTS (SELECT 1 FROM sessions WHERE concept_graph_id IS NULL) THEN
        SELECT id INTO placeholder_id FROM concept_graphs
            WHERE topic = 'legacy-ungrouped'
            LIMIT 1;
        IF placeholder_id IS NULL THEN
            INSERT INTO concept_graphs (id, topic)
            VALUES (gen_random_uuid(), 'legacy-ungrouped')
            RETURNING id INTO placeholder_id;
        END IF;

        UPDATE sessions SET concept_graph_id = placeholder_id
        WHERE concept_graph_id IS NULL;
    END IF;
END $$;

ALTER TABLE sessions ALTER COLUMN concept_graph_id SET NOT NULL;
ALTER TABLE sessions
    ADD CONSTRAINT sessions_concept_graph_id_fkey
    FOREIGN KEY (concept_graph_id)
    REFERENCES concept_graphs (id)
    ON DELETE RESTRICT;

CREATE INDEX idx_sessions_concept_graph ON sessions (concept_graph_id);
