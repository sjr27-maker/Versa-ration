-- probe: scope concept_nodes and everything that references it to a
-- concept_graph.
--
-- concept_id (concept_nodes.id) is only unique WITHIN a graph now that
-- a deployment can seed multiple topics — composite (concept_graph_id,
-- id) is the real key everywhere a plain concept_id used to be enough
-- on its own: concept_prerequisites, hypothesis_concepts,
-- learner_overlay, world_model_revisions.
--
-- Backfill: any pre-existing rows in these tables predate concept_graphs
-- and get moved onto one placeholder graph (topic 'legacy-ungrouped')
-- rather than dropped or left dangling — same reasoning as the learner
-- backfill in migration 008. On a fresh schema (tests rebuild from
-- scratch) every one of these tables is empty and this is a no-op.
--
-- hypothesis_concepts/learner_overlay/world_model_revisions don't get a
-- direct FK to concept_graphs: their composite FK to concept_nodes
-- (concept_graph_id, id) already transitively requires concept_graph_id
-- to be a real graph, since concept_nodes.concept_graph_id itself is
-- FK'd to concept_graphs. Only concept_nodes needs that FK directly.

ALTER TABLE concept_nodes ADD COLUMN concept_graph_id UUID;
ALTER TABLE concept_prerequisites ADD COLUMN concept_graph_id UUID;
ALTER TABLE hypothesis_concepts ADD COLUMN concept_graph_id UUID;
ALTER TABLE learner_overlay ADD COLUMN concept_graph_id UUID;
ALTER TABLE world_model_revisions ADD COLUMN concept_graph_id UUID;

DO $$
DECLARE
    placeholder_id UUID;
BEGIN
    IF EXISTS (SELECT 1 FROM concept_nodes WHERE concept_graph_id IS NULL) THEN
        INSERT INTO concept_graphs (id, topic)
        VALUES (gen_random_uuid(), 'legacy-ungrouped')
        RETURNING id INTO placeholder_id;

        UPDATE concept_nodes
            SET concept_graph_id = placeholder_id
            WHERE concept_graph_id IS NULL;
        UPDATE concept_prerequisites
            SET concept_graph_id = placeholder_id
            WHERE concept_graph_id IS NULL;
        UPDATE hypothesis_concepts
            SET concept_graph_id = placeholder_id
            WHERE concept_graph_id IS NULL;
        UPDATE learner_overlay
            SET concept_graph_id = placeholder_id
            WHERE concept_graph_id IS NULL;
        UPDATE world_model_revisions
            SET concept_graph_id = placeholder_id
            WHERE concept_graph_id IS NULL;
    END IF;
END $$;

ALTER TABLE concept_nodes ALTER COLUMN concept_graph_id SET NOT NULL;
ALTER TABLE concept_prerequisites ALTER COLUMN concept_graph_id SET NOT NULL;
ALTER TABLE hypothesis_concepts ALTER COLUMN concept_graph_id SET NOT NULL;
ALTER TABLE learner_overlay ALTER COLUMN concept_graph_id SET NOT NULL;
ALTER TABLE world_model_revisions ALTER COLUMN concept_graph_id SET NOT NULL;

-- Drop every old FK/PK that pointed at concept_nodes(id) alone, before
-- re-keying concept_nodes onto the composite primary key.
ALTER TABLE concept_prerequisites DROP CONSTRAINT concept_prerequisites_concept_id_fkey;
ALTER TABLE concept_prerequisites DROP CONSTRAINT concept_prerequisites_prerequisite_id_fkey;
ALTER TABLE hypothesis_concepts DROP CONSTRAINT hypothesis_concepts_concept_id_fkey;
ALTER TABLE learner_overlay DROP CONSTRAINT learner_overlay_concept_id_fkey;
ALTER TABLE world_model_revisions DROP CONSTRAINT world_model_revisions_concept_id_fkey;
ALTER TABLE concept_prerequisites DROP CONSTRAINT concept_prerequisites_pkey;
ALTER TABLE hypothesis_concepts DROP CONSTRAINT hypothesis_concepts_pkey;
ALTER TABLE learner_overlay DROP CONSTRAINT learner_overlay_pkey;
ALTER TABLE concept_nodes DROP CONSTRAINT concept_nodes_pkey;

ALTER TABLE concept_nodes
    ADD CONSTRAINT concept_nodes_pkey PRIMARY KEY (concept_graph_id, id);
ALTER TABLE concept_nodes
    ADD CONSTRAINT concept_nodes_graph_fkey
    FOREIGN KEY (concept_graph_id) REFERENCES concept_graphs (id) ON DELETE RESTRICT;

ALTER TABLE concept_prerequisites
    ADD CONSTRAINT concept_prerequisites_concept_fkey
    FOREIGN KEY (concept_graph_id, concept_id)
    REFERENCES concept_nodes (concept_graph_id, id) ON DELETE RESTRICT;
ALTER TABLE concept_prerequisites
    ADD CONSTRAINT concept_prerequisites_prerequisite_fkey
    FOREIGN KEY (concept_graph_id, prerequisite_id)
    REFERENCES concept_nodes (concept_graph_id, id) ON DELETE RESTRICT;
ALTER TABLE concept_prerequisites
    ADD CONSTRAINT concept_prerequisites_pkey
    PRIMARY KEY (concept_graph_id, concept_id, prerequisite_id);

ALTER TABLE hypothesis_concepts
    ADD CONSTRAINT hypothesis_concepts_concept_fkey
    FOREIGN KEY (concept_graph_id, concept_id)
    REFERENCES concept_nodes (concept_graph_id, id) ON DELETE RESTRICT;
ALTER TABLE hypothesis_concepts
    ADD CONSTRAINT hypothesis_concepts_pkey
    PRIMARY KEY (hypothesis_id, concept_graph_id, concept_id);

ALTER TABLE learner_overlay
    ADD CONSTRAINT learner_overlay_concept_fkey
    FOREIGN KEY (concept_graph_id, concept_id)
    REFERENCES concept_nodes (concept_graph_id, id) ON DELETE RESTRICT;
ALTER TABLE learner_overlay
    ADD CONSTRAINT learner_overlay_pkey
    PRIMARY KEY (learner_id, concept_graph_id, concept_id);

ALTER TABLE world_model_revisions
    ADD CONSTRAINT world_model_revisions_concept_fkey
    FOREIGN KEY (concept_graph_id, concept_id)
    REFERENCES concept_nodes (concept_graph_id, id) ON DELETE RESTRICT;

CREATE INDEX idx_concept_nodes_graph ON concept_nodes (concept_graph_id);
CREATE INDEX idx_hypothesis_concepts_graph_concept
    ON hypothesis_concepts (concept_graph_id, concept_id);
CREATE INDEX idx_learner_overlay_graph_concept
    ON learner_overlay (concept_graph_id, concept_id);
CREATE INDEX idx_world_model_revisions_graph_concept_status
    ON world_model_revisions (concept_graph_id, concept_id, status);
