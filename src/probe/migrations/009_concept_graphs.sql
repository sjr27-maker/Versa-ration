-- probe: concept_graphs — group concept_nodes by seed-graph run/topic.
--
-- Previously every `probe seed-graph` run dumped its nodes into one
-- single flat concept_nodes table with nothing to distinguish "the
-- graph for topic A" from "the graph for topic B". This table is that
-- grouping. concept_id (concept_nodes.id) becomes unique only WITHIN a
-- graph, not globally, as of the next migration.
--
-- No delete method on ConceptGraph for graphs either — same append-only
-- reasoning as CLAUDE.md invariant 4: sessions may reference a graph
-- indefinitely, so removing one out from under them isn't safe.
--
-- `topic` is deliberately not unique: re-running seed-graph on the same
-- topic string creates a second, independent graph rather than erroring
-- or silently merging — the CLI decides what to do about that ambiguity
-- at resolve time, not this table.

CREATE TABLE concept_graphs (
    id          UUID PRIMARY KEY,
    topic       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_concept_graphs_topic ON concept_graphs (topic);
