-- probe: sessions.concept_graph_id becomes nullable — topic inference
-- (AttachTopic node) now attaches it after the session starts, instead
-- of the CLI requiring --topic up front. SessionLoop.handle_turn hard-
-- fails on any turn past the first if it's still null by then; see
-- CLAUDE.md and loop.py for that invariant — it's enforced in code,
-- not the schema, since "past the first turn" isn't expressible as a
-- column constraint.
--
-- No backfill needed: relaxing NOT NULL never invalidates existing rows.

ALTER TABLE sessions ALTER COLUMN concept_graph_id DROP NOT NULL;
