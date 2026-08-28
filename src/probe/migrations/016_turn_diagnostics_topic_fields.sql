-- probe: turn_diagnostics gains inferred_topic / topic_seeded_new —
-- AttachTopic's own result was previously only visible via a raw
-- node_calls query. A wrong topic inference needs to be immediately
-- visible on turn 0 in the web UI's Diagnostics panel, not something
-- discovered turns later because the lesson feels off.
--
-- Both nullable: only turn 0 ever runs AttachTopic (see
-- SessionLoop.handle_turn), so every other turn's row leaves these
-- null rather than repeating turn 0's value.

ALTER TABLE turn_diagnostics
    ADD COLUMN inferred_topic TEXT,
    ADD COLUMN topic_seeded_new BOOLEAN;
