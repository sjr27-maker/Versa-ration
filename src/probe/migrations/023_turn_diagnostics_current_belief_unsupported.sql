-- probe: turn_diagnostics gains current_belief_unsupported -- a
-- structural backstop flag (hypothesis_generator.check_current_belief_leak)
-- for the exact failure where DerivePath promotes a branch's
-- predicted_next_turn (a hypothetical FUTURE reaction to something not
-- yet taught) into current_belief (a stated fact about the student's
-- EXISTING belief), which Teach then unwittingly affirms as something
-- the student already said. Prompts drift; this check catches what a
-- prompt misses and surfaces it for human review rather than silently
-- trusting DerivePath's output.

ALTER TABLE turn_diagnostics
    ADD COLUMN current_belief_unsupported BOOLEAN NOT NULL DEFAULT FALSE;
