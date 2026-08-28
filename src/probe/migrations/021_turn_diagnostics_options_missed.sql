-- probe: turn_diagnostics gains options_missed — true when options
-- were on offer from the prior turn and the student typed past them
-- without satisfying any live branch's requires_evidence either. This
-- is a signal about the branch/option set being wrong, not about the
-- student — fed into the next turn's generation context and surfaced
-- prominently in the web UI's Diagnostics panel.

ALTER TABLE turn_diagnostics
    ADD COLUMN options_missed BOOLEAN NOT NULL DEFAULT FALSE;
