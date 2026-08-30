-- probe: turn_diagnostics gains explicit_request_present/what/
-- unaddressed -- ExtractRequest's per-turn judgment (nodes.py) on
-- whether the student's message contained a concrete, answerable
-- request, and whether nodes.check_explicit_request_unaddressed
-- flagged Teach's output as failing to actually resolve it (deferred
-- or never mentioned, rather than worked). See loop.py's handle_turn:
-- an explicit request takes precedence over Plan's chosen
-- target_concept and DerivePath's scope -- the pedagogical machinery
-- decides HOW to teach, never WHETHER to answer what was asked.

ALTER TABLE turn_diagnostics
    ADD COLUMN explicit_request_present BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN explicit_request_what TEXT NULL,
    ADD COLUMN explicit_request_unaddressed BOOLEAN NOT NULL DEFAULT FALSE;
