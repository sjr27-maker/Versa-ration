-- probe: turn_diagnostics gains prior_reference_detected/unaddressed --
-- nodes.detect_prior_reference's per-turn judgment on whether the
-- student's message plausibly referenced something already established
-- this session (a prior example, analogy, or explanation), and whether
-- nodes.check_prior_reference_unaddressed flagged Teach's output as
-- failing to actually name what was referenced (the example/analogy
-- tracked from the immediately preceding turn's ExtractTeachingArtifact
-- call is absent from the response). See loop.py's handle_turn: Teach
-- now receives a compact recent-history block plus a structured list of
-- examples/analogies already used this session, so it can build on or
-- reuse an established one instead of introducing an unrelated new one
-- when the student refers back to prior work.

ALTER TABLE turn_diagnostics
    ADD COLUMN prior_reference_detected BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN prior_reference_unaddressed BOOLEAN NOT NULL DEFAULT FALSE;
