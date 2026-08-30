-- probe: turn_diagnostics gains generation_skipped_reason -- turn 0 has
-- no accumulated evidence yet, so BranchGenerate (and everything that
-- consumes its output: SelectBranch, DerivePath, GenerateOptions) is
-- skipped entirely rather than generating an unspecific tree from a
-- single opening message. Recording *why* a turn has no branches lets
-- that be told apart from a turn where generation actually ran and
-- failed, which is recorded as a warning instead.

ALTER TABLE turn_diagnostics
    ADD COLUMN generation_skipped_reason TEXT NULL;
