-- probe: branch_generations gains selected_branch_id / selection_rationale
-- / path_requirement — SelectBranch picks one branch from the just-
-- generated tree (by coverage, not raw plausibility) and DerivePath
-- turns its full root-to-leaf path into a PathRequirement that scopes
-- Teach. Both are written after create_generation()/add_branches()
-- already ran, via UPDATE, not at insert time — same append-friendly
-- pattern as retier()/approve() elsewhere in this codebase.
--
-- selected_branch_id references branches(id): branches.generation_id
-- already references branch_generations(id), so this is a circular
-- FK relationship between the two tables. Not a problem in practice —
-- selected_branch_id starts NULL and is only ever set via UPDATE once
-- both the generation row and its branches already exist, so no
-- insert-order conflict ever arises.
--
-- Selecting a branch is not a commitment: it stays selected on this
-- generation's row for display/audit, but nothing about branch
-- tiering, resolution, or supersession changes because of a
-- selection — an unselected branch is deferred, not discarded.

ALTER TABLE branch_generations
    ADD COLUMN selected_branch_id UUID REFERENCES branches(id),
    ADD COLUMN selection_rationale TEXT,
    ADD COLUMN path_requirement JSONB;
