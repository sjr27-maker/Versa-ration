-- probe: node_calls gains seq -- a DB-assigned, strictly monotonic
-- ordering column. `timestamp` alone is not a reliable ordering key:
-- under fast (stub-client) execution, several calls within the same
-- turn can land on the same wall-clock microsecond (observed directly
-- -- Update/Replan, SelectBranch/BranchGenerate, and Teach/DerivePath
-- all tied within one turn in a real test run), and ORDER BY timestamp
-- across a tie is not guaranteed to reflect actual call order. seq is
-- assigned by Postgres itself (BIGSERIAL), so it is exactly as
-- monotonic as insertion order regardless of clock resolution -- the
-- correctness invariant 2's audit trail actually needs.

ALTER TABLE node_calls ADD COLUMN seq BIGSERIAL;
