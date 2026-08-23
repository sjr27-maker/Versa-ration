-- probe: close the loop on evidence_refs.turn_id.
--
-- Migration 001 declared turn_id as a bare UUID column because the
-- transcript didn't exist yet. Now that migration 002 has created the
-- turns table, wire up the FK.
--
-- ON DELETE RESTRICT (never CASCADE): the hypothesis store is
-- append-only (CLAUDE.md invariant 1). Cascading a hypothetical turn
-- deletion into evidence_refs would silently destroy the audit trail;
-- RESTRICT makes any future attempt to remove a referenced turn fail
-- loudly instead. There is no path today that deletes a turn, and this
-- constraint is here so there's still no such path by accident.

ALTER TABLE evidence_refs
    ADD CONSTRAINT evidence_refs_turn_id_fkey
    FOREIGN KEY (turn_id)
    REFERENCES turns (id)
    ON DELETE RESTRICT;
