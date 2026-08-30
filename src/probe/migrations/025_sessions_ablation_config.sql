-- probe: sessions gain ablation_config -- the AblationConfig a session
-- was created with (ablation.py), fixed for that session's lifetime.
-- Nullable: a NULL here means "created before this migration, or
-- created without an explicit config" and is interpreted in code
-- (TranscriptStore.get_ablation_config) as AblationConfig()'s full-
-- system default, exactly matching every existing session's actual
-- behavior -- no backfill needed for old rows to stay meaningful.

ALTER TABLE sessions ADD COLUMN ablation_config JSONB NULL;
