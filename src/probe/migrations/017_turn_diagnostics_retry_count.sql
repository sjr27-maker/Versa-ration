-- probe: turn_diagnostics gains retry_count — this turn's total across
-- every LLMClient retry (GeminiLLMClient's own retry loop, not the
-- SDK's opaque internal one; see llm.py). A rate-limited call and a
-- genuinely slow one are otherwise indistinguishable from outside the
-- LLMClient boundary; this makes throttling visible in the web UI's
-- Diagnostics panel instead of only in logs.
--
-- NOT NULL DEFAULT 0: every turn has a defined retry count (0 against
-- StubLLMClient, which has no retry mechanism, and 0 for any turn that
-- didn't hit a retryable transport error), same convention as
-- total_call_count.

ALTER TABLE turn_diagnostics
    ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0;
