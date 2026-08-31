-- probe: the memory layer built on top of minimal_branch
-- (ReasoningMode.DISAMBIGUATE) -- see memory.py's module docstring.
-- Additive: DisambiguationStore's own tables (migration 029) are
-- untouched. This is a derived, searchable layer on top of them (and
-- of the plain student turns), not a replacement.
--
-- `CREATE EXTENSION IF NOT EXISTS vector` first, so a from-empty-
-- schema replay (a fresh environment, or the test suite's own
-- drop-and-replay-all-migrations fixture) self-documents and picks
-- this up automatically rather than needing a manual step first.
--
-- 768-dimensional embeddings (see embeddings.py's EMBEDDING_DIM):
-- gemini-embedding-001 is a Matryoshka-trained model natively
-- producing 3072 dimensions, but this pgvector build (0.8.6) caps
-- HNSW-indexable vectors at 2000 dimensions -- verified live, a
-- vector(3072) column rejects `CREATE INDEX ... USING hnsw` outright.
-- 768 is comfortably under that cap and still a strong embedding size.
--
-- HNSW over IVFFlat: IVFFlat's list-count parameter needs tuning
-- against the table's eventual row count and performs poorly while a
-- table is small/empty -- exactly this feature's own starting
-- condition (a few facts per learner, growing slowly). HNSW has no
-- such "needs enough rows first" requirement and is pgvector's own
-- current general recommendation.
--
-- Append-only, same discipline as every other store in this project
-- (CLAUDE.md invariant 10): no delete method, no DELETE SQL.
-- `learner_facts` never updates a row once written -- a fact is a
-- historical record of what was true at that turn. Retirement of a
-- `thinking_style_candidates` row is a status transition (candidate ->
-- confirmed/retired) via UPDATE, same resurrection-over-deletion
-- principle as HypothesisStore's tier transitions.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TYPE learner_fact_type AS ENUM ('branch_resolution', 'direct_answer');

CREATE TABLE learner_facts (
    id               UUID PRIMARY KEY,
    learner_id       UUID NOT NULL REFERENCES learners (id) ON DELETE RESTRICT,
    session_id       UUID NOT NULL REFERENCES sessions (id) ON DELETE RESTRICT,
    turn_index       INT NOT NULL,
    fact_type        learner_fact_type NOT NULL,
    situation        TEXT NOT NULL,
    resolution       TEXT NOT NULL,
    embedding        vector(768) NOT NULL,
    source_turn_id   UUID NOT NULL REFERENCES turns (id) ON DELETE RESTRICT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_learner_facts_learner ON learner_facts (learner_id, created_at);
CREATE INDEX idx_learner_facts_embedding_hnsw
    ON learner_facts USING hnsw (embedding vector_cosine_ops);

CREATE TYPE thinking_style_status AS ENUM ('candidate', 'confirmed', 'retired');

CREATE TABLE thinking_style_candidates (
    id                      UUID PRIMARY KEY,
    learner_id              UUID NOT NULL REFERENCES learners (id) ON DELETE RESTRICT,
    session_ids             UUID[] NOT NULL,
    path_summary            TEXT NOT NULL,
    -- Not in the original schema list (which named only path_summary)
    -- -- added because step 7's "semantic search on path_summary" is
    -- not possible without also storing its embedding. Same category
    -- of addition as learner_facts.embedding, just for this table.
    path_summary_embedding  vector(768) NOT NULL,
    confirmation_count      INT NOT NULL DEFAULT 1,
    status                  thinking_style_status NOT NULL DEFAULT 'candidate',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_thinking_style_candidates_learner
    ON thinking_style_candidates (learner_id, status);
CREATE INDEX idx_thinking_style_candidates_embedding_hnsw
    ON thinking_style_candidates USING hnsw (path_summary_embedding vector_cosine_ops);
