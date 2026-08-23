-- probe: initial schema for the hypothesis store.
--
-- Append-only by design: no cascading deletes, no soft-delete columns.
-- Retirement lives in the `tier` enum ('archived'); resurrection just
-- flips it back to 'active'.

CREATE TYPE hypothesis_layer AS ENUM (
    'goal',
    'knowledge',
    'mental_model',
    'cognitive_state',
    'teaching'
);

CREATE TYPE hypothesis_tier AS ENUM (
    'active',
    'background',
    'dormant',
    'archived'
);

CREATE TYPE evidence_polarity AS ENUM (
    'supporting',
    'contradicting'
);

CREATE TABLE hypotheses (
    id           UUID PRIMARY KEY,
    layer        hypothesis_layer NOT NULL,
    statement    TEXT NOT NULL,
    probability  DOUBLE PRECISION NOT NULL
                 CHECK (probability >= 0.0 AND probability <= 1.0),
    confidence   DOUBLE PRECISION NOT NULL
                 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    tier         hypothesis_tier NOT NULL,
    conditions   TEXT[] NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_hypotheses_layer_tier ON hypotheses (layer, tier);

CREATE TABLE evidence_refs (
    id             UUID PRIMARY KEY,
    hypothesis_id  UUID NOT NULL REFERENCES hypotheses (id),
    turn_id        UUID NOT NULL,
    polarity       evidence_polarity NOT NULL,
    timestamp      TIMESTAMPTZ NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_evidence_refs_hypothesis
    ON evidence_refs (hypothesis_id, timestamp);
