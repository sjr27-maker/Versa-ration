-- probe: transcript + audit trail.
--
-- Sessions and turns hold the raw student-facing history. `node_calls`
-- holds the reasoning history: every Node.run() invocation with its
-- inputs and outputs (see CLAUDE.md invariant 2).

CREATE TABLE sessions (
    id          UUID PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE turns (
    id          UUID PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES sessions (id),
    turn_index  INT NOT NULL,
    text        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, turn_index)
);

CREATE INDEX idx_turns_session ON turns (session_id, turn_index);

CREATE TABLE node_calls (
    id           UUID PRIMARY KEY,
    node_name    TEXT NOT NULL,
    session_id   UUID NOT NULL REFERENCES sessions (id),
    turn_index   INT NOT NULL,
    input_json   JSONB NOT NULL,
    output_json  JSONB NOT NULL,
    timestamp    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_node_calls_session
    ON node_calls (session_id, turn_index, timestamp);
CREATE INDEX idx_node_calls_node ON node_calls (node_name);
