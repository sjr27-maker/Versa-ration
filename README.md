# probe

Research artifact auditing how an AI tutor's beliefs about a student
evolve over time. See `CLAUDE.md` for invariants and setup.

## Manual smoke test: real Gemini client, one live turn

Not part of the automated suite (`pytest` only ever exercises
`StubLLMClient`) — this is a one-off, by-hand check that the real
`GeminiLLMClient` wiring produces a plausible turn end-to-end, run
manually whenever the client/tiering wiring changes.

Before running this against the real API, estimate the call count for
a representative turn against the stub first (no cost) — e.g. the
`estimate_turn_calls.py`-style script used to produce the numbers
below:

- **turn 1, empty hypothesis store** (the cheapest case — zero
  entropy, `generation_width` floors to the exploration reservation,
  `information_value` off): **10** calls.
- **mid-session, 5 active hypotheses spread across 4 layers plus 1
  dormant one** (entropy ≈4.6 bits → `generation_width=6`,
  `run_information_value=True`, an exploration candidate targeting the
  dormant hypothesis): **28** calls, and `MAX_CALLS_PER_TURN`'s own
  count for that same turn is also 28 — every LLM-calling node/term
  (Infer, Plan's proposer, Teach, and all four of ValueFunction's
  LLM-calling terms) is now individually instrumented, not just
  Diagnose + information_value, so the guardrail sees the real total.

A turn-1 smoke test will look cheap regardless of provider; the
mid-session number is the one to budget against for a real session.

```
GEMINI_API_KEY=<a real key> uv run probe chat --learner smoke-test --topic "derivatives"
```

Type one message and confirm:

- a plausible tutoring response comes back (not an error, not a raw
  JSON blob leaking into the chat)
- `probe portrait <learner_id>` afterward shows the turn's hypothesis
  updates (if any) and doesn't error
- no `LLMTransportError` in the terminal (a single turn making a
  handful of calls should not exhaust the client's retry budget)

Pass `--stub` instead of a real key to run the same command for free
against `StubLLMClient` (this is what the automated suite already
covers, so `--stub` here is just for manually eyeballing the CLI flow).
