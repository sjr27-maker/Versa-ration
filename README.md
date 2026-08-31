# Versa

**A tutor that stops guessing.**

Instead of quietly assuming what a student meant, Versa recognizes uncertainty, asks one sharp clarifying question with real options — not vague "what do you prefer" quizzing — and remembers how every past ambiguity was resolved. Over time it starts recognizing a student's own thinking style, not from a survey, but from evidence.

---

## What it does

When a student's message is genuinely ambiguous, Versa generates a few distinct interpretations, turns them into natural clickable options, and lets the student resolve it in one tap — no guessing, no wrong assumptions baked into the answer.

Every resolution is written down in plain English as a searchable memory: what was unclear, what was chosen. Before answering anything new, Versa checks that memory first — if it's seen this kind of ambiguity from this student before, it already knows the answer and skips the question.

Across many sessions, it looks for a repeating *order* in how a student reaches understanding — concrete before abstract, or the reverse — and only names that as a real trait once it's been confirmed independently, many times, never from a single guess.

---

## How we built it

We started by building a much larger architecture — a full learner model with hypotheses, a concept graph, a planner scoring six value terms, and a branching prediction tree — then tested it head-to-head against a single plain LLM call on identical conversations.

The comparison showed the heavy machinery wasn't earning its cost. So we cut it down to what actually worked: a lean three-call disambiguation flow, plus a semantic memory layer built on pgvector, plus a conservative, evidence-gated mechanism for detecting a student's thinking style over many sessions rather than asserting one on a hunch.

Every claim in the system is backed by a stored, readable trace — nothing is trusted just because it sounds right.

---

## Built with

**Languages:** Python

**Platforms:** Starlette + a static single-page UI (`probe serve`), Docker / Docker Compose (local dev), Google Cloud Run (deployment)

**Databases:** PostgreSQL 16 with the `pgvector` extension (Cloud SQL in production, Docker locally)

**APIs / SDKs:** Google Gen AI SDK (`google-genai`), Gemini API

**Infra:** Google Cloud SQL, Google Cloud Run, Google Artifact Registry, Google Secret Manager

**Tooling:** `uv` (Python package/env management), pytest, ruff

### Which Google AI models did we use

- **`gemini-3.6-flash`** — fast-tier: ambiguity detection, option generation, fact-writing, semantic confirmation checks
- **`gemini-3.5-flash`** — capable/best-tier: final answer generation (the response the student actually reads)

---

## Architecture

```mermaid
flowchart TD
    A[Student message] --> B[AssessAndBranch]
    B -->|unambiguous| E[FinalAnswer]
    B -->|ambiguous| C[DisambiguationOptions]
    C --> D{Student clicks<br/>or types}
    D -->|click| E
    D -->|types| B
    E --> F[Response shown to student]
    F --> G[WriteLearnerFact]
    G --> H[(learner_facts<br/>pgvector)]
    H -.->|semantic pre-check,<br/>every turn| B
    F --> I[Session end]
    I --> J[Consolidate: label session's<br/>order-of-moves]
    J --> K[(thinking_style_candidates)]
    K -.->|after many confirmed<br/>independent sessions| E
```

---

## Getting Started (Local)

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- Docker Desktop
- A Gemini API key ([aistudio.google.com/apikey](https://aistudio.google.com/apikey))

### Setup

1. Clone and enter the repo:
```bash
   git clone <repo-url>
   cd probe
```

2. Start Postgres (with pgvector):
```bash
   docker compose up -d
```

3. Install dependencies:
```bash
   uv sync
```

4. Configure environment:
```bash
   cp .env.example .env
   # edit .env and set:
   #   GEMINI_API_KEY=<your key>
   #   DATABASE_URL=postgresql://probe:probe@localhost:5434/probe
```

5. Run migrations:
```bash
   uv run probe migrate
```
   Confirm all migrations applied:
```bash
   uv run probe migrate --status
```

6. Launch the web UI:
```bash
   uv run probe serve
```
   Opens at `http://localhost:8000`.

---

## Testing

Run the full automated test suite (no API key required — uses a stub LLM client, zero external calls):

```bash
uv run pytest
```

Try a live session against the real API from the command line:

```bash
uv run probe chat --learner test-user
```

Or run without any API calls or cost, using the stub client:

```bash
uv run probe chat --learner test-user --stub
```

### Credentials required for testing

| Credential | Required for | Where to get it |
|---|---|---|
| `GEMINI_API_KEY` | Any real (non-`--stub`) session | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `DATABASE_URL` | All local testing | Provided automatically by `docker compose up` |

No credentials are required to run the automated test suite — it runs entirely against a stub LLM client with no external calls.

---

## CLI reference

| Command | Description |
|---|---|
| `probe chat --learner <label>` | Start an interactive session |
| `probe serve` | Launch the web UI (single-page app + Starlette API) |
| `probe migrate` | Apply pending database migrations |
| `probe migrate --status` | Show applied/pending migrations without changing anything |
| `probe consolidate-session <id>` | Run cross-session thinking-style detection for one completed session |

---

## Deployment

Deployed on Google Cloud Run, backed by Cloud SQL (Postgres + pgvector) and Secret Manager for credentials. See `Dockerfile` for the container build; the app reads `PORT` from the environment to bind correctly under Cloud Run.