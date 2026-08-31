"""probe/webserver.py — the Starlette API behind `probe serve` and the
calm single-page UI.

Same discipline as test_disambiguation_loop_wiring.py: drive a real
SessionLoop turn (StubLLMClient, canned responses) through the same
stores the server uses, then assert the read helpers shape the JSON the
frontend (probe/static/app.js) actually depends on — the branch
statements, the pending options, and the per-turn trace. The server's
routes are thin wrappers over these helpers, so this is where the
contract lives.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest

from probe import webserver
from probe.llm import StubLLMClient
from probe.loop import SessionLoop

_TWO_BRANCHES = json.dumps(
    {
        "needs_branches": True,
        "branches": [
            {"statement": "wants the power rule explained"},
            {"statement": "wants a worked numeric example"},
        ],
    }
)

_OPT_TEXT = {
    "wants the power rule explained": "Explain the power rule?",
    "wants a worked numeric example": "Show a worked example?",
}


class _BranchAwareStub:
    """StubLLMClient, except DISAMBIGUATE:OPTIONS is answered by reading
    the real branch ids back out of the prompt (nodes.py renders them as
    `- id=<uuid>: <statement>`) — a static canned string can't, since
    those ids are minted per turn. Same trick as
    test_disambiguation_loop_wiring.py's fake clients.
    """

    def __init__(self) -> None:
        self._inner = StubLLMClient(canned={"ASSESS:BRANCH": _TWO_BRANCHES})
        self.prompts: list[str] = self._inner.prompts

    async def complete(self, prompt: str) -> str:
        if prompt.startswith("DISAMBIGUATE:OPTIONS"):
            rows = re.findall(r"- id=([0-9a-f-]{36}): (.+)", prompt)
            return json.dumps(
                [
                    {"branch_id": bid, "text": _OPT_TEXT.get(stmt.strip(), "confirm?")}
                    for bid, stmt in rows
                ]
            )
        if prompt.startswith("FINAL:ANSWER"):
            return "here is the power rule"
        return await self._inner.complete(prompt)


def _stores_dict(pool) -> dict:
    from probe.audit import NodeCallStore, TranscriptStore
    from probe.diagnostics import TurnDiagnosticsStore
    from probe.disambiguate import DisambiguationStore
    from probe.learner import LearnerStore

    return {
        "transcript": TranscriptStore(pool),
        "node_calls": NodeCallStore(pool),
        "diagnostics": TurnDiagnosticsStore(pool),
        "learners": LearnerStore(pool),
        "disambiguation": DisambiguationStore(pool),
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_learner_label_creates_then_resumes(learner_store):
    a = await webserver._resolve_learner(learner_store, "calm-ui-test")
    assert a.label == "calm-ui-test"
    b = await webserver._resolve_learner(learner_store, "calm-ui-test")
    assert b.id == a.id  # same label resumes, does not duplicate


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_learner_unknown_uuid_raises(learner_store):
    with pytest.raises(LookupError):
        await webserver._resolve_learner(
            learner_store, "00000000-0000-0000-0000-000000000000"
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_inspect_and_history_payloads_track_a_branch_then_click(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    diagnostics_store,
):
    session_id = await transcript.create_session(learner_id)
    loop = SessionLoop(
        transcript=transcript,
        node_calls=node_calls,
        llm=_BranchAwareStub(),
        diagnostics_store=diagnostics_store,
        disambiguation_store=disambiguation_store,
    )
    stores = _stores_dict(clean_pool)

    # turn 0: ambiguous -> options shown, no answer
    msg = await loop.handle_turn(session_id, 0, "explain x squared")
    assert msg == "Which of these did you mean?"

    pending = await webserver._pending_options(stores, session_id)
    assert len(pending) == 2
    assert all("id" in o and "text" in o for o in pending)

    inspect = await webserver._inspect_payload(stores, session_id)
    assert inspect["branched"] is True
    assert [b["statement"] for b in inspect["branches"]] == [
        "wants the power rule explained",
        "wants a worked numeric example",
    ]
    assert all(b["status"] == "open" for b in inspect["branches"])
    # each branch carries the option text it maps to (the Story view)
    assert {b["option"] for b in inspect["branches"]} == {
        "Explain the power rule?",
        "Show a worked example?",
    }
    assert inspect["diagnostics"]["node_call_counts"]  # trace is populated

    # turn 1: click the first option -> its branch matched, sibling
    # superseded, a real answer this time
    answer = await loop.handle_turn(
        session_id, 1, pending[0]["text"], uuid.UUID(pending[0]["id"])
    )
    assert answer == "here is the power rule"

    assert await webserver._pending_options(stores, session_id) == []

    inspect2 = await webserver._inspect_payload(stores, session_id)
    statuses = {b["statement"]: b["status"] for b in inspect2["branches"]}
    assert statuses["wants the power rule explained"] == "matched"
    assert statuses["wants a worked numeric example"] == "superseded"
    # TRACE follows the latest turn (the click), not the branch turn
    assert inspect2["diagnostics"]["turn_index"] == 1
    assert "FinalAnswer" in inspect2["diagnostics"]["node_call_counts"]

    history = await webserver._rebuild_history(stores, session_id)
    assert history[0] == {"role": "student", "text": "explain x squared"}
    assert history[-1] == {"role": "tutor", "text": "here is the power rule"}
    # the "show options" turn contributes a student line but no tutor line
    assert [h["role"] for h in history] == ["student", "student", "tutor"]


def test_create_app_serves_the_spa_and_404s_unknown_session(monkeypatch):
    from starlette.testclient import TestClient

    from tests.conftest import DATABASE_URL

    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    with TestClient(webserver.create_app()) as client:
        root = client.get("/")
        assert root.status_code == 200
        # the single-page shell (title string is mid-rename probe->versa,
        # so assert on structure, not the brand word)
        assert root.headers["content-type"].startswith("text/html")
        assert '<script src="/static/app.js">' in root.text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/session/not-a-uuid").status_code == 404

        # create a session, then it must appear in that learner's
        # resume list
        made = client.post(
            "/api/session", json={"learner": "calm-client-test", "stub": True}
        )
        assert made.status_code == 200
        learner_id = made.json()["learner"]["id"]
        session_id = made.json()["session_id"]
        listed = client.get(f"/api/learners/{learner_id}/sessions")
        assert listed.status_code == 200
        assert session_id in {
            s["session_id"] for s in listed.json()["sessions"]
        }
        # consolidate on a fresh stub session: nothing to consolidate
        con = client.post(f"/api/session/{session_id}/consolidate")
        assert con.status_code == 200
        assert con.json()["consolidated"] is False

        # Learners panel: the created learner is in the list
        all_learners = client.get("/api/learners").json()["learners"]
        assert learner_id in {x["id"] for x in all_learners}

        # POST /api/learners creates one; Story endpoint returns [] for it
        made2 = client.post(
            "/api/learners", json={"label": "calm-client-story"}
        )
        assert made2.status_code == 200
        lid2 = made2.json()["id"]
        assert client.get(f"/api/learners/{lid2}/facts").json()["facts"] == []

        # Compare needs two real sessions; a bad pair 404s cleanly
        s2 = client.post(
            "/api/session", json={"learner": "calm-client-test", "stub": True}
        ).json()["session_id"]
        cmp = client.get(f"/api/compare?a={session_id}&b={s2}")
        assert cmp.status_code == 200
        assert cmp.json()["mode_differs"] is False
        assert client.get("/api/compare?a=nope&b=nope").status_code == 400
