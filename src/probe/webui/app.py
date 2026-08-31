"""probe web — Setup page (entry point).

Launch with `probe web` or `streamlit run src/probe/webui/app.py`.

Pure view-and-drive: every value shown here comes straight from a
store read (LearnerStore.list_all_with_session_counts,
TranscriptStore.list_sessions_for_learner) or a store write
(LearnerStore.create, TranscriptStore.create_session). A session
always runs in minimal_branch (SessionMode.MINIMAL_BRANCH) — there is
no mode picker: the full reasoning path and its ablation presets were
removed.
"""

from __future__ import annotations

import streamlit as st

from probe.webui.backend import get_stores, run_async

st.set_page_config(page_title="probe", page_icon="🔎", layout="wide")
st.title("probe")
st.caption("Local session runner and learner review — replaces the CLI.")

stores = get_stores()

st.header("1. Learner")

tab_existing, tab_new = st.tabs(["Existing learner", "New learner"])

with tab_existing:
    summaries = run_async(stores.learners.list_all_with_session_counts())
    if not summaries:
        st.info("No learners yet — create one in the 'New learner' tab.")
    else:
        labels = []
        by_label: dict[str, object] = {}
        for s in summaries:
            last = (
                s.last_session_at.strftime("%Y-%m-%d %H:%M")
                if s.last_session_at
                else "never"
            )
            name = s.learner.label or str(s.learner.id)
            display = f"{name} — {s.session_count} session(s), last: {last}"
            labels.append(display)
            by_label[display] = s.learner
        choice = st.selectbox("Pick a learner", labels, index=None)
        if choice is not None:
            st.session_state["learner"] = by_label[choice]

with tab_new:
    label = st.text_input("Label", key="new_learner_label")
    display_name = st.text_input("Display name (optional)", key="new_learner_display_name")
    if st.button("Create learner"):
        learner = run_async(
            stores.learners.create(
                label=label.strip() or None,
                display_name=display_name.strip() or None,
            )
        )
        st.session_state["learner"] = learner
        st.success(f"Created learner {learner.id}")
        st.rerun()

learner = st.session_state.get("learner")
if learner is None:
    st.stop()

st.success(f"Selected learner: {learner.label or learner.id}")

st.header("2. Start or resume a session")

use_stub = st.checkbox(
    "Use stub LLM (no cost, no API key needed)",
    value=False,
    help="Matches `probe chat`'s default (the real Gemini API, needs "
    "GEMINI_API_KEY in .env). Check this to run free/offline against "
    "StubLLMClient instead, same as `probe chat --stub`.",
    key="use_stub",
)

st.subheader("New session")
st.caption(
    "Runs in minimal_branch: AssessAndBranch → [options] → FinalAnswer, "
    "at most 3 LLM calls per exchange, plus the memory layer. No concept "
    "graph, no portrait, no planner."
)

if st.button("Start new session", type="primary"):
    session_id = run_async(
        stores.transcript.create_session(learner.id)
    )
    st.session_state["session_id"] = session_id
    st.session_state["turn_index"] = 0
    st.session_state.pop("chat_history", None)
    st.session_state.pop("loop", None)
    st.switch_page("pages/1_Session.py")

st.subheader("Resume a prior session")
sessions = run_async(stores.transcript.list_sessions_for_learner(learner.id))
if not sessions:
    st.write("No prior sessions for this learner.")
else:
    header = st.columns([3, 2, 1])
    header[0].markdown("**Started**")
    header[1].markdown("**Turns**")
    for s in sessions:
        cols = st.columns([3, 2, 1])
        cols[0].write(s.created_at.strftime("%Y-%m-%d %H:%M"))
        cols[1].write(str(s.turn_count))
        if cols[2].button("Resume", key=f"resume-{s.session_id}"):
            st.session_state["session_id"] = s.session_id
            st.session_state["turn_index"] = s.turn_count
            st.session_state.pop("chat_history", None)
            st.session_state.pop("loop", None)
            st.switch_page("pages/1_Session.py")
