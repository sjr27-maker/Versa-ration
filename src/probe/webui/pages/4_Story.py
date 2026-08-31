"""probe web — Story page: the memory layer's one required UI surface
(see memory.py's module docstring / step 9). Every `learner_facts` row
for this learner, in order, rendered as plain-language narrative — not
a table, not raw JSON. This page computes nothing beyond formatting: it
exists purely so a human can read what the memory layer actually wrote
and judge whether it's accurate and useful before anything downstream
(the semantic pre-check, thinking-style detection) is trusted to act
on it.

Thinking-style candidates (memory.py steps 6-8) are deliberately absent
from this page — backend-only for this pass, per the same instruction
that scoped this page to the fact layer alone.
"""

from __future__ import annotations

import streamlit as st

from probe.models import LearnerFactType
from probe.webui.backend import get_stores, run_async

st.set_page_config(page_title="probe — story", page_icon="📖", layout="wide")

learner = st.session_state.get("learner")
if learner is None:
    st.warning("No learner selected — pick one from the Setup page.")
    st.stop()

stores = get_stores()

st.title("probe — story")
st.caption(
    f"Learner: {learner.label or learner.id} — every resolved turn "
    "this memory layer has on record, in order, across every session."
)

facts = run_async(stores.learner_facts.list_by_learner(learner.id))

if not facts:
    st.info(
        "No facts recorded yet — this learner has no minimal_branch "
        "(ReasoningMode.DISAMBIGUATE) turns that resolved something, or "
        "the memory layer wasn't configured for those sessions."
    )
    st.stop()

for fact in facts:
    with st.container(border=True):
        st.caption(
            f"session {fact.session_id} · turn {fact.turn_index} · "
            f"{fact.created_at.strftime('%Y-%m-%d %H:%M')}"
        )
        if fact.fact_type is LearnerFactType.BRANCH_RESOLUTION:
            st.markdown(
                f"**{learner.label or 'The student'}** asked something that "
                f"turned out to be unclear: *{fact.situation}*"
            )
            st.markdown(f"It was resolved like this: {fact.resolution}")
        else:
            st.markdown(
                f"**{learner.label or 'The student'}** asked or did this: "
                f"*{fact.situation}*"
            )
            st.markdown(f"They were answered like this: {fact.resolution}")
