"""probe web — Portrait page: the reveal report, rendered instead of
printed. Everything here comes from `build_portrait()` (unchanged,
existing) plus the two new BranchStore cross-session reads — nothing
is computed in this file beyond display formatting/rounding.
"""

from __future__ import annotations

import json

import streamlit as st

from probe.portrait import build_portrait
from probe.revision import RevisionApplicationError
from probe.webui.backend import get_stores, run_async

st.set_page_config(page_title="probe — portrait", page_icon="📊", layout="wide")

learner = st.session_state.get("learner")
if learner is None:
    st.warning("No learner selected — pick one from the Setup page.")
    st.stop()

stores = get_stores()

st.title("probe — portrait")
st.caption(f"Learner: {learner.label or learner.id}")

report = run_async(
    build_portrait(
        learner.id,
        stores.hypotheses,
        stores.transcript,
        stores.concept_graph,
        stores.learner_overlay,
        stores.revisions,
    )
)

st.subheader("Top hypothesis per layer")
for top in report.top_hypotheses:
    if top.hypothesis is None:
        st.write(f"**{top.layer.value}**: _(none active)_")
        continue
    h = top.hypothesis
    st.markdown(f"**{top.layer.value}**: {h.statement}")
    st.caption(f"p={h.probability:.2f}  c={h.confidence:.2f}")
    evidence_lines = []
    for ref in h.evidence_refs:
        turn = run_async(stores.transcript.get_turn(ref.turn_id))
        if turn is not None:
            evidence_lines.append(f"turn {turn.turn_index}: {turn.text}")
    if evidence_lines:
        with st.expander(f"{len(evidence_lines)} evidence turn(s)"):
            for line in evidence_lines:
                st.write(f"- {line}")

st.subheader("Tier counts vs. session count")
st.caption(f"{report.session_count} session(s) so far")
st.bar_chart(report.tier_counts)

st.subheader("Learner overlay")
if not report.overlay:
    st.write("_(no concepts touched yet)_")
else:
    st.dataframe(
        [
            {
                "concept": e.concept_name or e.concept_id,
                "state": e.entry.state.value,
                "confidence": round(e.entry.confidence, 2),
            }
            for e in report.overlay
        ],
        hide_index=True,
    )

st.subheader("Pending world-model revisions")
if not report.pending_revisions:
    st.write("_(none)_")
for revision in report.pending_revisions:
    with st.expander(f"{revision.concept_id}: {revision.proposed_change[:80]}"):
        st.write(f"confidence: {revision.confidence:.2f}")
        st.write(f"proposed_change: {revision.proposed_change}")
        if revision.evidence_refs:
            st.write("evidence:")
            for ref in revision.evidence_refs:
                turn = run_async(stores.transcript.get_turn(ref.turn_id))
                st.write(
                    f"- turn {turn.turn_index if turn else '?'} ({ref.polarity.value})"
                )
        else:
            st.write("evidence: (none)")

        st.caption(
            "Enter the structured edit as JSON, same contract as "
            "`probe review-revisions` — e.g. "
            '{"common_misconceptions": ["..."]}'
        )
        field_updates_raw = st.text_area(
            "field_updates", key=f"field-updates-{revision.id}"
        )
        col_approve, col_reject = st.columns(2)
        if col_approve.button("Approve", key=f"approve-{revision.id}"):
            try:
                parsed = json.loads(field_updates_raw) if field_updates_raw.strip() else {}
            except json.JSONDecodeError:
                st.error("field_updates must be valid JSON.")
            else:
                if not isinstance(parsed, dict) or not parsed:
                    st.error(
                        "field_updates is required to approve — same "
                        "structured-edit contract `probe review-revisions` "
                        "enforces; blank never auto-parses proposed_change."
                    )
                else:
                    try:
                        run_async(stores.revisions.approve(revision.id, parsed))
                    except RevisionApplicationError as exc:
                        st.error(str(exc))
                    else:
                        st.success("Approved.")
                        st.rerun()
        if col_reject.button("Reject", key=f"reject-{revision.id}"):
            run_async(stores.revisions.reject(revision.id))
            st.success("Rejected.")
            st.rerun()

st.subheader('Branch track record — "does it actually predict me"')
st.caption(
    "A click confirms the student picked an option the system offered — "
    "it is not evidence the system predicted them. Kept as a separate "
    "number below, never folded into match_rate."
)
points = run_async(stores.branches.match_rate_by_session_for_learner(learner.id))
if points:
    st.line_chart({"match_rate (text match only)": [p.match_rate for p in points]})
    total_resolved = sum(p.total_resolved for p in points)
    total_matched = sum(p.matched_count for p in points)
    total_clicked = sum(p.option_click_count for p in points)
    st.caption(
        f"{total_matched} / {total_resolved} leaf predictions matched via "
        f"text (prediction accuracy), across all sessions"
    )
    st.caption(f"{total_clicked} leaves resolved via an option click, separately")
else:
    st.write("No resolved branch predictions yet.")

recurring = run_async(stores.branches.recurring_root_statements_for_learner(learner.id))
if recurring:
    st.markdown("**Most-recurring intents**")
    st.caption(
        "Grouped by exact statement text — a differently-worded but "
        "semantically identical intent is counted separately. "
        "\"matched\" and \"match_rate\" are text-match predictions only; "
        "\"clicked\" is tracked separately and never combined into them."
    )
    st.dataframe(
        [
            {
                "statement": r.statement,
                "seen": r.total_count,
                "matched": r.matched_count,
                "match_rate": round(r.match_rate, 2),
                "clicked": r.matched_via_click_count,
            }
            for r in recurring
        ],
        hide_index=True,
    )
