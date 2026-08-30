"""probe web — Comparison page: pick two sessions, see their
AblationConfigs (differences highlighted), per-turn latency/call count,
and their full transcripts aligned turn by turn.

Quality is judged by reading, not scored automatically — this page's
only job is making two transcripts readable side by side. Every value
shown comes straight from a store read (TranscriptStore, node_calls,
TurnDiagnosticsStore) — nothing here computes a judgment about which
session did better.
"""

from __future__ import annotations

import streamlit as st

from probe.webui.backend import get_stores, run_async

st.set_page_config(page_title="probe — compare", page_icon="⚖️", layout="wide")
st.title("probe — compare two sessions")
st.caption(
    "The harness that decides which layers earn their latency and cost: "
    "pick any two sessions (different ablation configs, or the same "
    "config run twice) and read them side by side."
)

stores = get_stores()

learner_summaries = run_async(stores.learners.list_all_with_session_counts())
picker_options: list[tuple[str, object]] = []
for summary in learner_summaries:
    sessions = run_async(stores.transcript.list_sessions_for_learner(summary.learner.id))
    for s in sessions:
        label = (
            f"{summary.learner.label or summary.learner.id} — "
            f"{s.topic or '(no topic)'} — "
            f"{s.created_at.strftime('%Y-%m-%d %H:%M')} ({s.turn_count} turns)"
        )
        picker_options.append((label, s))

if len(picker_options) < 2:
    st.info("Need at least two sessions to compare — start some from the Setup page.")
    st.stop()

labels = [label for label, _ in picker_options]
by_label = dict(picker_options)

col_pick_a, col_pick_b = st.columns(2)
with col_pick_a:
    choice_a = st.selectbox("Session A", labels, index=0, key="compare_session_a")
with col_pick_b:
    choice_b = st.selectbox(
        "Session B", labels, index=min(1, len(labels) - 1), key="compare_session_b"
    )

if choice_a == choice_b:
    st.warning("Pick two different sessions to compare.")
    st.stop()

session_a = by_label[choice_a]
session_b = by_label[choice_b]

config_a = run_async(stores.transcript.get_ablation_config(session_a.session_id))
config_b = run_async(stores.transcript.get_ablation_config(session_b.session_id))

st.header("1. Config")
_fields = [
    "enable_portrait",
    "enable_concept_graph",
    "enable_diagnose",
    "enable_planner",
    "enable_branches",
    "enable_options",
    "enable_exploration_slot",
    "reasoning_budget_mode",
]


def _display(value: object) -> str:
    # Always a string: this column mixes booleans (the enable_* flags)
    # with an enum value (reasoning_budget_mode) row to row, which
    # pandas/pyarrow can't give one consistent dtype -- st.dataframe
    # silently "fixes" it with a scary traceback logged underneath,
    # rather than erroring, so this is worth avoiding outright.
    return str(value.value if hasattr(value, "value") else value)


config_rows = []
for field in _fields:
    value_a = getattr(config_a, field)
    value_b = getattr(config_b, field)
    config_rows.append(
        {
            "field": field,
            "A": _display(value_a),
            "B": _display(value_b),
            "differs": "⚠ differs" if value_a != value_b else "",
        }
    )
st.dataframe(config_rows, hide_index=True)

st.header("2. Per-turn cost")


def _cost_rows(session_id):
    diagnostics = run_async(stores.diagnostics.list_for_session(session_id))
    return diagnostics, [
        {
            "turn": d.turn_index,
            "duration_ms": round(d.duration_ms, 0),
            "calls": d.total_call_count,
            "retries": d.retry_count,
        }
        for d in diagnostics
    ]


diagnostics_a, rows_a = _cost_rows(session_a.session_id)
diagnostics_b, rows_b = _cost_rows(session_b.session_id)

cost_col_a, cost_col_b = st.columns(2)
for col, session, diagnostics, rows in (
    (cost_col_a, session_a, diagnostics_a, rows_a),
    (cost_col_b, session_b, diagnostics_b, rows_b),
):
    with col:
        st.subheader(session.topic or "(no topic)")
        if not rows:
            st.write("No diagnostics recorded for this session.")
            continue
        st.dataframe(rows, hide_index=True)
        st.caption(
            f"mean: {sum(d.duration_ms for d in diagnostics) / len(diagnostics):.0f} ms  ·  "
            f"{sum(d.total_call_count for d in diagnostics) / len(diagnostics):.1f} calls/turn"
        )

st.header("3. Transcripts, aligned turn by turn")


def _tutor_message(session_id, turn_index) -> str | None:
    # A BASELINE turn's response lives under "BaselineTeach", not
    # "Teach" (see loop.py's _handle_bypass_turn) — check both so this
    # page doesn't quietly show blank tutor turns for a baseline session.
    for node_name in ("Teach", "BaselineTeach"):
        call = run_async(stores.node_calls.get_call_for_turn(session_id, turn_index, node_name))
        if call is not None:
            return str(call.output_json)
    return None


turns_a = run_async(stores.transcript.list_turns(session_a.session_id))
turns_b = run_async(stores.transcript.list_turns(session_b.session_id))

for i in range(max(len(turns_a), len(turns_b))):
    col_a, col_b = st.columns(2)
    for col, turns, session_id in (
        (col_a, turns_a, session_a.session_id),
        (col_b, turns_b, session_b.session_id),
    ):
        with col:
            if i >= len(turns):
                st.write("_(session ended)_")
                continue
            st.markdown(f"**Turn {i} — student**")
            st.write(turns[i].text)
            st.markdown("**tutor**")
            message = _tutor_message(session_id, i)
            st.write(message if message is not None else "_(no response recorded)_")
    st.divider()
