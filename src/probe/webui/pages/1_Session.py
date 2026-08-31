"""probe web — Session page: chat on the left, machine state on the
right. The right side is two panels only now that the full reasoning
path is gone: the **Story** panel (this session's disambiguation
branches and how each turn resolved) and the **Diagnostics** strip
(latency, call count, the MAX_CALLS_PER_TURN guardrail, warnings, and
the memory layer's skip-visibility fields). Every value comes from a
store read or from node_calls output — nothing is computed here beyond
display formatting.
"""

from __future__ import annotations

import streamlit as st

from probe.baseline import MAX_CALLS_PER_TURN
from probe.loop import SessionLoop
from probe.models import BranchStatus, OptionStatus
from probe.webui.backend import (
    get_embedding_client,
    get_stores,
    get_tier_clients,
    make_progress_tracker,
    run_async,
    run_turn_with_progress,
)

st.set_page_config(page_title="probe — session", page_icon="💬", layout="wide")

session_id = st.session_state.get("session_id")
learner = st.session_state.get("learner")
if session_id is None or learner is None:
    st.warning("No active session — start or resume one from the Setup page.")
    st.stop()

stores = get_stores()

# The SessionLoop instance is cached across reruns (Streamlit reruns
# the whole script every interaction): its turn-to-turn state
# (self._prior_disambiguation_turn_id) is in-memory only, same as the
# CLI's run_interactive — recreating the loop between turns would
# silently reset it.
use_stub = st.session_state.get("use_stub", False)
# Fixed at session creation and never re-read after — a session's
# config is set-once, so caching it alongside the loop instance below
# is exactly as safe as caching the loop itself.
ablation_config = run_async(stores.transcript.get_ablation_config(session_id))
if (
    "loop" not in st.session_state
    or st.session_state.get("loop_session_id") != session_id
    or st.session_state.get("loop_use_stub") != use_stub
):
    tiers = get_tier_clients(use_stub)
    embedding_client = get_embedding_client(use_stub)
    progress, on_node_start = make_progress_tracker()
    st.session_state["loop"] = SessionLoop(
        transcript=stores.transcript,
        node_calls=stores.node_calls,
        llm=tiers.fast,
        model_tier_clients=tiers,
        diagnostics_store=stores.diagnostics,
        on_node_start=on_node_start,
        ablation_config=ablation_config,
        disambiguation_store=stores.disambiguation,
        learner_fact_store=stores.learner_facts,
        thinking_style_store=stores.thinking_styles,
        embedding_client=embedding_client,
    )
    st.session_state["progress"] = progress
    st.session_state["loop_session_id"] = session_id
    st.session_state["loop_use_stub"] = use_stub
loop: SessionLoop = st.session_state["loop"]
progress: dict = st.session_state["progress"]


def _tutor_message_for_turn(turn_index: int) -> str | None:
    for node_name in ("FinalAnswer", "BaselineTeach"):
        call = run_async(
            stores.node_calls.get_call_for_turn(session_id, turn_index, node_name)
        )
        if call is not None:
            return str(call.output_json)
    return None


if not st.session_state.get("chat_history"):
    # Reconstruct from the durable record: student turns from `turns`,
    # tutor messages from that turn's own FinalAnswer/BaselineTeach
    # node_calls row (a "show options" turn has neither — nothing to
    # append for it).
    history: list[tuple[str, str]] = []
    for t in run_async(stores.transcript.list_turns(session_id)):
        history.append(("student", t.text))
        tutor = _tutor_message_for_turn(t.turn_index)
        if tutor is not None:
            history.append(("tutor", tutor))
    st.session_state["chat_history"] = history

st.title("probe — session")
st.caption(f"Learner: {learner.label or learner.id}")

if ablation_config.is_full_bypass:
    st.warning("⚗ BASELINE — plain LLM, one call per turn. Fixed for this session.")
else:
    st.info(
        "⚗ minimal_branch (SessionMode.MINIMAL_BRANCH) — AssessAndBranch "
        "→ [options] → FinalAnswer, at most 3 calls per exchange, plus "
        "the memory layer."
    )

# The memory layer's explicit, unambiguous consolidation trigger (see
# memory.py steps 6-8 / SessionLoop.consolidate_session) — a person
# clicking this has decided the session is done. A no-op (reported
# plainly) for a session with no learner_facts (a BASELINE session, or
# one that never resolved anything).
if st.button("End session & consolidate"):
    consolidation_result = run_async(loop.consolidate_session(session_id))
    if consolidation_result is None:
        st.info(
            "Nothing to consolidate — this session wrote no learner_facts "
            "(a BASELINE session, or it never resolved anything)."
        )
    else:
        st.success(
            f"Thinking-style candidate {consolidation_result.id}: "
            f"{consolidation_result.path_summary!r} "
            f"(confirmation_count={consolidation_result.confirmation_count}, "
            f"status={consolidation_result.status.value})"
        )

left, right = st.columns([3, 2])


def _run_turn(text: str, progress_placeholder, selected_option_id=None) -> None:
    """Shared by typed input and option-button clicks — both are just a
    message plus, for a click, which option produced it."""
    turn_index = st.session_state.get("turn_index", 0)
    progress["node"] = None
    progress_placeholder.markdown("⏳ starting…")
    try:
        message = run_turn_with_progress(
            loop.handle_turn(session_id, turn_index, text, selected_option_id),
            progress,
            progress_placeholder,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced to the user, not swallowed
        st.error(f"Turn failed: {exc}")
    else:
        st.session_state["chat_history"].append(("student", text))
        st.session_state["chat_history"].append(("tutor", message))
        st.session_state["turn_index"] = turn_index + 1
    finally:
        progress_placeholder.empty()
    st.rerun()


with left:
    st.subheader("Conversation")
    chat_box = st.container(height=480)
    with chat_box:
        for role, text in st.session_state["chat_history"]:
            with st.chat_message("user" if role == "student" else "assistant"):
                st.write(text)

    # Options pending from the most recent disambiguation turn — both
    # channels stay open: clicking one here resolves its branch
    # directly (no LLM call); typing in the box instead is treated as
    # typing past the options (see disambiguate.py). Only ever the
    # latest turn's still-open options.
    pending_options = []
    if not ablation_config.is_full_bypass:
        latest_disambiguation_turn = run_async(
            stores.disambiguation.get_latest_turn(session_id)
        )
        if latest_disambiguation_turn is not None:
            all_options = run_async(
                stores.disambiguation.list_options_for_turn(latest_disambiguation_turn.id)
            )
            pending_options = [o for o in all_options if o.status is OptionStatus.OPEN]

    progress_placeholder = st.empty()

    if pending_options:
        option_cols = st.columns(len(pending_options))
        for col, option in zip(option_cols, pending_options, strict=True):
            if col.button(option.text, key=f"option-{option.id}", use_container_width=True):
                _run_turn(option.text, progress_placeholder, selected_option_id=option.id)

    user_text = st.chat_input("Say something…")
    if user_text:
        _run_turn(user_text, progress_placeholder)

latest_turn_index = st.session_state.get("turn_index", 0) - 1

with right:
    tab_story, tab_diag = st.tabs(["Story", "Diagnostics"])

    # --- 1. Story -----------------------------------------------------
    with tab_story:
        if ablation_config.is_full_bypass:
            st.write("BASELINE sessions have no branches — see Diagnostics.")
        else:
            latest = run_async(stores.disambiguation.get_latest_turn(session_id))
            if latest is None:
                st.write("No disambiguation branches generated yet this session.")
            else:
                branches = run_async(
                    stores.disambiguation.list_branches_for_turn(latest.id)
                )
                options = run_async(
                    stores.disambiguation.list_options_for_turn(latest.id)
                )
                option_text_by_branch = {o.branch_id: o.text for o in options}
                status_marker = {
                    BranchStatus.MATCHED: "✅",
                    BranchStatus.SUPERSEDED: "⏹",
                    BranchStatus.OPEN: "•",
                    BranchStatus.UNMATCHED: "❌",
                }
                st.caption(
                    f"latest turn {latest.turn_index} · "
                    + ("branched" if latest.needs_branches else "answered directly")
                )
                if not latest.needs_branches:
                    st.write("The last message was judged unambiguous — answered directly.")
                for b in branches:
                    st.markdown(
                        f"{status_marker.get(b.status, '•')} {b.statement}"
                    )
                    if b.id in option_text_by_branch:
                        st.caption(f"　option shown: “{option_text_by_branch[b.id]}”")

    # --- 2. Diagnostics ---------------------------------------------------
    with tab_diag:
        diagnostics = (
            run_async(stores.diagnostics.get_for_turn(session_id, latest_turn_index))
            if latest_turn_index >= 0
            else None
        )
        if diagnostics is None:
            st.write("No diagnostics recorded yet this session.")
        else:
            cols = st.columns(3)
            cols[0].metric("Calls this turn", diagnostics.total_call_count)
            cols[1].metric(
                "MAX_CALLS_PER_TURN",
                f"{'EXCEEDED' if diagnostics.guardrail_fired else 'ok'} ({MAX_CALLS_PER_TURN})",
            )
            cols[2].metric("Duration", f"{diagnostics.duration_ms:.0f} ms")
            st.caption(f"retries this turn: {diagnostics.retry_count}")
            st.markdown("**Calls by node**")
            st.json(diagnostics.node_call_counts)
            if diagnostics.teach_failed:
                st.error("The response call failed this turn — no real answer was produced.")
            if diagnostics.branching_skipped_by_memory:
                st.success(
                    "⚡ branching skipped by memory — a past fact "
                    f"(`{diagnostics.matched_fact_id}`) was confirmed to "
                    "resolve this message, so AssessAndBranch never ran."
                )
            elif diagnostics.memory_match_found:
                st.info(
                    "Memory match found (vector search cleared the "
                    "threshold) but "
                    + (
                        "the confirmation call said it does not resolve this message."
                        if not diagnostics.memory_match_confirmed_resolution
                        else "confirmed."
                    )
                )
            if diagnostics.warnings:
                st.markdown("**Warnings**")
                for w in diagnostics.warnings:
                    st.warning(w)
