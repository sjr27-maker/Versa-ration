"""probe web — Session page: chat on the left, machine state on the
right. Every number on the right comes from a store read or from
`SessionLoop`/node_calls output — nothing is computed here beyond
display formatting (rounding, sorting, grouping by an already-present
field).
"""

from __future__ import annotations

import streamlit as st

from probe.hypothesis_generator import build_branch_path
from probe.loop import SessionLoop
from probe.models import BranchStatus, Tier
from probe.nodes import MAX_CALLS_PER_TURN, SessionMissingTopicError
from probe.webui.backend import (
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
# (self._last_teach_message, self._generation_width,
# self._prior_generation_id, self._consecutive_ungrounded_turns) is
# in-memory only, same as the CLI's run_interactive — recreating the
# loop between turns would silently reset all of it.
use_stub = st.session_state.get("use_stub", False)
if (
    "loop" not in st.session_state
    or st.session_state.get("loop_session_id") != session_id
    or st.session_state.get("loop_use_stub") != use_stub
):
    tiers = get_tier_clients(use_stub)
    progress, on_node_start = make_progress_tracker()
    st.session_state["loop"] = SessionLoop(
        hypothesis_store=stores.hypotheses,
        transcript=stores.transcript,
        node_calls=stores.node_calls,
        concept_graph=stores.concept_graph,
        learner_overlay=stores.learner_overlay,
        revision_store=stores.revisions,
        llm=tiers.fast,
        model_tier_clients=tiers,
        branch_store=stores.branches,
        diagnostics_store=stores.diagnostics,
        on_node_start=on_node_start,
    )
    st.session_state["progress"] = progress
    st.session_state["loop_session_id"] = session_id
    st.session_state["loop_use_stub"] = use_stub
loop: SessionLoop = st.session_state["loop"]
progress: dict = st.session_state["progress"]

if not st.session_state.get("chat_history"):
    # Reconstruct from the durable record: student turns from `turns`,
    # tutor messages from that turn's own Teach node_calls row.
    history: list[tuple[str, str]] = []
    for t in run_async(stores.transcript.list_turns(session_id)):
        history.append(("student", t.text))
        teach_call = run_async(
            stores.node_calls.get_call_for_turn(session_id, t.turn_index, "Teach")
        )
        if teach_call is not None:
            history.append(("tutor", str(teach_call.output_json)))
    st.session_state["chat_history"] = history

st.title("probe — session")
graph_id = run_async(stores.transcript.get_concept_graph_id(session_id))
topic = None
if graph_id is not None:
    meta = run_async(stores.concept_graph.get_graph(graph_id))
    topic = meta.topic if meta is not None else None
st.caption(
    f"Learner: {learner.label or learner.id}  ·  topic: {topic or '(not yet attached)'}"
)

left, right = st.columns([3, 2])

with left:
    st.subheader("Conversation")
    chat_box = st.container(height=480)
    with chat_box:
        for role, text in st.session_state["chat_history"]:
            with st.chat_message("user" if role == "student" else "assistant"):
                st.write(text)

    progress_placeholder = st.empty()
    user_text = st.chat_input("Say something…")
    if user_text:
        turn_index = st.session_state.get("turn_index", 0)
        progress["node"] = None
        progress_placeholder.markdown("⏳ starting…")
        try:
            message = run_turn_with_progress(
                loop.handle_turn(session_id, turn_index, user_text),
                progress,
                progress_placeholder,
            )
        except SessionMissingTopicError as exc:
            st.error(f"Session has no topic attached: {exc}")
        except Exception as exc:  # noqa: BLE001 — surfaced to the user, not swallowed
            st.error(f"Turn failed: {exc}")
        else:
            st.session_state["chat_history"].append(("student", user_text))
            st.session_state["chat_history"].append(("tutor", message))
            st.session_state["turn_index"] = turn_index + 1
        finally:
            progress_placeholder.empty()
        st.rerun()

latest_turn_index = st.session_state.get("turn_index", 0) - 1

with right:
    tab_hyp, tab_branch, tab_decision, tab_diag = st.tabs(
        ["Hypotheses", "Branch tree", "Decision trace", "Diagnostics"]
    )

    # --- 1. Hypotheses -------------------------------------------------
    with tab_hyp:
        tier_counts = {
            tier.value: len(run_async(stores.hypotheses.list_by_learner(learner.id, tier=tier)))
            for tier in Tier
        }
        st.caption(" · ".join(f"{k}: {v}" for k, v in tier_counts.items()))

        active = run_async(
            stores.hypotheses.list_by_learner(learner.id, tier=Tier.ACTIVE)
        )
        by_layer: dict[str, list] = {}
        for h in active:
            by_layer.setdefault(h.layer.value, []).append(h)

        for layer_name in sorted(by_layer):
            st.markdown(f"**{layer_name}**")
            for h in sorted(by_layer[layer_name], key=lambda x: -x.probability):
                st.progress(
                    h.probability,
                    text=f"{h.statement}  (p={h.probability:.2f}, c={h.confidence:.2f})",
                )
                dated = [r for r in h.evidence_refs if r.resulting_probability is not None]
                dated.sort(key=lambda r: r.timestamp)
                if len(dated) >= 2:
                    delta = dated[-1].resulting_probability - dated[-2].resulting_probability
                    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "→")
                    st.caption(f"{arrow} Δ{delta:+.2f} since previous update")
                tier_changes = run_async(stores.hypotheses.list_tier_changes(h.id))
                if tier_changes:
                    last_change = tier_changes[-1]
                    if last_change.new_tier is Tier.ACTIVE and last_change.old_tier in (
                        Tier.DORMANT,
                        Tier.BACKGROUND,
                    ):
                        st.caption(f"↩ resurrected from {last_change.old_tier.value}")
                with st.expander("Evidence"):
                    for ref in h.evidence_refs:
                        turn = run_async(stores.transcript.get_turn(ref.turn_id))
                        st.write(
                            f"- turn {turn.turn_index if turn else '?'}: "
                            f"{turn.text if turn else '(turn not found)'}"
                        )

    # --- 2. Branch tree --------------------------------------------------
    with tab_branch:
        resolution_call = run_async(
            stores.node_calls.get_latest_call(session_id, "BranchResolve")
        )
        if resolution_call is not None:
            status = resolution_call.output_json.get("status")
            if status == "matched":
                st.success("Previous turn: MATCHED")
            elif status == "unmatched":
                st.error("⚠ Previous turn: NOTHING MATCHED")

        generation = run_async(stores.branches.get_latest_generation(session_id))
        if generation is None:
            st.write("No branches generated yet.")
        else:
            branches = run_async(stores.branches.list_by_generation(generation.id))
            by_parent: dict = {}
            for b in branches:
                by_parent.setdefault(b.parent_id, []).append(b)

            status_marker = {
                BranchStatus.MATCHED: "✅",
                BranchStatus.UNMATCHED: "❌",
                BranchStatus.SUPERSEDED: "⏹",
                BranchStatus.OPEN: "•",
            }
            selected_id = generation.selected_branch_id

            def render(parent_id, indent):
                for b in by_parent.get(parent_id, []):
                    prefix = "  " * indent
                    weight = "**" if b.depth == 0 else ""
                    star = "⭐ " if b.id == selected_id else ""
                    st.markdown(
                        f"{prefix}{status_marker[b.status]} {star}{weight}[{b.depth_label}] "
                        f"{b.statement}{weight}  \n"
                        f"{prefix}　plausibility={b.plausibility:.2f} · "
                        f"predicts: _{b.predicted_next_turn}_"
                    )
                    render(b.id, indent + 1)

            render(None, 0)

            if selected_id is not None:
                path = build_branch_path(branches, selected_id)
                st.markdown("**⭐ Selected path (root → selected)**")
                st.markdown(
                    " → ".join(f"[{b.depth_label}] {b.statement}" for b in path)
                )
                if generation.selection_rationale:
                    st.caption(f"Why: {generation.selection_rationale}")

            if generation.path_requirement is not None:
                pr = generation.path_requirement
                st.markdown("**Derived PathRequirement (what Teach was told)**")
                st.write(f"Believes: {pr.current_belief or '_(none)_'}")
                st.write(f"Needs: {pr.needed or '_(none)_'}")
                st.write(f"Scope: {pr.scope or '_(none)_'}")
                if pr.must_not_assume:
                    st.warning(
                        "Must NOT assume:\n"
                        + "\n".join(f"- {item}" for item in pr.must_not_assume)
                    )

        generation_call = run_async(
            stores.node_calls.get_latest_call(session_id, "BranchGenerate")
        )
        if generation_call is not None:
            notes = [
                n
                for n in generation_call.output_json.get("redundancy_notes", [])
            ]
            if notes:
                with st.expander(f"Redundancy check ({len(notes)} branch(es) cleared it)"):
                    for note in notes:
                        st.write(f"- {note}")

    # --- 3. Decision trace -------------------------------------------------
    with tab_decision:
        plan_call = (
            run_async(stores.node_calls.get_call_for_turn(session_id, latest_turn_index, "Plan"))
            if latest_turn_index >= 0
            else None
        )
        if plan_call is None:
            st.write("No Plan call yet this session.")
        else:
            st.caption(
                f"generation_width={plan_call.input_json.get('generation_width')}  ·  "
                f"exploration_target="
                f"{plan_call.input_json.get('exploration_target') or 'none available'}"
            )
            winner_action = plan_call.output_json["winner"]["action"]
            if plan_call.output_json.get("argmax_changes_without_information_value"):
                st.warning("Winner would change if information_value were zeroed.")
            rows = []
            for score in plan_call.output_json["scores"]:
                rows.append(
                    {
                        "action": score["candidate"]["action"],
                        "winner": "🏆" if score["candidate"]["action"] == winner_action else "",
                        "learning_value": round(score["learning_value"], 3),
                        "information_value": round(score["information_value"], 3),
                        "long_term_value": round(score["long_term_value"], 3),
                        "time_cost": round(score["time_cost"], 3),
                        "cognitive_cost": round(score["cognitive_cost"], 3),
                        "frustration_risk": round(score["frustration_risk"], 3),
                        "total": round(score["total"], 3),
                        "flags": ", ".join(score.get("flags", [])),
                    }
                )
            st.dataframe(rows, hide_index=True)

    # --- 4. Diagnostics --------------------------------------------------
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
            if diagnostics.inferred_topic is not None:
                st.info(
                    f"AttachTopic inferred: **{diagnostics.inferred_topic}** "
                    f"({'seeded a fresh graph' if diagnostics.topic_seeded_new else 'resumed an existing graph'})"
                )
            if diagnostics.entropy_bits is not None:
                st.caption(f"entropy_bits: {diagnostics.entropy_bits:.2f}")
            st.markdown("**Calls by node**")
            st.json(diagnostics.node_call_counts)
            if diagnostics.teach_failed:
                st.error("Teach failed this turn — no real teaching content was produced.")
            if diagnostics.warnings:
                st.markdown("**Warnings**")
                for w in diagnostics.warnings:
                    st.warning(w)
