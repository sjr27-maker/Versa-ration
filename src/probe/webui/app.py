"""probe web — Setup page (entry point).

Launch with `probe web` or `streamlit run src/probe/webui/app.py`.

Pure view-and-drive: every value shown here comes straight from a
store read (LearnerStore.list_all_with_session_counts,
TranscriptStore.list_sessions_for_learner) or a store write
(LearnerStore.create, TranscriptStore.create_session). No scoring, no
topic resolution — that happens turn 0, inside SessionLoop, via
AttachTopic.
"""

from __future__ import annotations

import streamlit as st

from probe.ablation import AblationConfig, AblationPreset, ReasoningMode, build_preset
from probe.webui.backend import get_stores, run_async

_PRESET_LABELS: dict[AblationPreset, str] = {
    AblationPreset.BASELINE: "baseline — plain LLM, every subsystem off",
    AblationPreset.PORTRAIT: "+portrait — hypotheses only",
    AblationPreset.GRAPH: "+graph — portrait + concept grounding + diagnosis",
    AblationPreset.PLANNER: "+planner — the above + full Plan/value scoring",
    AblationPreset.BRANCHES: "+branches — the above + generation/selection/path",
    AblationPreset.OPTIONS: "+options — the above + evidence extraction (full system)",
}
_DISABLED_BY_PRESET: dict[AblationPreset, list[str]] = {
    AblationPreset.BASELINE: [
        "hypotheses", "concept grounding", "diagnosis", "planning",
        "branch generation", "evidence-extraction options",
    ],
    AblationPreset.PORTRAIT: [
        "concept grounding", "diagnosis", "planning", "branch generation",
        "evidence-extraction options",
    ],
    AblationPreset.GRAPH: ["planning", "branch generation", "evidence-extraction options"],
    AblationPreset.PLANNER: ["branch generation", "evidence-extraction options"],
    AblationPreset.BRANCHES: ["evidence-extraction options"],
    AblationPreset.OPTIONS: [],
}

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
    "No topic field — the first message you send determines the topic "
    "(AttachTopic infers it and attaches or seeds a concept graph "
    "automatically)."
)

st.markdown("**Reasoning mode** — which architecture this session runs, not a toggle.")
reasoning_mode = st.radio(
    "Reasoning mode",
    [ReasoningMode.FULL, ReasoningMode.DISAMBIGUATE],
    format_func=lambda m: (
        "Full system — ablatable via the presets/toggles below"
        if m is ReasoningMode.FULL
        else "Minimal branch (disambiguation) — AssessAndBranch → "
        "[options] → FinalAnswer, at most 3 calls per exchange"
    ),
    index=0,
    key="reasoning_mode",
    label_visibility="collapsed",
)

if reasoning_mode is ReasoningMode.DISAMBIGUATE:
    # A distinct choice, not one more preset under AblationPreset: this
    # is a different reasoning_mode, not another toggle combination of
    # the full system (see disambiguate.py's module docstring) — none
    # of the preset/toggle machinery below applies, so it's skipped
    # entirely rather than shown disabled.
    st.caption(
        "Replaces the branch tree / SelectBranch / DerivePath / Plan / "
        "concept-graph machinery outright for this session — no "
        "hypotheses, no concept grounding, no Plan. See CLAUDE.md "
        "invariant 9 / disambiguate.py."
    )
    ablation_config = AblationConfig(reasoning_mode=ReasoningMode.DISAMBIGUATE)
    config_error = None
else:
    st.markdown(
        "**Ablation config** — the harness that decides which layers earn their cost."
    )
    preset = st.selectbox(
        "Preset (starting point — every toggle below stays editable)",
        list(AblationPreset),
        format_func=lambda p: _PRESET_LABELS[p],
        index=len(list(AblationPreset)) - 1,  # default: full system
        key="ablation_preset",
    )
    base_config = build_preset(preset)
    disabled = _DISABLED_BY_PRESET[preset]
    st.caption(
        "This preset disables: "
        + (", ".join(disabled) if disabled else "nothing — full system")
    )
    with st.expander("Adjust individual toggles"):
        enable_portrait = st.checkbox("enable_portrait", value=base_config.enable_portrait)
        enable_concept_graph = st.checkbox(
            "enable_concept_graph", value=base_config.enable_concept_graph
        )
        enable_diagnose = st.checkbox("enable_diagnose", value=base_config.enable_diagnose)
        enable_planner = st.checkbox("enable_planner", value=base_config.enable_planner)
        enable_branches = st.checkbox("enable_branches", value=base_config.enable_branches)
        enable_options = st.checkbox(
            "enable_options",
            value=base_config.enable_options,
            help="Requires enable_branches — options map onto branches.",
        )
        enable_exploration_slot = st.checkbox(
            "enable_exploration_slot", value=base_config.enable_exploration_slot
        )

    try:
        ablation_config = AblationConfig(
            reasoning_mode=ReasoningMode.FULL,
            enable_portrait=enable_portrait,
            enable_concept_graph=enable_concept_graph,
            enable_diagnose=enable_diagnose,
            enable_planner=enable_planner,
            enable_branches=enable_branches,
            enable_options=enable_options,
            enable_exploration_slot=enable_exploration_slot,
        )
        config_error = None
    except ValueError as exc:
        ablation_config = None
        config_error = str(exc)

    if config_error:
        st.error(f"Invalid config: {config_error}")

if st.button("Start new session", type="primary", disabled=ablation_config is None):
    session_id = run_async(
        stores.transcript.create_session(
            learner.id, concept_graph_id=None, ablation_config=ablation_config
        )
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
    header = st.columns([3, 2, 1, 1])
    header[0].markdown("**Topic**")
    header[1].markdown("**Started**")
    header[2].markdown("**Turns**")
    for s in sessions:
        cols = st.columns([3, 2, 1, 1])
        cols[0].write(s.topic or "_(no topic attached)_")
        cols[1].write(s.created_at.strftime("%Y-%m-%d %H:%M"))
        cols[2].write(str(s.turn_count))
        if cols[3].button("Resume", key=f"resume-{s.session_id}"):
            st.session_state["session_id"] = s.session_id
            st.session_state["turn_index"] = s.turn_count
            st.session_state.pop("chat_history", None)
            st.session_state.pop("loop", None)
            st.switch_page("pages/1_Session.py")
