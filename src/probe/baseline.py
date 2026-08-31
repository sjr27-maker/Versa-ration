"""The plain-LLM BASELINE — one call per turn, no reasoning
scaffolding, the floor every other configuration is measured against.

Split out of the old `nodes.py` (deleted with the full reasoning path)
so `SessionLoop._handle_bypass_turn` and the Compare page still have it.
`MAX_CALLS_PER_TURN` lives here too now: it is the only constant the
surviving loop still needs from that module.
"""

from __future__ import annotations

from probe.llm import LLMClient

# Per-turn LLM-call guardrail, checked in SessionLoop against the
# call-count instrumentation on every LLM-calling node. A loud-warning
# guardrail, not a hard stop: crossing it logs a warning and the turn
# continues. 30 is an arbitrary starting point, not a measured budget;
# minimal_branch turns cost at most ~7 calls (memory pre-check +
# assess + options/answer + fact write), so this is comfortably slack
# for that mode and still catches a runaway.
MAX_CALLS_PER_TURN = 30


class BaselineTeach:
    """The true plain-LLM baseline (see loop.py's `_handle_bypass_turn`):
    one call, no memory, no branches -- just the student's message plus
    this session's prior turns as context. Deliberately does not reuse
    any architecture-specific scaffolding; it DOES reuse the plain
    output-shape instructions (no JSON wrapper, no gratuitous headers,
    never close by asking how the student feels) since those are
    prompt-writing hygiene, not architecture, and withholding them
    would confound the comparison this node exists to make honest.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        # Read by SessionLoop into the MAX_CALLS_PER_TURN accounting,
        # same convention as every other node.
        self.last_call_count: int = 0

    async def run(self, turn_text: str, prior_turns: list[str]) -> str:
        self.last_call_count = 0
        history = (
            "\n".join(f"student: {t}" for t in prior_turns)
            if prior_turns
            else "(no prior turns)"
        )
        prompt = (
            # The "BASELINE:TEACH" prefix matters, not just for logging:
            # llm.py's GeminiLLMClient dispatches structured-output
            # config by longest matching prefix, and an unrecognized
            # prefix defaults to forcing response_mime_type=application/
            # json at the API level -- no prompt text can override that
            # after the fact. TEACH: is registered free-text for the
            # same reason; this prefix must be too.
            "BASELINE:TEACH\n"
            "You are a tutor having a conversation with a student. "
            "Respond directly and helpfully to their latest message.\n\n"
            f"Prior turns in this conversation:\n{history}\n\n"
            f"Student's latest message: {turn_text}\n\n"
            "Lead with the direct answer or key idea — do not open "
            "with setup or a restatement of the question. Do not "
            "partition the response into steps or add headers/numbered "
            "lists unless the content genuinely requires that "
            "structure.\n"
            "Never end by asking how the student feels, what they "
            "prefer, or what kind of learner they are.\n"
            "Respond with plain prose only — never wrap your answer in "
            "JSON or any other structured/markup format."
        )
        result = await self._llm.complete(prompt)
        self.last_call_count += 1
        return result
