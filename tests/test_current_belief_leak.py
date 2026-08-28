"""check_current_belief_leak — the structural backstop for the exact
failure DerivePath's own prompt is meant to prevent: a predicted
*future* reaction (or the tutor's own not-yet-taught idea) being
promoted into a stated *current* belief. A heuristic (word overlap,
not semantic understanding), same "flag for a human to judge" spirit
as the redundancy check's own wording-not-semantics caveat.
"""

from probe.hypothesis_generator import check_current_belief_leak


def test_fires_when_belief_only_overlaps_the_predicted_reaction():
    current_belief = (
        "The student believes the whiteboard analogy translates to "
        "roommates acting as processors that read and write to a "
        "shared memory space."
    )
    predicted_next_turn = (
        "That whiteboard idea makes sense, so are the roommates acting "
        "like processors sharing the same memory?"
    )
    action_rationale = (
        "An analogy of roommates sharing a kitchen whiteboard helps "
        "explain shared memory architecture."
    )
    student_message = "what is parallel computing"

    assert check_current_belief_leak(
        current_belief, predicted_next_turn, action_rationale, student_message
    )


def test_does_not_fire_when_belief_is_grounded_in_the_students_message():
    current_belief = (
        "The student is asking a broad definitional question about "
        "parallel computing and has not yet expressed any specific "
        "misconception."
    )
    predicted_next_turn = (
        "That whiteboard idea makes sense, so are the roommates acting "
        "like processors sharing the same memory?"
    )
    action_rationale = "roommates sharing a kitchen whiteboard analogy"
    student_message = "what is parallel computing"

    assert not check_current_belief_leak(
        current_belief, predicted_next_turn, action_rationale, student_message
    )


def test_does_not_fire_on_the_insufficient_evidence_escape_hatch():
    current_belief = (
        "Insufficient evidence to characterize current belief beyond "
        "the root intent."
    )
    predicted_next_turn = (
        "That whiteboard idea makes sense, so are the roommates acting "
        "like processors sharing the same memory?"
    )
    action_rationale = "roommates sharing a kitchen whiteboard analogy"
    student_message = "what is parallel computing"

    assert not check_current_belief_leak(
        current_belief, predicted_next_turn, action_rationale, student_message
    )


def test_does_not_fire_when_belief_words_also_appear_in_student_message():
    """Real overlap with the prediction alone isn't automatically
    disqualifying if the same content is *also* actually in what the
    student said -- only "leaked from prediction, absent from the
    student's own words" should flag."""
    current_belief = "The student is asking about parallel computing directly."
    predicted_next_turn = "Something about parallel computing hardware."
    action_rationale = "unrelated analogy about kitchens"
    student_message = "what is parallel computing"

    assert not check_current_belief_leak(
        current_belief, predicted_next_turn, action_rationale, student_message
    )


def test_fires_when_belief_leaks_from_action_rationale_specifically():
    current_belief = "The student already understands the kitchen whiteboard concept."
    predicted_next_turn = "unrelated prediction text"
    action_rationale = "the kitchen whiteboard concept explains shared memory"
    student_message = "what is parallel computing"

    assert check_current_belief_leak(
        current_belief, predicted_next_turn, action_rationale, student_message
    )


def test_empty_current_belief_never_fires():
    assert not check_current_belief_leak(
        "", "roommates whiteboard analogy", "roommates whiteboard analogy", "hello"
    )
