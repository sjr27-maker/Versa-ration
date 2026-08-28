-- probe: branches gain matched_via -- which channel resolved a
-- matched branch: 'option_click' (an unambiguous button click, no
-- interpretation) or 'text_match' (RESOLVE:MATCH's LLM judgment
-- against predicted_next_turn). NULL for anything not currently
-- matched (open/unmatched/superseded).
--
-- This exists so aggregate queries (BranchStore.match_rate_by_session_for_learner,
-- recurring_root_statements_for_learner) can report the two channels
-- as separate numbers -- a click-resolved match is evidence the
-- student picked an option the system offered, not evidence the
-- system predicted the student correctly, and blending the two into
-- one "match rate" would misrepresent what the number means.

ALTER TABLE branches ADD COLUMN matched_via TEXT;
