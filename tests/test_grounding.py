"""Step 3.6 — the grounding check. Mocked client throughout; no network."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from conftest import FakeAnthropic, FakeAsyncAnthropic, FakeLibrary
from personal_ra.ask import REFUSAL as PAPER_REFUSAL
from personal_ra.graph.build import build_graph, sqlite_checkpointer
from personal_ra.graph.grounding import (
    CHECKER_MODEL,
    GROUNDED,
    MAX_ATTEMPTS,
    NOT_CHECKED,
    PARTIALLY_GROUNDED,
    UNGROUNDED,
    build_context,
    check_grounding,
    refused_by_grounding,
    refused_by_prefix,
    verdict_for,
)
from personal_ra.graph.nodes import after_grounding, grounding_node
from personal_ra.graph.state import initial_state
from personal_ra.search import LIBRARY_REFUSAL

CONTEXT = "<excerpt>roughly 250 poisoned documents compromise models of every size</excerpt>"


def _report(unsupported=(), substantive=True):
    block = SimpleNamespace(
        type="tool_use",
        name="report_grounding",
        input={
            "makes_substantive_claims": substantive,
            "unsupported_claims": list(unsupported),
        },
    )
    return SimpleNamespace(
        content=[block],
        usage=SimpleNamespace(input_tokens=1200, output_tokens=60),
    )


def _client(*responses):
    client = MagicMock()
    client.messages.create.side_effect = list(responses) or [_report()]
    return client


def citation(quote: str = "250 poisoned documents") -> dict:
    return {"quote": quote, "page": 3, "verified": True, "match_type": "exact"}


# --- catching invention -----------------------------------------------------------


def test_fabricated_claim_is_caught():
    fabricated = [{"claim": "the effect disappears above 70B parameters", "why": "no such finding"}]
    result = check_grounding(
        'The paper reports "250 poisoned documents" [p. 3], and the effect disappears above 70B.',
        CONTEXT,
        citations=[citation()],
        client=_client(_report(fabricated)),
    )
    assert result["verdict"] == PARTIALLY_GROUNDED
    assert result["unsupported"][0]["claim"].startswith("the effect disappears")


def test_fully_quoted_answer_passes():
    result = check_grounding(
        'The paper reports "250 poisoned documents" [p. 3].',
        CONTEXT,
        citations=[citation()],
        client=_client(_report()),
    )
    assert result["verdict"] == GROUNDED
    assert result["unsupported"] == []


def test_answer_resting_on_nothing_is_ungrounded():
    """No verified citation and an invented claim — there is nothing under it."""
    result = check_grounding(
        "Roughly 250 documents suffice, and the authors recommend gradient clipping.",
        CONTEXT,
        citations=[],
        client=_client(_report([{"claim": "recommend gradient clipping", "why": "absent"}])),
    )
    assert result["verdict"] == UNGROUNDED


def test_unverified_quote_alone_downgrades_the_verdict():
    """cite.py caught a fabricated quote; the claim checker found nothing wrong. The
    answer is still not fully grounded."""
    result = check_grounding(
        'It says "a quote that is not in the paper" [unverified].',
        CONTEXT,
        citations=[citation()],
        unverified=[{"quote": "a quote that is not in the paper", "verified": False}],
        client=_client(_report()),
    )
    assert result["verdict"] == PARTIALLY_GROUNDED
    assert result["unverified_quotes"] == 1


def test_verdict_table():
    assert verdict_for([], 0, 2, True) == GROUNDED
    assert verdict_for([], 0, 0, False) == GROUNDED  # a refusal invents nothing
    assert verdict_for([{"claim": "x", "why": "y"}], 0, 3, True) == PARTIALLY_GROUNDED
    assert verdict_for([{"claim": "x", "why": "y"}], 0, 0, True) == UNGROUNDED
    assert verdict_for([], 1, 2, True) == PARTIALLY_GROUNDED


# --- refusal correctness, both ways -----------------------------------------------


HEDGED = (
    "I could not find anything in the retrieved excerpts that speaks to ImageNet "
    "accuracy — these papers are about safety evaluations, not image classification."
)


def test_hedged_non_answer_is_a_correct_refusal_under_grounding_and_not_under_prefix():
    """The exact case the v2 README counts as a miss: a well-judged hedge scored as a
    failure purely because its wording differs from the fixed string."""
    result = check_grounding(HEDGED, CONTEXT, client=_client(_report(substantive=False)))

    assert refused_by_grounding(result) is True
    assert refused_by_prefix(HEDGED) is False


def test_the_fixed_strings_still_count_as_refusals_under_both_methods():
    for refusal in (PAPER_REFUSAL, LIBRARY_REFUSAL):
        result = check_grounding(refusal, CONTEXT, client=_client(_report(substantive=False)))
        assert refused_by_prefix(refusal) is True
        assert refused_by_grounding(result) is True


def test_an_invented_answer_is_not_a_refusal_under_either_method():
    invented = "These papers report 91.2% top-1 accuracy on ImageNet."
    result = check_grounding(
        invented,
        CONTEXT,
        client=_client(_report([{"claim": "91.2% top-1 on ImageNet", "why": "absent"}])),
    )
    assert refused_by_prefix(invented) is False
    assert refused_by_grounding(result) is False


def test_a_hedge_that_smuggles_in_a_claim_is_not_a_correct_refusal():
    """Declining and then inventing anyway must not score as a clean refusal."""
    result = check_grounding(
        "The papers do not cover this, though ImageNet accuracy is typically around 90%.",
        CONTEXT,
        client=_client(_report([{"claim": "around 90%", "why": "absent"}], substantive=False)),
    )
    assert refused_by_grounding(result) is False


# --- request shape and robustness -------------------------------------------------


def test_checker_uses_haiku_with_forced_tool_use():
    client = _client()
    check_grounding("an answer", CONTEXT, client=client)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == CHECKER_MODEL == "claude-haiku-4-5"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "report_grounding"}
    schema = kwargs["tools"][0]["input_schema"]
    assert kwargs["tools"][0]["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["properties"]["unsupported_claims"]["items"]["additionalProperties"] is False


def test_empty_answer_is_not_checked_and_costs_nothing():
    client = _client()
    result = check_grounding("   ", CONTEXT, client=client)
    assert result["verdict"] == NOT_CHECKED
    client.messages.create.assert_not_called()


def test_checker_failure_does_not_sink_the_run():
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("upstream exploded")
    result = check_grounding("an answer", CONTEXT, citations=[citation()], client=client)
    assert result["verdict"] == NOT_CHECKED
    assert result["error"] == "RuntimeError"


def test_build_context_marks_web_results_as_external():
    context = build_context(
        [{"text": "library text", "metadata": {"paper_title": "Paper A", "page": 2}}],
        [{"url": "https://example.com", "content": "web text"}],
    )
    assert "<excerpt" in context and "Paper A" in context
    assert "<web_result" in context and "https://example.com" in context


# --- the node and the regeneration cap --------------------------------------------


def test_node_records_verdict_attempt_and_usage():
    state = initial_state("q")
    state["answer"] = 'It reports "250 poisoned documents" [p. 3].'
    state["citations"] = [citation()]
    delta = grounding_node(state, client=_client(_report()))

    assert delta["grounding"]["verdict"] == GROUNDED
    assert delta["grounding"]["attempt"] == 1
    assert delta["usage"]["grounding_1"]["cost_usd"] > 0
    assert "usage" not in delta["grounding"]  # usage lives in one place, not two


def test_ungrounded_first_attempt_asks_for_a_regeneration():
    state = initial_state("q", route="library")
    state["answer"] = "an invented answer"
    delta = grounding_node(state, client=_client(_report([{"claim": "invented", "why": "absent"}])))
    assert delta["grounding"]["verdict"] == UNGROUNDED
    assert "answer" not in delta  # answer preserved for the retry
    assert after_grounding({**state, **delta}) == "generate"


def test_second_ungrounded_attempt_refuses_instead_of_answering():
    state = initial_state("q", route="library")
    state["answer"] = "an invented answer"
    state["grounding"] = {"attempt": 1, "verdict": UNGROUNDED}
    delta = grounding_node(state, client=_client(_report([{"claim": "invented", "why": "absent"}])))
    assert delta["answer"] == LIBRARY_REFUSAL
    assert delta["grounding"]["refused_after_retry"] is True
    assert after_grounding({**state, **delta}) == "end"


def test_refusal_string_matches_the_route():
    state = initial_state("q", paper_id="p1", route="single_paper")
    state["answer"] = "an invented answer"
    state["grounding"] = {"attempt": 1, "verdict": UNGROUNDED}
    delta = grounding_node(state, client=_client(_report([{"claim": "invented", "why": "absent"}])))
    assert delta["answer"] == PAPER_REFUSAL


def test_regeneration_fires_once_and_only_once():
    starved = {"verdict": UNGROUNDED}
    assert (
        after_grounding({"route": "library", "grounding": {**starved, "attempt": 1}}) == "generate"
    )
    assert after_grounding({"route": "library", "grounding": {**starved, "attempt": 2}}) == "end"
    assert after_grounding({"route": "library", "grounding": {**starved, "attempt": 3}}) == "end"
    assert MAX_ATTEMPTS == 2


def test_single_paper_regenerates_through_its_own_generator():
    """Sending a single-paper answer to the library generator would change the
    question it was answering."""
    state = {"route": "single_paper", "grounding": {"verdict": UNGROUNDED, "attempt": 1}}
    assert after_grounding(state) == "single_paper"


def test_partially_grounded_does_not_regenerate():
    state = {
        "route": "library",
        "grounding": {"verdict": PARTIALLY_GROUNDED, "attempt": 1},
    }
    assert after_grounding(state) == "end"


def test_graph_still_terminates_with_the_regeneration_edge_wired(tmp_path):
    graph = build_graph(
        checkpointer=sqlite_checkpointer(tmp_path / "g.db"),
        client=FakeAnthropic(),
        async_client=FakeAsyncAnthropic(),
        library=FakeLibrary([]),
    )
    config = {"configurable": {"thread_id": "t"}}
    visited = [
        node
        for update in graph.stream(
            initial_state("q", route="library"), config, stream_mode="updates"
        )
        for node in update
    ]
    assert visited[-1] == "grounding"
    # generate is real now, so grounding audits an actual answer rather than
    # short-circuiting on an empty one.
    final = graph.get_state(config).values
    assert final["grounding"]["verdict"] == GROUNDED
    assert final["grounding"]["attempt"] == 1  # no regeneration on a clean verdict
