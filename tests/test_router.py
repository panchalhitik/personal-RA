"""Step 3.2 — the router node. Every test mocks the Anthropic client; nothing here
touches the network."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from personal_ra.graph.build import build_graph
from personal_ra.graph.nodes import route_node
from personal_ra.graph.router import (
    HISTORY_TURNS,
    ROUTER_MODEL,
    build_classification_input,
    classify_route,
)
from personal_ra.graph.state import initial_state


def _response(route: str, reason: str = "because", in_tokens: int = 700, out_tokens: int = 40):
    """Shape of a forced-tool-use response: one tool_use block, no text."""
    block = SimpleNamespace(
        type="tool_use",
        name="classify_route",
        input={"route": route, "reason": reason},
    )
    return SimpleNamespace(
        content=[block],
        usage=SimpleNamespace(input_tokens=in_tokens, output_tokens=out_tokens),
    )


def _client(*responses):
    client = MagicMock()
    client.messages.create.side_effect = list(responses) or [_response("library")]
    return client


# --- each route fires on a clear-cut fixture question ------------------------------


CLEAR_CUT = [
    ("single_paper", "does their ablation support the main claim?", "8b8c5e05deed"),
    ("library", "which of my papers use contrastive loss?", None),
    ("web", "has anyone published a follow-up to this?", "8b8c5e05deed"),
    ("direct", "thanks — can you rephrase that more simply?", None),
]


@pytest.mark.parametrize("expected,question,paper_id", CLEAR_CUT)
def test_each_route_fires(expected, question, paper_id):
    client = _client(_response(expected, "clear-cut fixture"))
    route, reason, usage = classify_route(initial_state(question, paper_id), client=client)
    assert route == expected
    assert reason
    assert usage["cost_usd"] > 0


# --- the paper-open signal reaches the model ---------------------------------------


def test_deictic_question_with_paper_open_routes_to_single_paper():
    client = _client(_response("single_paper", "deictic 'their method' with a paper open"))
    route, _, _ = classify_route(
        initial_state("what does their method do about reward hacking?", "8b8c5e05deed"),
        client=client,
    )
    assert route == "single_paper"


def test_same_question_without_paper_routes_to_library():
    client = _client(_response("library", "no paper open, so single_paper is unavailable"))
    route, _, _ = classify_route(
        initial_state("what does their method do about reward hacking?"), client=client
    )
    assert route == "library"


def test_prompt_states_whether_a_paper_is_open():
    """The model can't apply the 'no paper means no single_paper' rule if we don't
    tell it. Assert both directions of the signal explicitly."""
    with_paper = build_classification_input(initial_state("q", "abc123"))
    without = build_classification_input(initial_state("q"))
    assert "abc123" in with_paper
    assert "none" in without and "single_paper is not available" in without


# --- history ----------------------------------------------------------------------


def test_history_reaches_the_classification_input():
    history = [
        {"role": "user", "content": "which papers cover sandbagging?"},
        {"role": "assistant", "content": "Two: Auditing Games, and Monitoring Monitorability."},
    ]
    prompt = build_classification_input(
        initial_state("what about the second one?", history=history)
    )
    assert "which papers cover sandbagging?" in prompt
    assert "Auditing Games" in prompt


def test_history_is_truncated_to_recent_turns():
    history = [{"role": "user", "content": f"turn {i}"} for i in range(20)]
    prompt = build_classification_input(initial_state("and then?", history=history))
    assert "turn 19" in prompt
    assert f"turn {20 - HISTORY_TURNS - 1}" not in prompt


def test_followup_routes_correctly_given_history():
    """'What about the second one?' is unroutable without history — assert the history
    is in the payload the client actually received, not just that a route came back."""
    history = [
        {"role": "user", "content": "which of my papers measure monitorability?"},
        {"role": "assistant", "content": "A Pragmatic Way to Measure CoT Monitorability."},
    ]
    client = _client(_response("library", "follow-up inherits the cross-paper subject"))
    route, _, _ = classify_route(
        initial_state("what about the second one?", history=history), client=client
    )
    assert route == "library"
    sent = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "measure monitorability" in sent


# --- request shape ----------------------------------------------------------------


def test_uses_haiku_with_forced_strict_tool_use():
    client = _client(_response("library"))
    classify_route(initial_state("q"), client=client)
    kwargs = client.messages.create.call_args.kwargs

    assert kwargs["model"] == ROUTER_MODEL == "claude-haiku-4-5"
    assert kwargs["temperature"] == 0
    assert kwargs["tool_choice"] == {"type": "tool", "name": "classify_route"}

    tool = kwargs["tools"][0]
    assert tool["strict"] is True
    assert tool["input_schema"]["additionalProperties"] is False
    assert tool["input_schema"]["properties"]["route"]["enum"] == [
        "single_paper",
        "library",
        "web",
        "direct",
    ]
    assert set(tool["input_schema"]["required"]) == {"route", "reason"}


# --- robustness -------------------------------------------------------------------


def test_explicit_route_bypasses_the_classifier():
    client = _client()
    route, reason, _ = classify_route(initial_state("q", route="web"), client=client)
    assert route == "web"
    assert "supplied by caller" in reason
    client.messages.create.assert_not_called()


def test_missing_tool_block_falls_back_instead_of_raising():
    """Forced tool_choice makes this unreachable in practice, but a router that raises
    takes the whole graph down with it."""
    empty = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="I'm not sure.")],
        usage=SimpleNamespace(input_tokens=700, output_tokens=5),
    )
    route, reason, _ = classify_route(initial_state("q", "abc123"), client=_client(empty))
    assert route == "single_paper"
    assert "fell back" in reason


# --- the node inside the graph ----------------------------------------------------


def test_route_node_records_reason_and_usage():
    client = _client(_response("direct", "conversational cue: 'thanks'"))
    delta = route_node(initial_state("thanks!"), client=client)
    assert delta["route"] == "direct"
    assert delta["route_reason"] == "conversational cue: 'thanks'"
    assert delta["usage"]["route"]["input_tokens"] == 700


def test_graph_routes_a_live_classification_to_the_right_branch():
    """End to end through the compiled graph with a mocked client — the classifier's
    answer, not a caller override, picks the branch."""
    graph = build_graph(client=_client(_response("direct", "greeting")))
    visited = [
        node
        for update in graph.stream(
            initial_state("hey, thanks for that"),
            {"configurable": {"thread_id": "t"}},
            stream_mode="updates",
        )
        for node in update
    ]
    assert visited == ["route", "direct"]
