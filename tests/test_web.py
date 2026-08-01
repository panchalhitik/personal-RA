"""Step 3.7 — approval gate before web search. No network, no Tavily key needed."""

from __future__ import annotations

import pytest
from langgraph.types import Command

from personal_ra.graph.build import build_graph, pending_approval, sqlite_checkpointer
from personal_ra.graph.nodes import after_approval
from personal_ra.graph.state import initial_state
from personal_ra.graph.web import (
    CREDITS_PER_SEARCH,
    MAX_RESULTS,
    SEARCH_DEPTH,
    MissingTavilyKey,
    approval_payload,
    format_web_context,
    interpret_decision,
    search_web,
    web_citation,
)

TAVILY_RESPONSE = {
    "results": [
        {
            "title": "A follow-up to the poisoning paper",
            "url": "https://arxiv.org/abs/2601.00001",
            "content": "A 2026 replication extends the result to 70B models.",
            "score": 0.94,
            # Tavily does not send this today, but if it ever did, a web snippet
            # must still never end up looking like a verified library quote.
            "page": 7,
        }
    ]
}


class FakeTavily:
    def __init__(self, response=None, fail=False):
        self.response = TAVILY_RESPONSE if response is None else response
        self.fail = fail
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        if self.fail:
            raise RuntimeError("tavily is down")
        return self.response


def _graph(tmp_path, web_client=None):
    return build_graph(
        checkpointer=sqlite_checkpointer(tmp_path / "g.db"),
        web_client=web_client or FakeTavily(),
    )


# --- the interrupt ----------------------------------------------------------------


def test_graph_interrupts_before_web_search_and_does_not_proceed(tmp_path):
    client = FakeTavily()
    graph = _graph(tmp_path, client)
    config = {"configurable": {"thread_id": "t"}}

    graph.invoke(initial_state("is there a newer version?", route="web"), config)

    assert client.calls == []  # nothing was searched, nothing was spent
    assert graph.get_state(config).next == ("approve",)
    assert graph.get_state(config).values["answer"] == ""


def test_interrupt_payload_shows_query_reason_and_cost(tmp_path):
    graph = _graph(tmp_path)
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke(initial_state("has anyone published a follow-up?", route="web"), config)

    payload = pending_approval(graph, config)
    assert payload["query"] == "has anyone published a follow-up?"
    assert payload["reason"]  # the router's route_reason
    assert payload["provider"] == "tavily"
    assert str(CREDITS_PER_SEARCH) in payload["estimated_cost"]


def test_pending_approval_is_none_when_the_graph_is_not_waiting(tmp_path):
    graph = _graph(tmp_path)
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke(initial_state("thanks!", route="direct"), config)
    assert pending_approval(graph, config) is None


# --- the two cycles ---------------------------------------------------------------


def test_approve_cycle_proceeds_to_the_web(tmp_path):
    client = FakeTavily()
    graph = _graph(tmp_path, client)
    config = {"configurable": {"thread_id": "approve"}}

    graph.invoke(initial_state("is there a newer version?", route="web"), config)
    graph.invoke(Command(resume="approve"), config)

    final = graph.get_state(config).values
    assert final["approved"] is True
    assert len(client.calls) == 1
    assert client.calls[0]["search_depth"] == SEARCH_DEPTH == "advanced"
    assert client.calls[0]["max_results"] == MAX_RESULTS == 5
    assert final["web_results"][0]["url"] == "https://arxiv.org/abs/2601.00001"


def test_deny_cycle_falls_back_to_the_library(tmp_path):
    """Declining the web search declines the spend, not the question."""
    client = FakeTavily()
    graph = _graph(tmp_path, client)
    config = {"configurable": {"thread_id": "deny"}}

    graph.invoke(initial_state("is there a newer version?", route="web"), config)
    visited = [
        node
        for update in graph.stream(Command(resume="deny"), config, stream_mode="updates")
        for node in update
    ]

    assert client.calls == []  # nothing spent
    assert graph.get_state(config).values["approved"] is False
    assert "retrieve" in visited and "web_search" not in visited
    assert visited[-1] == "grounding"


# --- persistence across a process restart -----------------------------------------


def test_pending_approval_survives_losing_the_graph_object(tmp_path):
    """The thing checkpointers are for: the graph object is discarded entirely and a
    fresh one built from the same file still knows it is waiting, and on what."""
    db = tmp_path / "g.db"
    config = {"configurable": {"thread_id": "restart"}}

    writer = build_graph(checkpointer=sqlite_checkpointer(db), web_client=FakeTavily())
    writer.invoke(initial_state("has anyone published a follow-up?", route="web"), config)
    del writer

    client = FakeTavily()
    reader = build_graph(checkpointer=sqlite_checkpointer(db), web_client=client)
    payload = pending_approval(reader, config)
    assert payload is not None
    assert payload["query"] == "has anyone published a follow-up?"

    reader.invoke(Command(resume="approve"), config)
    assert len(client.calls) == 1
    assert reader.get_state(config).values["web_results"]


# --- web results never look like library citations --------------------------------


def test_web_citation_cannot_carry_a_page_number():
    """Built by whitelisting fields, so a stray upstream `page` cannot leak through
    and make a web snippet look like a verified quote."""
    citation = web_citation(TAVILY_RESPONSE["results"][0])
    assert "page" not in citation
    assert citation["source"] == "web"
    assert citation["verified"] is False
    assert citation["url"] == "https://arxiv.org/abs/2601.00001"


def test_search_results_carry_a_url_and_no_page():
    results = search_web("q", client=FakeTavily())
    assert results[0]["source"] == "web"
    assert "page" not in results[0]
    assert results[0]["url"]


def test_web_results_in_the_graph_never_acquire_a_page(tmp_path):
    graph = _graph(tmp_path)
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke(initial_state("newer version?", route="web"), config)
    graph.invoke(Command(resume=True), config)

    for result in graph.get_state(config).values["web_results"]:
        assert "page" not in result
        assert result["source"] == "web"


def test_web_context_is_labelled_distinctly_from_library_excerpts():
    context = format_web_context(search_web("q", client=FakeTavily()))
    assert "<web_result" in context
    assert "<excerpt" not in context
    assert "https://arxiv.org/abs/2601.00001" in context


# --- decisions and failures -------------------------------------------------------


@pytest.mark.parametrize(
    "decision,expected",
    [
        (True, True),
        ("approve", True),
        ("yes", True),
        ({"approved": True}, True),
        (False, False),
        ("deny", False),
        ("no", False),
        ({"approved": False}, False),
        (None, False),
        ("", False),
        ("maybe", False),
        ({"unrelated": "junk"}, False),
    ],
)
def test_decision_shapes(decision, expected):
    assert interpret_decision(decision) is expected


def test_unrecognised_decisions_deny_rather_than_spend():
    """Defaulting an unclear answer to 'yes, spend money and leave the library' is
    the wrong way to be wrong."""
    assert interpret_decision(object()) is False


def test_after_approval_branches_on_the_decision():
    assert after_approval({"approved": True}) == "web_search"
    assert after_approval({"approved": False}) == "retrieve"
    assert after_approval({}) == "retrieve"


def test_search_failure_returns_nothing_rather_than_raising():
    """The spend already happened; a dead search should degrade the answer, not
    destroy the run."""
    assert search_web("q", client=FakeTavily(fail=True)) == []


def test_missing_key_explains_how_to_fix_it():
    import personal_ra.graph.web as web

    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(MissingTavilyKey) as excinfo:
            web.search_web("q")
    assert "TAVILY_API_KEY" in str(excinfo.value)
    assert "app.tavily.com" in str(excinfo.value)


def test_approval_payload_carries_the_original_question_too():
    """After two rewrites the live query can look nothing like what was asked."""
    state = initial_state("has anyone followed this up?", route="web")
    state["question"] = "poisoning follow-up replication 2026 70B"
    payload = approval_payload(state)
    assert payload["query"] == "poisoning follow-up replication 2026 70B"
    assert payload["original_question"] == "has anyone followed this up?"
