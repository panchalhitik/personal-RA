"""Step 3.5 — the rewrite node. Mocked client throughout; no network."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from conftest import FakeAnthropic, FakeAsyncAnthropic, FakeLibrary
from personal_ra.graph.build import build_graph, sqlite_checkpointer
from personal_ra.graph.nodes import after_grade, rewrite_node
from personal_ra.graph.rewrite import (
    MAX_REJECTED_IN_PROMPT,
    REWRITER_MODEL,
    build_rewrite_input,
    rewrite_query,
)
from personal_ra.graph.state import MAX_REWRITES, MIN_GRADED_CHUNKS, initial_state

BETTER = "sandbagging deliberate underperformance capability evaluation auditing game"


def _response(query: str = BETTER, rationale: str = "narrowed to named phenomenon"):
    block = SimpleNamespace(
        type="tool_use",
        name="rewrite_query",
        input={"query": query, "rationale": rationale},
    )
    return SimpleNamespace(
        content=[block],
        usage=SimpleNamespace(input_tokens=900, output_tokens=30),
    )


def _client(*responses):
    client = MagicMock()
    client.messages.create.side_effect = list(responses) or [_response()]
    return client


def rejected(text: str, reason: str, title: str = "Auditing Games", cid: str = "c1") -> dict:
    return {
        "id": cid,
        "text": text,
        "metadata": {"paper_id": "p1", "paper_title": title, "page": 4},
        "score": 0.3,
        "grade_reason": reason,
    }


# --- the prompt gets the signal it needs ------------------------------------------


def test_rejected_chunks_and_their_reasons_reach_the_prompt():
    """§3.5: the rejected chunks are the useful signal — they reveal the corpus
    vocabulary, and the grader's reason says what each one was about instead."""
    chunks = [
        rejected("we survey prior work on model evaluations", "surveys related work, not results"),
        rejected(
            "the appendix lists hyperparameters", "hyperparameter table, no findings", cid="c2"
        ),
    ]
    prompt = build_rewrite_input("does the model hide its abilities?", "hide abilities", chunks)
    assert "we survey prior work" in prompt
    assert "surveys related work, not results" in prompt
    assert "hyperparameter table" in prompt
    assert "Auditing Games" in prompt


def test_prompt_carries_the_original_question_and_the_failed_query():
    prompt = build_rewrite_input("does the model hide its abilities?", "hide abilities", [])
    assert "does the model hide its abilities?" in prompt
    assert "hide abilities" in prompt


def test_prompt_is_bounded_when_many_chunks_were_rejected():
    """Eight full chunks would crowd out the instructions; the grader's reason
    carries most of the signal anyway."""
    chunks = [rejected("x" * 3000, "off topic", cid=f"c{i}") for i in range(20)]
    prompt = build_rewrite_input("q", "q", chunks)
    assert prompt.count("rejected:") == MAX_REJECTED_IN_PROMPT
    assert "x" * 3000 not in prompt


def test_empty_retrieval_is_stated_rather_than_left_blank():
    prompt = build_rewrite_input("q", "q", [])
    assert "nothing at all" in prompt


def test_rewriter_uses_haiku_with_forced_tool_use():
    client = _client()
    rewrite_query("q", "q", [], client=client)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == REWRITER_MODEL == "claude-haiku-4-5"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "rewrite_query"}
    assert kwargs["tools"][0]["strict"] is True


def test_system_prompt_pushes_toward_specificity():
    """The BM25 finding says generalising retrieves worse in this corpus. If that
    instruction ever gets softened, this test should be the thing that objects."""
    from personal_ra.graph.rewrite import REWRITER_SYSTEM

    assert "MORE specific, never more general" in REWRITER_SYSTEM


# --- the returned rewrite ---------------------------------------------------------


def test_rewrite_returns_query_and_rationale():
    query, rationale, usage = rewrite_query("q", "q", [], client=_client())
    assert query == BETTER
    assert rationale == "narrowed to named phenomenon"
    assert usage["cost_usd"] > 0


def test_rewrite_failure_leaves_the_query_unchanged():
    """A failed rewrite must not sink a run that still has chunks to answer from."""
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("upstream exploded")
    query, rationale, _ = rewrite_query("original", "current", [], client=client)
    assert query == "current"
    assert "rewrite failed" in rationale


def test_empty_rewrite_is_ignored():
    query, rationale, _ = rewrite_query("original", "current", [], client=_client(_response("   ")))
    assert query == "current"
    assert "no query" in rationale


# --- the node ---------------------------------------------------------------------


def test_rewrite_node_replaces_question_but_not_original_question():
    state = initial_state("does the model hide its abilities?")
    state["rejected_chunks"] = [rejected("prior work survey", "related work")]
    delta = rewrite_node(state, client=_client())

    assert delta["question"] == BETTER
    assert "original_question" not in delta  # never touched
    assert state["original_question"] == "does the model hide its abilities?"
    assert delta["rewrite_reason"] == "narrowed to named phenomenon"
    assert delta["rewrite_count"] == 1


def test_each_rewrite_keeps_its_own_usage_entry():
    """§3.8 reports the rewrite loop's cost, so a second rewrite must not overwrite
    the first one's tokens."""
    state = initial_state("q")
    first = rewrite_node(state, client=_client())
    second = rewrite_node({**state, **first}, client=_client())
    assert "rewrite_1" in first["usage"]
    assert {"rewrite_1", "rewrite_2"} <= set(second["usage"])


# --- when it fires, and when it stops ---------------------------------------------


def test_fires_below_the_threshold_and_not_above():
    below = {"graded_chunks": [{"id": "c1"}], "rewrite_count": 0}
    at = {"graded_chunks": [{"id": "c1"}, {"id": "c2"}], "rewrite_count": 0}
    assert len(below["graded_chunks"]) < MIN_GRADED_CHUNKS
    assert after_grade(below) == "rewrite"
    assert after_grade(at) == "generate"


def test_cap_holds_at_two():
    starved = [{"id": "c1"}]
    assert after_grade({"graded_chunks": starved, "rewrite_count": MAX_REWRITES - 1}) == "rewrite"
    assert after_grade({"graded_chunks": starved, "rewrite_count": MAX_REWRITES}) == "generate"
    assert after_grade({"graded_chunks": starved, "rewrite_count": MAX_REWRITES + 1}) == "generate"


def _starved_graph(client, db):
    """An empty library starves grading on every pass, which is what drives the loop.
    All the other nodes are real now, so they need doubles too."""
    return build_graph(
        checkpointer=sqlite_checkpointer(db),
        client=client,
        async_client=FakeAsyncAnthropic(),
        library=FakeLibrary([]),
    )


def test_loop_terminates_and_the_query_evolves_each_pass(tmp_path):
    """Retrieval finds nothing, so grading starves every cycle — the loop must still
    stop at the cap, and each pass must feed the rewriter forward."""
    client = FakeAnthropic(rewrite=["first rewrite", "second rewrite"])
    graph = _starved_graph(client, tmp_path / "g.db")
    config = {"configurable": {"thread_id": "loop"}}

    visited = [
        node
        for update in graph.stream(
            initial_state("original question", route="library"), config, stream_mode="updates"
        )
        for node in update
    ]
    final = graph.get_state(config).values

    assert visited.count("rewrite") == MAX_REWRITES
    assert final["rewrite_count"] == MAX_REWRITES
    assert final["question"] == "second rewrite"
    assert final["original_question"] == "original question"
    assert final["rewrite_reason"] == "fake rewrite 2"
    assert visited[-1] == "grounding"


def test_second_rewrite_sees_the_first_rewritten_query():
    """Otherwise the loop just re-asks the same failed question twice."""
    client = FakeAnthropic(rewrite=["first rewrite", "second rewrite"])
    graph = _starved_graph(client, ":memory:")
    graph.invoke(
        initial_state("original question", route="library"), {"configurable": {"thread_id": "t"}}
    )
    rewrite_calls = client.calls_for("rewrite_query")
    assert len(rewrite_calls) == MAX_REWRITES
    second_prompt = rewrite_calls[1]["messages"][0]["content"]
    assert "first rewrite" in second_prompt
    assert "original question" in second_prompt  # intent is still anchored
