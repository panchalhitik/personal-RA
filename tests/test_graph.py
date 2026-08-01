"""Step 3.1 — the skeleton: it compiles, every node is reachable, all four routes
traverse to END, and state survives a checkpointer round-trip."""

from __future__ import annotations

from collections import defaultdict, deque

import pytest
from langgraph.graph import END, START

from personal_ra.graph.build import NODE_NAMES, build_graph, sqlite_checkpointer
from personal_ra.graph.nodes import after_grade
from personal_ra.graph.state import MAX_REWRITES, initial_state

ROUTES = ["single_paper", "library", "web", "direct"]


@pytest.fixture
def graph():
    return build_graph()


def _run(graph, question, route=None, paper_id=None, thread_id="t1"):
    """Run to completion, returning (final_state, ordered node names visited)."""
    state = initial_state(question, paper_id, thread_id, route)
    config = {"configurable": {"thread_id": thread_id}}
    visited = [
        node for update in graph.stream(state, config, stream_mode="updates") for node in update
    ]
    return graph.get_state(config).values if graph.checkpointer else None, visited


def test_graph_compiles(graph):
    assert graph is not None


def test_every_node_is_reachable_from_start(graph):
    """BFS the compiled edge list — an orphaned node is a wiring bug the stubs hide."""
    edges = defaultdict(set)
    for edge in graph.get_graph().edges:
        edges[edge.source].add(edge.target)

    seen, queue = set(), deque([START])
    while queue:
        node = queue.popleft()
        for target in edges[node] - seen:
            seen.add(target)
            queue.append(target)

    assert set(NODE_NAMES) - seen == set()
    assert END in seen


@pytest.mark.parametrize("route", ROUTES)
def test_each_route_traverses_to_end(route):
    graph = build_graph()
    _, visited = _run(graph, "q", route=route)
    assert visited[0] == "route"
    expected_first_branch = {
        "single_paper": "single_paper",
        "library": "retrieve",
        "web": "approve",
        "direct": "direct",
    }[route]
    assert visited[1] == expected_first_branch


def test_library_route_visits_the_retrieval_chain():
    graph = build_graph()
    _, visited = _run(graph, "q", route="library")
    for node in ["retrieve", "rerank", "grade", "generate", "grounding"]:
        assert node in visited


def test_web_route_goes_through_approval_before_search():
    graph = build_graph()
    _, visited = _run(graph, "q", route="web")
    assert visited.index("approve") < visited.index("web_search")
    assert visited.index("web_search") < visited.index("generate")


def test_rewrite_loop_terminates_at_the_cap():
    """Stub retrieval returns nothing, so grading always falls short — the loop must
    still stop, and stop at exactly MAX_REWRITES."""
    graph = build_graph(checkpointer=sqlite_checkpointer(":memory:"))
    final, visited = _run(graph, "q", route="library")
    assert visited.count("rewrite") == MAX_REWRITES
    assert final["rewrite_count"] == MAX_REWRITES
    assert visited[-1] == "grounding"


def test_after_grade_branches_on_chunk_count_and_cap():
    enough = [{"id": "a"}, {"id": "b"}]
    assert after_grade({"graded_chunks": enough, "rewrite_count": 0}) == "generate"
    assert after_grade({"graded_chunks": [{"id": "a"}], "rewrite_count": 0}) == "rewrite"
    assert after_grade({"graded_chunks": [], "rewrite_count": MAX_REWRITES}) == "generate"


def test_router_stub_defaults_on_paper_id():
    graph = build_graph(checkpointer=sqlite_checkpointer(":memory:"))
    with_paper, _ = _run(graph, "does their ablation hold?", paper_id="abc123")
    no_paper, _ = _run(graph, "does their ablation hold?", thread_id="t2")
    assert with_paper["route"] == "single_paper"
    assert no_paper["route"] == "library"
    assert with_paper["route_reason"]  # a reason is recorded on every path


def test_state_survives_a_checkpointer_round_trip(tmp_path):
    """Write with one graph object, read with a fresh one built from the same file —
    this is the property step 3.7's process-restart test depends on."""
    db = tmp_path / "graph.db"
    config = {"configurable": {"thread_id": "conv-42"}}

    writer = build_graph(checkpointer=sqlite_checkpointer(db))
    writer.invoke(initial_state("what is sandbagging?", "paper-7", "conv-42", "library"), config)

    reader = build_graph(checkpointer=sqlite_checkpointer(db))
    restored = reader.get_state(config).values
    assert restored["original_question"] == "what is sandbagging?"
    assert restored["paper_id"] == "paper-7"
    assert restored["route"] == "library"
    assert restored["rewrite_count"] == MAX_REWRITES


def test_threads_are_isolated(tmp_path):
    db = tmp_path / "graph.db"
    graph = build_graph(checkpointer=sqlite_checkpointer(db))
    graph.invoke(
        initial_state("first question", thread_id="a", route="direct"),
        {"configurable": {"thread_id": "a"}},
    )
    graph.invoke(
        initial_state("second question", thread_id="b", route="direct"),
        {"configurable": {"thread_id": "b"}},
    )
    a = graph.get_state({"configurable": {"thread_id": "a"}}).values
    b = graph.get_state({"configurable": {"thread_id": "b"}}).values
    assert a["question"] == "first question"
    assert b["question"] == "second question"
