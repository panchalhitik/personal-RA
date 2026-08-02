"""Step 3.9 — per-node tracing. No Langfuse, no Docker, no network."""

from __future__ import annotations

import pytest
from langgraph.types import Command

from conftest import FakeAnthropic, FakeAsyncAnthropic, FakeLibrary
from personal_ra.graph.build import build_graph, sqlite_checkpointer
from personal_ra.graph.state import initial_state
from personal_ra.graph.tracing import (
    PUBLIC_KEY_ENV,
    LangfuseTracer,
    NullTracer,
    make_tracer,
    node_usage,
    safe_attributes,
    traced,
)


class FakeObservation:
    def __init__(self, name, kwargs):
        self.name = name
        self.created_with = kwargs
        self.updates = []
        self.ended = False

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        self.ended = True

    @property
    def final(self) -> dict:
        merged = {}
        for u in self.updates:
            merged.update({k: v for k, v in u.items() if v is not None})
        return merged


class FakeLangfuse:
    def __init__(self):
        self.observations: list[FakeObservation] = []
        self.flushed = 0

    def create_trace_id(self, seed=None):
        return f"trace-{abs(hash(seed)) % 10000:04d}"

    def start_observation(self, **kwargs):
        obs = FakeObservation(kwargs.get("name"), kwargs)
        self.observations.append(obs)
        return obs

    def flush(self):
        self.flushed += 1


def _tracer():
    client = FakeLangfuse()
    return LangfuseTracer(client), client


# --- unconfigured is a no-op ------------------------------------------------------


def test_no_key_gives_a_null_tracer(monkeypatch):
    """Someone cloning the repo must not need Docker to ask a question."""
    monkeypatch.delenv(PUBLIC_KEY_ENV, raising=False)
    tracer = make_tracer()
    assert isinstance(tracer, NullTracer)
    assert tracer.enabled is False


def test_the_null_tracer_returns_the_node_unwrapped():
    """Not merely disabled — the wrapper is not applied at all, so a traced graph
    and an untraced one execute identical code."""

    def node(state):
        return {"answer": "x"}

    assert traced("generate", node, NullTracer()) is node


def test_a_null_span_still_supports_the_record_protocol():
    with NullTracer().span("route", {}) as record:
        record.set({"route": "library"})  # must not raise


def test_flush_on_the_null_tracer_is_harmless():
    NullTracer().flush()


# --- spans per node ---------------------------------------------------------------


def test_a_span_is_emitted_per_node():
    tracer, client = _tracer()

    def node(state):
        return {"route": "library"}

    traced("route", node, tracer)(initial_state("q"))
    assert [o.name for o in client.observations] == ["route"]
    assert client.observations[0].ended is True


def test_the_span_records_latency_route_and_tokens():
    tracer, client = _tracer()

    def node(state):
        return {
            "route": "library",
            "usage": {"route": {"input_tokens": 700, "output_tokens": 40, "cost_usd": 0.0012}},
        }

    traced("route", node, tracer)(initial_state("q"))
    final = client.observations[0].final
    assert final["metadata"]["latency_ms"] >= 0
    assert final["metadata"]["route"] == "library"
    assert final["usage_details"] == {"input": 700, "output": 40}
    assert final["cost_details"] == {"total": 0.0012}


def test_cache_tokens_are_recorded_for_the_single_paper_cache_hit_rate():
    """§3.9 asks for the cache hit rate on single_paper specifically, which is only
    computable if both halves of the cache split reach the span."""
    tracer, client = _tracer()

    def node(state):
        return {
            "usage": {
                "single_paper": {
                    "input_tokens": 25,
                    "output_tokens": 300,
                    "cache_write_tokens": 0,
                    "cache_read_tokens": 17152,
                    "cost_usd": 0.02,
                }
            }
        }

    traced("single_paper", node, tracer)(initial_state("q"))
    metadata = client.observations[0].final["metadata"]
    assert metadata["cache_read_tokens"] == 17152
    assert metadata["cache_write_tokens"] == 0


def test_nodes_of_one_question_share_a_trace():
    tracer, client = _tracer()
    state = initial_state("what is sandbagging?", thread_id="t1")
    for node in ("route", "retrieve", "generate"):
        traced(node, lambda s: {}, tracer)(state)
    trace_ids = {o.created_with["trace_context"]["trace_id"] for o in client.observations}
    assert len(trace_ids) == 1


def test_a_different_question_gets_a_different_trace():
    """Seeded on thread AND question, so a follow-up does not append its spans to
    the previous answer's trace."""
    tracer, client = _tracer()
    traced("route", lambda s: {}, tracer)(initial_state("first", thread_id="t1"))
    traced("route", lambda s: {}, tracer)(initial_state("second", thread_id="t1"))
    trace_ids = {o.created_with["trace_context"]["trace_id"] for o in client.observations}
    assert len(trace_ids) == 2


def test_node_usage_finds_suffixed_entries():
    """rewrite and grounding suffix their usage keys with an attempt number."""
    usage = {"route": {"cost_usd": 1}, "rewrite_2": {"cost_usd": 2}, "grounding_1": {"cost_usd": 3}}
    assert node_usage(usage, "route") == {"cost_usd": 1}
    assert node_usage(usage, "rewrite") == {"cost_usd": 2}
    assert node_usage(usage, "grounding") == {"cost_usd": 3}
    assert node_usage(usage, "generate") == {}


# --- failures and interrupts ------------------------------------------------------


def test_a_failing_node_marks_the_span_and_re_raises():
    tracer, client = _tracer()

    def node(state):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        traced("generate", node, tracer)(initial_state("q"))
    assert client.observations[0].final["level"] == "ERROR"
    assert client.observations[0].ended is True


def test_an_approval_interrupt_is_not_an_error():
    """GraphInterrupt is the gate working. Marking it ERROR would paint every
    approval request red in the dashboard."""
    from langgraph.errors import GraphInterrupt

    tracer, client = _tracer()

    def node(state):
        raise GraphInterrupt(())

    with pytest.raises(GraphInterrupt):
        traced("approve", node, tracer)(initial_state("q"))
    assert "level" not in client.observations[0].final
    assert client.observations[0].ended is True


# --- what may leave the process ---------------------------------------------------


def test_span_attributes_are_whitelisted_not_copied():
    """A tracer that ships whatever is in scope is one refactor away from shipping
    a key. Anything not named in the whitelist must not appear."""
    state = initial_state("a question", paper_id="p1", thread_id="t1")
    state["api_key"] = "sk-ant-super-secret"
    state["answer"] = "a long answer body"
    state["history"] = [{"role": "user", "content": "earlier turn"}]

    attrs = safe_attributes(state)
    serialised = repr(attrs)
    assert "sk-ant-super-secret" not in serialised
    assert "a long answer body" not in serialised
    assert "earlier turn" not in serialised
    assert "api_key" not in attrs


def test_bulk_fields_are_counted_rather_than_copied():
    state = initial_state("q")
    state["chunks"] = [{"text": "a very long chunk body"} for _ in range(8)]
    attrs = safe_attributes(state)
    assert attrs["n_chunks"] == 8
    assert "a very long chunk body" not in repr(attrs)


def test_secrets_do_not_reach_the_span_through_the_graph():
    tracer, client = _tracer()

    def node(state):
        return {"answer": "SECRET-ANSWER-BODY", "route": "library"}

    traced("generate", node, tracer)(initial_state("q"))
    assert "SECRET-ANSWER-BODY" not in repr(
        [o.created_with for o in client.observations] + [o.updates for o in client.observations]
    )


# --- through the compiled graph ---------------------------------------------------


def test_a_traced_graph_emits_one_span_per_visited_node(tmp_path):
    tracer, client = _tracer()
    graph = build_graph(
        checkpointer=sqlite_checkpointer(tmp_path / "g.db"),
        client=FakeAnthropic(),
        async_client=FakeAsyncAnthropic(),
        library=FakeLibrary(),
        tracer=tracer,
    )
    graph.invoke(initial_state("q", route="library"), {"configurable": {"thread_id": "t"}})
    names = [o.name for o in client.observations]
    assert names == ["route", "retrieve", "rerank", "grade", "generate", "grounding"]


def test_the_approval_gate_traces_without_an_error_span(tmp_path):
    tracer, client = _tracer()
    graph = build_graph(
        checkpointer=sqlite_checkpointer(tmp_path / "g.db"),
        client=FakeAnthropic(),
        async_client=FakeAsyncAnthropic(),
        library=FakeLibrary(),
        tracer=tracer,
    )
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke(initial_state("newer version?", route="web"), config)
    graph.invoke(Command(resume="deny"), config)
    assert all("level" not in o.final for o in client.observations)
    assert "approve" in [o.name for o in client.observations]


def test_an_untraced_graph_still_runs(tmp_path, monkeypatch):
    monkeypatch.delenv(PUBLIC_KEY_ENV, raising=False)
    graph = build_graph(
        checkpointer=sqlite_checkpointer(tmp_path / "g.db"),
        client=FakeAnthropic(),
        async_client=FakeAsyncAnthropic(),
        library=FakeLibrary(),
    )
    graph.invoke(initial_state("q", route="library"), {"configurable": {"thread_id": "t"}})
