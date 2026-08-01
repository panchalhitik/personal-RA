"""Step 3.4 — the grader node. Mocked async client throughout; no network."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from personal_ra.graph.build import build_graph
from personal_ra.graph.grade import GRADER_MODEL, grade_chunks
from personal_ra.graph.nodes import after_grade, grade_node
from personal_ra.graph.state import MIN_GRADED_CHUNKS, initial_state

RELEVANT = "roughly 250 poisoned documents compromise models across all sizes tested"
IRRELEVANT = "prior work on federated learning is surveyed by Smith et al. (2019)"


def _verdict(relevant: bool, reason: str, in_tokens: int = 400, out_tokens: int = 20):
    block = SimpleNamespace(
        type="tool_use",
        name="grade_excerpt",
        input={"relevant": relevant, "reason": reason},
    )
    return SimpleNamespace(
        content=[block],
        usage=SimpleNamespace(input_tokens=in_tokens, output_tokens=out_tokens),
    )


class FakeAsyncClient:
    """Grades by a keyword rule, with an optional per-call delay for timing tests."""

    def __init__(self, delay: float = 0.0, fail: bool = False) -> None:
        self.delay = delay
        self.fail = fail
        self.calls = 0
        self.max_concurrent = 0
        self._in_flight = 0
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls += 1
        self._in_flight += 1
        self.max_concurrent = max(self.max_concurrent, self._in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.fail:
                raise RuntimeError("upstream exploded")
            # Judge the excerpt only. The prompt also carries the question, and
            # keying on the whole payload would make every chunk look relevant
            # whenever the question shares a word with the relevant excerpt.
            prompt = kwargs["messages"][0]["content"]
            excerpt = prompt.split("</excerpt>")[0]
            if "poisoned" in excerpt:
                return _verdict(True, "gives the poisoned-document count directly")
            return _verdict(False, "surveys federated learning, a different topic")
        finally:
            self._in_flight -= 1


def chunk(text: str, cid: str = "c1", paper: str = "A") -> dict:
    return {
        "id": cid,
        "text": text,
        "metadata": {"paper_id": paper, "paper_title": "Paper A", "page": 3},
        "score": 0.5,
    }


# --- the judgment itself ----------------------------------------------------------


def test_relevant_chunk_survives_and_irrelevant_is_rejected():
    kept, rejected, _ = grade_chunks(
        "how many poisoned documents backdoor a model?",
        [chunk(RELEVANT, "c1"), chunk(IRRELEVANT, "c2")],
        client=FakeAsyncClient(),
    )
    assert [c["id"] for c in kept] == ["c1"]
    assert [c["id"] for c in rejected] == ["c2"]


def test_rejection_reasons_are_populated():
    """3.5's rewrite reads these to learn the corpus vocabulary, so an empty reason
    silently degrades the rewrite rather than failing loudly."""
    _, rejected, _ = grade_chunks(
        "poisoned documents?", [chunk(IRRELEVANT)], client=FakeAsyncClient()
    )
    assert rejected[0]["grade_reason"]
    assert "federated" in rejected[0]["grade_reason"]


def test_kept_chunks_keep_their_reason_too():
    kept, _, _ = grade_chunks("poisoned?", [chunk(RELEVANT)], client=FakeAsyncClient())
    assert kept[0]["grade_reason"]
    assert kept[0]["text"] == RELEVANT  # the chunk itself survives intact


def test_usage_is_reported():
    _, _, usage = grade_chunks(
        "poisoned?", [chunk(RELEVANT, "c1"), chunk(IRRELEVANT, "c2")], client=FakeAsyncClient()
    )
    assert usage["n_graded"] == 2
    assert usage["input_tokens"] == 800 and usage["cost_usd"] > 0


def test_empty_chunk_list_makes_no_calls():
    client = FakeAsyncClient()
    kept, rejected, usage = grade_chunks("q", [], client=client)
    assert (kept, rejected) == ([], [])
    assert client.calls == 0 and usage["n_graded"] == 0


def test_grader_uses_haiku_with_forced_tool_use():
    captured = {}

    class Capturing(FakeAsyncClient):
        async def _create(self, **kwargs):
            captured.update(kwargs)
            return await super()._create(**kwargs)

    grade_chunks("q", [chunk(RELEVANT)], client=Capturing())
    assert captured["model"] == GRADER_MODEL == "claude-haiku-4-5"
    assert captured["tool_choice"] == {"type": "tool", "name": "grade_excerpt"}
    assert captured["tools"][0]["strict"] is True
    assert captured["tools"][0]["input_schema"]["additionalProperties"] is False


# --- concurrency ------------------------------------------------------------------


def test_grading_is_concurrent_not_serial():
    """Eight chunks at 100ms each: concurrent finishes near 100ms, serial near 800ms."""
    client = FakeAsyncClient(delay=0.1)
    chunks = [chunk(RELEVANT, f"c{i}") for i in range(8)]

    start = time.perf_counter()
    kept, _, _ = grade_chunks("poisoned?", chunks, client=client)
    elapsed = time.perf_counter() - start

    assert len(kept) == 8
    assert client.max_concurrent == 8  # all eight were in flight at once
    assert elapsed < 0.4, f"took {elapsed:.2f}s — grading looks serial"


def test_sync_wrapper_works_from_inside_a_running_loop():
    """Streamlit has no loop, but FastAPI will — the wrapper must not raise there."""

    async def outer():
        return grade_chunks("poisoned?", [chunk(RELEVANT)], client=FakeAsyncClient())

    kept, _, _ = asyncio.run(outer())
    assert len(kept) == 1


# --- failure handling -------------------------------------------------------------


def test_grading_failure_keeps_the_chunk_rather_than_dropping_it():
    """Failing closed would let a transient API error trigger a spurious rewrite."""
    kept, rejected, _ = grade_chunks(
        "poisoned?", [chunk(RELEVANT)], client=FakeAsyncClient(fail=True)
    )
    assert len(kept) == 1 and rejected == []
    assert "ungraded" in kept[0]["grade_reason"]


def test_one_failure_does_not_sink_the_whole_batch():
    calls = {"n": 0}

    class FlakyOnce(FakeAsyncClient):
        async def _create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return await FakeAsyncClient._create(self, **kwargs)

    kept, rejected, _ = grade_chunks(
        "poisoned?",
        [chunk(RELEVANT, "c1"), chunk(IRRELEVANT, "c2"), chunk(RELEVANT, "c3")],
        client=FlakyOnce(),
    )
    assert len(kept) + len(rejected) == 3


# --- the node, and the rewrite threshold ------------------------------------------


def test_grade_node_writes_both_lists_and_usage():
    state = initial_state("how many poisoned documents?")
    state["chunks"] = [chunk(RELEVANT, "c1"), chunk(IRRELEVANT, "c2")]
    delta = grade_node(state, client=FakeAsyncClient())
    assert [c["id"] for c in delta["graded_chunks"]] == ["c1"]
    assert [c["id"] for c in delta["rejected_chunks"]] == ["c2"]
    assert delta["usage"]["grade"]["n_graded"] == 2


def test_grade_node_judges_the_original_question_not_the_rewrite():
    """The rewrite is a retrieval device; relevance is judged against what was asked."""
    seen = []

    class Capturing(FakeAsyncClient):
        async def _create(self, **kwargs):
            seen.append(kwargs["messages"][0]["content"])
            return await super()._create(**kwargs)

    state = initial_state("how many poisoned documents?")
    state["question"] = "near-constant poison sample count Chinchilla-optimal pretraining"
    state["chunks"] = [chunk(RELEVANT)]
    grade_node(state, client=Capturing())

    assert "how many poisoned documents?" in seen[0]
    assert "Chinchilla-optimal" not in seen[0]


def test_too_few_survivors_triggers_the_rewrite_branch():
    state = initial_state("q")
    state["chunks"] = [chunk(IRRELEVANT, "c1"), chunk(IRRELEVANT, "c2")]
    delta = grade_node(state, client=FakeAsyncClient())
    assert len(delta["graded_chunks"]) < MIN_GRADED_CHUNKS
    assert after_grade({**state, **delta}) == "rewrite"


def test_enough_survivors_proceeds_to_generate():
    state = initial_state("poisoned?")
    state["chunks"] = [chunk(RELEVANT, "c1"), chunk(RELEVANT, "c2")]
    delta = grade_node(state, client=FakeAsyncClient())
    assert len(delta["graded_chunks"]) >= MIN_GRADED_CHUNKS
    assert after_grade({**state, **delta}) == "generate"


@pytest.mark.parametrize("enabled", [False, True])
def test_graph_runs_grading_in_the_library_branch(enabled):
    graph = build_graph(async_client=FakeAsyncClient())
    state = initial_state("poisoned documents?", route="library", rerank=enabled)
    visited = [
        node
        for update in graph.stream(
            state, {"configurable": {"thread_id": f"t{enabled}"}}, stream_mode="updates"
        )
        for node in update
    ]
    assert "grade" in visited and visited[-1] == "grounding"
