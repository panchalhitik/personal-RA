"""Per-entity retrieval for comparison questions.

Built after the 60-run eval showed q48/q49 still answering with zero citations
*after* the grader fix — because retrieval returned only one side of a two-sided
comparison, which no grading rule can assemble an answer from.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from conftest import FakeAnthropic, FakeLibrary, make_chunk
from personal_ra.graph.decompose import (
    MAX_FACETS,
    decompose_question,
    interleave,
    looks_multi_part,
)
from personal_ra.graph.rerank import TOP_K
from personal_ra.graph.retrieve import retrieve_chunks

# --- the gate ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "How do STAR-1 and RealSafe-R1 differ in aligning reasoning models?",
        "How do the two DeepMind chain-of-thought papers compare?",
        "OpenAI versus DeepMind monitorability",
        "Do both papers use the same benchmark?",
    ],
)
def test_gate_catches_comparison_questions(question):
    assert looks_multi_part(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "How many poisoned documents does it take to backdoor a model?",
        "Which of my papers use contrastive loss?",
        "What is sandbagging?",
    ],
)
def test_gate_lets_single_subject_questions_through_untouched(question):
    assert looks_multi_part(question) is False


def test_a_gated_question_never_calls_the_splitter():
    """Single-subject questions are the common case and must pay nothing extra."""
    client = MagicMock()
    queries, usage = decompose_question("What is sandbagging?", client=client)
    assert queries == ["What is sandbagging?"]
    assert usage == {}
    client.messages.create.assert_not_called()


# --- splitting --------------------------------------------------------------------


def test_a_comparison_is_split_into_one_query_per_side():
    client = FakeAnthropic(split=["STAR-1 alignment method", "RealSafe-R1 alignment method"])
    queries, usage = decompose_question("How do STAR-1 and RealSafe-R1 differ?", client=client)
    assert queries == ["STAR-1 alignment method", "RealSafe-R1 alignment method"]
    assert usage["cost_usd"] > 0


def test_splitting_is_capped():
    client = FakeAnthropic(split=[f"query {i}" for i in range(10)])
    queries, _ = decompose_question("how do all of these differ?", client=client)
    assert len(queries) == MAX_FACETS


def test_a_failed_split_falls_back_to_the_original_question():
    """A broken splitter must not make retrieval worse than not having one."""
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("upstream exploded")
    queries, usage = decompose_question("how do X and Y differ?", client=client)
    assert queries == ["how do X and Y differ?"]
    assert usage == {}


def test_an_empty_split_falls_back_too():
    client = FakeAnthropic(split=[])
    queries, _ = decompose_question("how do X and Y differ?", client=client)
    assert queries == ["how do X and Y differ?"]


# --- interleaving -----------------------------------------------------------------


def test_interleave_is_round_robin_not_concatenation():
    """The two sides are rarely equally well matched. Concatenating would let the
    stronger side fill every slot, which is the bug this module exists to prevent."""
    left = [make_chunk("a", "a1"), make_chunk("a", "a2"), make_chunk("a", "a3")]
    right = [make_chunk("b", "b1"), make_chunk("b", "b2")]
    assert [c.id for c in interleave([left, right], k=4)] == ["a1", "b1", "a2", "b2"]


def test_interleave_dedups_across_rankings():
    """The same chunk can match both sides; it must not occupy two slots."""
    shared = make_chunk("shared", "s1")
    left = [shared, make_chunk("a", "a1")]
    right = [shared, make_chunk("b", "b1")]
    assert [c.id for c in interleave([left, right], k=4)] == ["s1", "a1", "b1"]


def test_interleave_respects_k_and_uneven_lengths():
    left = [make_chunk("a", f"a{i}") for i in range(5)]
    right = [make_chunk("b", "b1")]
    merged = interleave([left, right], k=3)
    assert [c.id for c in merged] == ["a0", "b1", "a1"]


def test_interleave_on_empty_input():
    assert interleave([], k=5) == []
    assert interleave([[], []], k=5) == []


# --- retrieval wiring -------------------------------------------------------------


def test_a_comparison_searches_once_per_side():
    library = FakeLibrary([make_chunk("x", f"c{i}") for i in range(8)])
    client = FakeAnthropic(split=["side one", "side two"])
    chunks, queries, _ = retrieve_chunks("how do X and Y differ?", library=library, client=client)

    assert [s["query"] for s in library.searches] == ["side one", "side two"]
    assert queries == ["side one", "side two"]
    assert len(chunks) <= TOP_K


def test_each_side_is_searched_at_full_depth():
    """Splitting the budget k/n would make each side shallower than the single-query
    case and lose deep matches grading might have wanted."""
    library = FakeLibrary([make_chunk("x", f"c{i}") for i in range(30)])
    client = FakeAnthropic(split=["side one", "side two"])
    retrieve_chunks("how do X and Y differ?", library=library, client=client)
    assert [s["k"] for s in library.searches] == [TOP_K, TOP_K]


def test_a_single_subject_question_makes_exactly_one_search():
    library = FakeLibrary()
    client = MagicMock()
    chunks, queries, _ = retrieve_chunks("what is sandbagging?", library=library, client=client)
    assert len(library.searches) == 1
    assert queries == ["what is sandbagging?"]
    client.messages.create.assert_not_called()
