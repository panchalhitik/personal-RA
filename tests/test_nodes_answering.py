"""The three answering nodes: retrieve, generate, single_paper.

Built after step 3.8 showed that route accuracy was the only metric a stubbed graph
could produce. All doubles; no network, no Chroma.
"""

from __future__ import annotations

from conftest import FakeAnthropic, FakeLibrary, make_chunk, make_paper
from personal_ra.graph.generate import STRICTER, library_system_prompt
from personal_ra.graph.nodes import generate_node, retrieve_node, single_paper_node
from personal_ra.graph.rerank import RETRIEVE_DEPTH, TOP_K
from personal_ra.graph.state import (
    MAX_REWRITES,
    chunk_from_dict,
    chunk_to_dict,
    initial_state,
)

# --- retrieve ---------------------------------------------------------------------


def test_retrieve_returns_state_shaped_dicts():
    library = FakeLibrary()
    delta = retrieve_node(initial_state("sandbagging?", route="library"), library=library)
    assert len(delta["chunks"]) == 2
    assert delta["chunks"][0]["metadata"]["paper_title"] == "Paper P1"
    assert isinstance(delta["chunks"][0], dict)  # State carries dicts, not dataclasses


def test_retrieve_uses_hybrid_and_the_current_query():
    """The rewritten query is what retrieval should use — that is the whole point
    of rewriting it."""
    library = FakeLibrary()
    state = initial_state("original", route="library")
    state["question"] = "sharper rewritten query"
    retrieve_node(state, library=library)
    assert library.searches[0]["query"] == "sharper rewritten query"
    assert library.searches[0]["mode"] == "hybrid"


def test_retrieve_depth_follows_the_rerank_flag():
    shallow, deep = FakeLibrary(), FakeLibrary()
    retrieve_node(initial_state("q", route="library"), library=shallow)
    retrieve_node(initial_state("q", route="library", rerank=True), library=deep)
    assert shallow.searches[0]["k"] == TOP_K
    assert deep.searches[0]["k"] == RETRIEVE_DEPTH


def test_retrieve_scopes_to_the_open_paper_only_on_the_single_paper_route():
    """A library question asked with a paper open must still search the library."""
    scoped, unscoped = FakeLibrary(), FakeLibrary()
    retrieve_node(initial_state("q", "abc123", route="single_paper"), library=scoped)
    retrieve_node(initial_state("q", "abc123", route="library"), library=unscoped)
    assert scoped.searches[0]["paper_id"] == "abc123"
    assert unscoped.searches[0]["paper_id"] is None


def test_chunk_round_trip_tolerates_the_grade_annotation():
    """Graded chunks carry grade_reason, which is not a RetrievedChunk field."""
    graded = chunk_to_dict(make_chunk("text")) | {"grade_reason": "kept: on topic"}
    restored = chunk_from_dict(graded)
    assert restored.text == "text"
    assert restored.dense_rank == 1


# --- generate ---------------------------------------------------------------------


def _generating_state(**overrides):
    state = initial_state("how many poisoned documents?", route="library")
    state["question"] = "poisoning near-constant sample count"  # a rewrite happened
    state["graded_chunks"] = [
        chunk_to_dict(make_chunk("roughly 250 documents suffice", "c1")) | {"grade_reason": "ok"},
        chunk_to_dict(make_chunk("the effect is size-independent", "c2")) | {"grade_reason": "ok"},
    ]
    state.update(overrides)
    return state


def test_generate_answers_the_original_question_not_the_rewrite():
    """The rewrite steered retrieval; the user is owed an answer to what they asked."""
    client = FakeAnthropic(answer="An answer.")
    generate_node(_generating_state(), client=client)
    sent = client.generation_calls()[0]["messages"][0]["content"]
    assert "how many poisoned documents?" in sent
    assert "near-constant sample count" not in sent


def test_generate_passes_the_graded_chunks_as_context():
    client = FakeAnthropic()
    generate_node(_generating_state(), client=client)
    sent = client.generation_calls()[0]["messages"][0]["content"]
    assert "roughly 250 documents suffice" in sent
    assert "<paper" in sent


def test_generate_records_answer_and_usage():
    delta = generate_node(_generating_state(), client=FakeAnthropic(answer="Plain answer."))
    assert delta["answer"] == "Plain answer."
    assert delta["usage"]["generate"]["input_tokens"] == 900


def test_generate_uses_the_stricter_prompt_only_on_a_regeneration():
    first = FakeAnthropic()
    generate_node(_generating_state(), client=first)
    assert STRICTER not in first.generation_calls()[0]["system"]

    retry = FakeAnthropic()
    generate_node(
        _generating_state(grounding={"verdict": "ungrounded", "attempt": 1}), client=retry
    )
    assert STRICTER in retry.generation_calls()[0]["system"]


def test_generate_says_so_when_retrieval_stayed_thin_after_two_rewrites():
    """§3.5: proceed with whatever survived, and say so in the answer."""
    client = FakeAnthropic()
    state = _generating_state(rewrite_count=MAX_REWRITES)
    state["graded_chunks"] = state["graded_chunks"][:1]  # below MIN_GRADED_CHUNKS
    generate_node(state, client=client)
    assert "rewritten twice" in client.generation_calls()[0]["system"]


def test_a_healthy_run_gets_no_thin_retrieval_note():
    client = FakeAnthropic()
    generate_node(_generating_state(), client=client)
    assert "rewritten twice" not in client.generation_calls()[0]["system"]


def test_web_results_reach_the_generator_labelled_as_external():
    """They must be visibly not-library in the prompt, since only <paper> excerpts
    are quote-verified and can earn a page citation."""
    client = FakeAnthropic()
    state = _generating_state(
        web_results=[{"title": "A blog", "url": "https://example.com", "content": "web text"}]
    )
    generate_node(state, client=client)
    sent = client.generation_calls()[0]["messages"][0]["content"]
    assert "<web_result" in sent and "https://example.com" in sent
    assert "web text" in sent


def test_library_system_prompt_composes_both_modifiers():
    both = library_system_prompt(stricter=True, thin=True)
    assert STRICTER in both and "rewritten twice" in both
    assert library_system_prompt() != both


# --- single_paper -----------------------------------------------------------------


def test_single_paper_degrades_when_the_paper_id_cannot_be_resolved():
    """A stale UI selection should not crash the graph."""
    delta = single_paper_node(
        initial_state("q", "no-such-paper", route="single_paper"),
        client=FakeAnthropic(),
        library=FakeLibrary([]),
    )
    assert delta["answer"] == ""
    assert delta["citations"] == []


def test_single_paper_with_no_paper_open_returns_nothing_rather_than_guessing():
    delta = single_paper_node(initial_state("q"), client=FakeAnthropic(), library=FakeLibrary())
    assert delta["answer"] == ""


def test_single_paper_sends_the_whole_paper_and_answers_the_original_question(monkeypatch):
    import personal_ra.graph.nodes as nodes_mod
    import personal_ra.graph.retrieve as retrieve_mod

    paper = make_paper(["page one about sandbagging", "page two about evaluations"])
    monkeypatch.setattr(retrieve_mod, "parsed_paper", lambda path: paper)
    monkeypatch.setattr(nodes_mod, "enrich_paper", lambda p: p)  # no vision calls

    client = FakeAnthropic(answer="Sandbagging is deliberate underperformance.")
    delta = single_paper_node(
        initial_state("what is sandbagging?", "p1", route="single_paper"),
        client=client,
        library=FakeLibrary(),
    )

    assert delta["answer"] == "Sandbagging is deliberate underperformance."
    call = client.generation_calls()[0]
    assert call["messages"][-1]["content"] == "what is sandbagging?"
    # v0's core insight: the whole paper goes in the system prompt, cached.
    assert "page two about evaluations" in call["system"][1]["text"]
    assert call["system"][1]["cache_control"] == {"type": "ephemeral"}
