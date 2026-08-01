"""Step 3.3 — the cross-encoder reranker. The real model is a ~90MB download and
a torch import, so every test here injects a fake scorer instead."""

from __future__ import annotations

import pytest

from personal_ra.eval import RETRIEVAL_MODES, Question, build_search_fn, evaluate_config
from personal_ra.graph import rerank as rr
from personal_ra.search import RetrievedChunk


class FakeCrossEncoder:
    """Scores by keyword overlap — enough to drive a deterministic reordering."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        scores = []
        for query, text in pairs:
            terms = set(query.lower().split())
            scores.append(sum(1.0 for t in terms if t in text.lower()))
        return scores


def chunk(cid: str, text: str, paper: str = "p1", score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid,
        text=text,
        metadata={"paper_id": paper, "paper_title": paper.upper(), "page": 1},
        score=score,
        dense_rank=1,
        bm25_rank=2,
    )


# --- reordering -------------------------------------------------------------------


def test_reranker_reorders_a_synthetic_case():
    """Retrieval ranked the off-topic chunk first (high RRF score); the cross-encoder
    must pull the on-topic one to the top."""
    chunks = [
        chunk("c1", "unrelated discussion of dataset licensing", score=0.9),
        chunk("c2", "sandbagging is deliberate underperformance on evaluations", score=0.1),
    ]
    out = rr.rerank("what is sandbagging", chunks, model=FakeCrossEncoder())
    assert [c.id for c in out] == ["c2", "c1"]


def test_rerank_replaces_score_but_keeps_retrieval_provenance():
    out = rr.rerank(
        "sandbagging", [chunk("c1", "sandbagging", score=0.42)], model=FakeCrossEncoder()
    )
    assert out[0].score != 0.42  # now the cross-encoder score
    assert out[0].dense_rank == 1 and out[0].bm25_rank == 2


def test_rerank_truncates_to_top_k():
    chunks = [chunk(f"c{i}", f"sandbagging {i}") for i in range(20)]
    assert len(rr.rerank("sandbagging", chunks, top_k=8, model=FakeCrossEncoder())) == 8


def test_rerank_is_stable_on_ties():
    """Equal cross-encoder scores must keep retrieval order, not shuffle."""
    chunks = [chunk(f"c{i}", "identical text") for i in range(5)]
    out = rr.rerank("nomatch", chunks, model=FakeCrossEncoder())
    assert [c.id for c in out] == [f"c{i}" for i in range(5)]


def test_rerank_handles_empty_input():
    assert rr.rerank("q", [], model=FakeCrossEncoder()) == []


# --- max_per_paper, post-rerank ---------------------------------------------------


def test_max_per_paper_caps_after_reranking():
    """Paper A wins on relevance for all three of its chunks, but the cap must still
    let paper B through — that is diversity, applied after relevance, not instead."""
    chunks = [
        chunk("a1", "sandbagging sandbagging sandbagging", paper="A"),
        chunk("a2", "sandbagging sandbagging", paper="A"),
        chunk("a3", "sandbagging sandbagging", paper="A"),
        chunk("b1", "sandbagging", paper="B"),
    ]
    out = rr.rerank("sandbagging", chunks, top_k=3, max_per_paper=2, model=FakeCrossEncoder())
    assert [c.metadata["paper_id"] for c in out] == ["A", "A", "B"]


def test_max_per_paper_unset_leaves_ranking_alone():
    chunks = [chunk(f"a{i}", "sandbagging", paper="A") for i in range(4)]
    out = rr.rerank("sandbagging", chunks, top_k=3, model=FakeCrossEncoder())
    assert len(out) == 3 and {c.metadata["paper_id"] for c in out} == {"A"}


def test_cap_per_paper_preserves_relevance_order():
    ordered = [
        chunk("a1", "x", paper="A"),
        chunk("b1", "x", paper="B"),
        chunk("a2", "x", paper="A"),
        chunk("c1", "x", paper="C"),
    ]
    kept = rr.cap_per_paper(ordered, max_per_paper=1, top_k=10)
    assert [c.id for c in kept] == ["a1", "b1", "c1"]


# --- model loading ----------------------------------------------------------------


def test_model_loads_once_across_repeated_calls(monkeypatch):
    loads = []

    class FakeModule:
        @staticmethod
        def CrossEncoder(name):  # noqa: N802 - mirrors the real class name
            loads.append(name)
            return FakeCrossEncoder()

    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", FakeModule)
    rr.load_model.cache_clear()
    try:
        first, second = rr.load_model(), rr.load_model()
        assert first is second
        assert loads == [rr.RERANK_MODEL]  # constructed exactly once
    finally:
        rr.load_model.cache_clear()


# --- retrieve-then-rerank ---------------------------------------------------------


class FakeLibrary:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_k = None
        self.last_mode = None

    def search(self, query, k=8, mode="hybrid", **kwargs):
        self.last_k, self.last_mode = k, mode
        return self._chunks[:k]


def test_retrieve_and_rerank_fetches_deeper_than_it_returns():
    library = FakeLibrary([chunk(f"c{i}", f"sandbagging {i}") for i in range(50)])
    out = rr.retrieve_and_rerank(library, "sandbagging", top_k=rr.TOP_K, model=FakeCrossEncoder())
    assert library.last_k == rr.RETRIEVE_DEPTH
    assert rr.RETRIEVE_DEPTH > rr.TOP_K  # reranking needs candidates to choose between
    assert library.last_mode == "hybrid"
    assert len(out) == rr.TOP_K


def test_explicit_depth_is_honoured_not_inflated():
    """The depth sweep at Checkpoint 3.3 measures nothing if a requested depth is
    silently raised — this is a regression test for exactly that bug."""
    library = FakeLibrary([chunk(f"c{i}", "x") for i in range(200)])
    rr.retrieve_and_rerank(library, "q", top_k=10, depth=15, model=FakeCrossEncoder())
    assert library.last_k == 15


def test_depth_never_drops_below_top_k():
    """You cannot return 8 chunks from a pool of 5."""
    library = FakeLibrary([chunk(f"c{i}", "x") for i in range(200)])
    rr.retrieve_and_rerank(library, "q", top_k=20, depth=5, model=FakeCrossEncoder())
    assert library.last_k == 20


# --- the 12-config matrix ---------------------------------------------------------


def test_retrieval_modes_now_number_four():
    assert RETRIEVAL_MODES == ("dense", "bm25", "hybrid", "rerank")


def test_matrix_runs_all_twelve_configs_end_to_end(monkeypatch):
    """3 chunking x 4 retrieval on a 3-paper fixture library, offline."""
    fixture = [
        chunk("a1", "sandbagging is deliberate underperformance", paper="A"),
        chunk("b1", "constitutional classifiers reduce cost", paper="B"),
        chunk("c1", "poisoned documents backdoor a model", paper="C"),
    ]
    questions = [
        Question(id="q1", question="sandbagging", category="factual", expected_paper_ids=["A"]),
        Question(
            id="q2", question="poisoned documents", category="factual", expected_paper_ids=["C"]
        ),
        Question(id="q3", question="nothing here", category="unanswerable"),
    ]
    monkeypatch.setattr(rr, "load_model", lambda *a, **kw: FakeCrossEncoder())

    results = []
    for strategy in ("fixed", "section", "section_context"):
        for mode in RETRIEVAL_MODES:
            library = FakeLibrary(fixture)
            search_fn = build_search_fn(library, mode, rerank_model=FakeCrossEncoder())
            outcome = evaluate_config(questions, search_fn, k=10, k_values=(1, 3, 5))
            results.append({"chunking": strategy, "retrieval": mode, **outcome})

    assert len(results) == 12
    assert {r["retrieval"] for r in results} == set(RETRIEVAL_MODES)
    # The rerank configs must actually rank the right paper first, not just run.
    for r in results:
        if r["retrieval"] == "rerank":
            assert r["metrics"]["recall@1"] == 1.0


@pytest.mark.parametrize("mode", RETRIEVAL_MODES)
def test_build_search_fn_returns_paper_ids_for_every_mode(mode):
    library = FakeLibrary([chunk("a1", "sandbagging", paper="A")])
    search_fn = build_search_fn(library, mode, rerank_model=FakeCrossEncoder())
    assert search_fn("sandbagging", 5) == ["A"]
