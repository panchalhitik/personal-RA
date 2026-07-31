from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import chromadb
import pytest

from personal_ra.library import COLLECTION
from personal_ra.search import Library, answer_across_library, rrf_fuse

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_PDF = FIXTURES / "two_column.pdf"

# Deterministic 3-d embeddings: each doc gets a distinct axis so "nearest" is
# predictable without a real model.
VECTORS = {
    "contrastive loss": [1.0, 0.0, 0.0],
    "transformer attention": [0.0, 1.0, 0.0],
    "reinforcement learning": [0.0, 0.0, 1.0],
}

CHUNKS = [
    {
        "id": "paperA:0",
        "text": "We train with a contrastive loss over augmented pairs of examples.",
        "vec": VECTORS["contrastive loss"],
        "meta": {
            "paper_id": "paperA",
            "paper_title": "Contrastive Methods",
            "page": 3,
            "section": "2. Method",
            "chunk_index": 0,
            "year": 2021,
            "source_path": str(FIXTURE_PDF),
        },
    },
    {
        "id": "paperB:0",
        "text": "alpha bravo studies attention mechanisms in depth on page 1",
        "vec": VECTORS["transformer attention"],
        "meta": {
            "paper_id": "paperB",
            "paper_title": "Attention Study",
            "page": 1,
            "section": "1. Introduction",
            "chunk_index": 0,
            "year": 2023,
            "source_path": str(FIXTURE_PDF),
        },
    },
    {
        "id": "paperC:0",
        "text": "Our agent is trained with reinforcement learning from human feedback.",
        "vec": VECTORS["reinforcement learning"],
        "meta": {
            "paper_id": "paperC",
            "paper_title": "RLHF Agents",
            "page": 5,
            "section": "3. Experiments",
            "chunk_index": 0,
            "year": 2025,
            "source_path": str(FIXTURE_PDF),
        },
    },
]


def fake_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for t in texts:
        low = t.lower()
        vec = next((v for kw, v in VECTORS.items() if kw in low), [0.33, 0.33, 0.33])
        out.append(vec)
    return out


@pytest.fixture()
def library(tmp_path: Path) -> Library:
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.upsert(
        ids=[c["id"] for c in CHUNKS],
        embeddings=[c["vec"] for c in CHUNKS],
        documents=[c["text"] for c in CHUNKS],
        metadatas=[c["meta"] for c in CHUNKS],
    )
    return Library(db_path=tmp_path / "db", embed_fn=fake_embed)


def make_client(text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )
    return client


def test_rrf_ranks_consensus_above_single_list_winner() -> None:
    # "b" tops neither list but appears in both; "a" and "d" each top one list only.
    scores = rrf_fuse([["a", "b", "c"], ["d", "b", "e"]], k=60)
    assert scores["b"] > scores["a"] == pytest.approx(scores["d"])
    # hand-computed: b = 1/62 + 1/62 ≈ 0.03226, a = 1/61 ≈ 0.01639
    assert scores["b"] == pytest.approx(2 / 62)
    assert scores["a"] == pytest.approx(1 / 61)


def test_rrf_is_convex_within_a_single_pair_of_lists() -> None:
    # Documented consequence of RRF's 1/(k+rank) shape: across two lists,
    # rank1+rank3 edges out rank2+rank2. Consensus helps, but not unconditionally.
    scores = rrf_fuse([["a", "b", "c"], ["c", "b", "a"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["b"] == pytest.approx(2 / 62)
    assert scores["a"] > scores["b"]


def test_rrf_single_ranking_preserves_order() -> None:
    scores = rrf_fuse([["x", "y", "z"]])
    assert scores["x"] > scores["y"] > scores["z"]


def test_dense_retrieval_finds_semantic_match(library: Library) -> None:
    hits = library.search("contrastive loss", k=1, mode="dense")
    assert hits[0].metadata["paper_id"] == "paperA"
    assert hits[0].dense_rank == 1 and hits[0].bm25_rank is None


def test_bm25_retrieval_finds_keyword_match(library: Library) -> None:
    hits = library.search("reinforcement learning human feedback", k=1, mode="bm25")
    assert hits[0].metadata["paper_id"] == "paperC"
    assert hits[0].bm25_rank == 1 and hits[0].dense_rank is None


def test_hybrid_returns_scores_and_both_ranks(library: Library) -> None:
    hits = library.search("contrastive loss", k=3, mode="hybrid")
    assert hits[0].metadata["paper_id"] == "paperA"
    assert all(h.score > 0 for h in hits)
    assert hits[0].dense_rank is not None and hits[0].bm25_rank is not None


def test_paper_id_filter(library: Library) -> None:
    hits = library.search("attention", k=5, paper_id="paperB")
    assert hits and all(h.metadata["paper_id"] == "paperB" for h in hits)


def test_year_range_filters(library: Library) -> None:
    hits = library.search("learning", k=5, year_min=2022)
    assert hits and all(h.metadata["year"] >= 2022 for h in hits)
    hits = library.search("learning", k=5, year_max=2022)
    assert hits and all(h.metadata["year"] <= 2022 for h in hits)
    hits = library.search("learning", k=5, year_min=2023, year_max=2023)
    assert all(h.metadata["year"] == 2023 for h in hits)


def test_retrieval_spans_multiple_papers(library: Library) -> None:
    hits = library.search("learning methods for models", k=3)
    assert len({h.metadata["paper_id"] for h in hits}) > 1


def test_answer_verifies_quote_and_attributes_paper(library: Library) -> None:
    # This text really is in the fixture PDF (page 1), which every chunk points at.
    client = make_client(
        "Several papers touch on this <quote>alpha bravo studies attention mechanisms "
        "in depth</quote> in their setup."
    )
    answer = answer_across_library("What is studied?", library=library, client=client)
    assert len(answer.citations) == 1
    assert answer.citations[0].citation.verified
    assert answer.citations[0].citation.page == 1
    assert "p. 1]" in answer.text
    assert answer.unverified == []


def test_answer_flags_hallucinated_quote(library: Library) -> None:
    client = make_client("It claims <quote>99.9% accuracy on ImageNet was reached</quote>.")
    answer = answer_across_library("What accuracy?", library=library, client=client)
    assert len(answer.unverified) == 1
    assert answer.citations == []
    assert "[unverified]" in answer.text


def test_prompt_groups_chunks_by_paper(library: Library) -> None:
    client = make_client("No quotes.")
    answer_across_library("learning methods for models", k=3, library=library, client=client)
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    chunks = library.search("learning methods for models", k=3)
    assert content.count("<paper title=") == len({c.metadata["paper_title"] for c in chunks})
    for chunk in chunks:
        assert f'<paper title="{chunk.metadata["paper_title"]}"' in content
    assert "[page " in content


def test_empty_library_refuses(tmp_path: Path) -> None:
    empty = Library(db_path=tmp_path / "empty", embed_fn=fake_embed)
    client = make_client("should not be called")
    answer = answer_across_library("anything", library=empty, client=client)
    client.messages.create.assert_not_called()
    assert answer.text.startswith("That isn't covered")
    assert answer.chunks == []
