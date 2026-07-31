from pathlib import Path

from conftest import make_paper
from personal_ra.library import chunk_paper, detect_year, ingest, paper_id, year_from_arxiv_id
from personal_ra.parse import Page, Paper

FIXTURES = Path(__file__).parent / "fixtures"

INTRO_FILLER = " ".join(
    f"Sentence number {i} discusses retrieval quality and various design choices in detail."
    for i in range(30)
)
METHOD_FILLER = " ".join(
    f"Step {i} of the method applies an embedding transformation to every chunk of text."
    for i in range(30)
)
PAGE_1 = f"1. Introduction\nWe study retrieval systems. {INTRO_FILLER}"
PAGE_2 = f"2. Method\nOur method uses embeddings. {METHOD_FILLER}\nConclusion\nIt works well."

SECTIONED = make_paper([PAGE_1, PAGE_2])
PID = "abc123def456"


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(t) % 11), float(t.count("e") % 7), 1.0] for t in texts]


def test_chunk_ids_deterministic_and_sequential() -> None:
    first = chunk_paper(SECTIONED, PID, 2024)
    second = chunk_paper(SECTIONED, PID, 2024)
    assert [c.id for c in first] == [c.id for c in second]
    assert [c.id for c in first] == [f"{PID}:{i}" for i in range(len(first))]


def test_sections_detected_and_not_spanned() -> None:
    chunks = chunk_paper(SECTIONED, PID, 2024)
    sections = {c.metadata["section"] for c in chunks}
    assert "1. Introduction" in sections
    assert "2. Method" in sections
    assert "Conclusion" in sections
    # section headers are labels, never chunk content
    assert all("1. Introduction" not in c.text for c in chunks)


def test_metadata_complete_on_every_chunk() -> None:
    for c in chunk_paper(SECTIONED, PID, 2024):
        assert set(c.metadata) == {
            "paper_id",
            "paper_title",
            "page",
            "section",
            "chunk_index",
            "year",
            "source_path",
        }
        assert c.metadata["paper_id"] == PID
        assert c.metadata["paper_title"] == "Synthetic"
        assert c.metadata["page"] in (1, 2)
        assert c.metadata["year"] == 2024


def test_chunk_size_and_sentence_boundaries() -> None:
    chunks = chunk_paper(SECTIONED, PID, 2024)
    assert len(chunks) > 3  # the filler forces several chunks per section
    for c in chunks:
        assert len(c.text) < 1600
        assert c.text.rstrip()[-1] in ".!?"


def test_consecutive_chunks_overlap_within_section() -> None:
    chunks = chunk_paper(SECTIONED, PID, 2024)
    pairs = [
        (a, b) for a, b in zip(chunks, chunks[1:]) if a.metadata["section"] == b.metadata["section"]
    ]
    assert pairs
    for a, b in pairs:
        first_sentence = b.text.split(".")[0]
        assert first_sentence in a.text  # b starts with a's tail


def test_context_prefix_in_embed_text() -> None:
    chunk = chunk_paper(SECTIONED, PID, 2024)[0]
    assert chunk.embed_text.startswith("From 'Synthetic', section '1. Introduction': ")
    assert chunk.text in chunk.embed_text


def test_section_strategy_drops_prefix_keeps_sections() -> None:
    chunks = chunk_paper(SECTIONED, PID, 2024, strategy="section")
    default = chunk_paper(SECTIONED, PID, 2024)
    assert [c.text for c in chunks] == [c.text for c in default]  # same splits
    assert all(c.embed_text == c.text for c in chunks)  # no prefix
    assert chunks[0].metadata["section"] == "1. Introduction"


def test_fixed_strategy_blind_windows() -> None:
    chunks = chunk_paper(SECTIONED, PID, 2024, strategy="fixed")
    assert chunks
    assert all(len(c.text) <= 1000 for c in chunks)
    assert all(c.metadata["section"] == "" for c in chunks)  # no section awareness
    assert all(c.embed_text == c.text for c in chunks)
    # blind splitting cuts mid-sentence somewhere — that's the point of the baseline
    assert any(c.text.rstrip()[-1] not in ".!?" for c in chunks)
    # section headers are NOT stripped in fixed mode (they're just text)
    assert any("1. Introduction" in c.text for c in chunks)


def test_unknown_strategy_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown strategy"):
        chunk_paper(SECTIONED, PID, 2024, strategy="typo")


def paper_named(name: str, page1: str) -> Paper:
    """A Paper with a controlled filename, for year-detection tests."""
    return Paper(
        path=Path(name),
        title="T",
        pages=[Page(number=1, text=page1)],
        full_text=page1,
        n_tokens=1,
    )


def test_year_from_arxiv_id() -> None:
    assert year_from_arxiv_id("2506.05346v1.pdf") == 2025
    assert year_from_arxiv_id("2601.04603v1.pdf") == 2026
    assert year_from_arxiv_id("6864_Scaling_Laws.pdf") is None  # not an arXiv id
    assert year_from_arxiv_id("2513.00001.pdf") is None  # month 13 is invalid


def test_detect_year_prefers_arxiv_id_over_citations() -> None:
    # Regression: the old "largest year on page 1" rule read citation years and
    # dated a 2025 paper to 2014.
    paper = paper_named("2510.06105v1.pdf", "Prior work (Smith et al., 2014) studied this.")
    assert detect_year(paper) == 2025


def test_detect_year_from_publication_context() -> None:
    assert detect_year(paper_named("x.pdf", "Published in 2024. See also work from 2019.")) == 2024
    assert detect_year(paper_named("x.pdf", "Preprint, 2025. Citing Jones 2011.")) == 2025
    assert detect_year(paper_named("x.pdf", "Accepted at ICLR 2026")) == 2026


def test_detect_year_ignores_bare_citation_years() -> None:
    # No publication context anywhere: better to return nothing than a wrong year.
    paper = paper_named("notes.pdf", "We build on Vaswani et al. 2017 and Brown et al. 2020.")
    assert detect_year(paper) is None


def test_paper_id_stable(tmp_path: Path) -> None:
    f = tmp_path / "a.pdf"
    f.write_bytes(b"same content")
    assert paper_id(f) == paper_id(f)
    g = tmp_path / "renamed.pdf"
    g.write_bytes(b"same content")
    assert paper_id(f) == paper_id(g)  # identity follows content, not filename


def test_ingest_idempotent(tmp_path: Path) -> None:
    first = ingest(FIXTURES, tmp_path / "db", embed_fn=fake_embed)
    assert first["papers"] == 1 and first["chunks"] > 0
    second = ingest(FIXTURES, tmp_path / "db", embed_fn=fake_embed)
    assert second["total_in_db"] == first["total_in_db"]  # re-run adds nothing


def test_ingest_rebuild_matches_fresh(tmp_path: Path) -> None:
    ingest(FIXTURES, tmp_path / "db", embed_fn=fake_embed)
    rebuilt = ingest(FIXTURES, tmp_path / "db", rebuild=True, embed_fn=fake_embed)
    fresh = ingest(FIXTURES, tmp_path / "db2", embed_fn=fake_embed)
    assert rebuilt["total_in_db"] == fresh["total_in_db"]
