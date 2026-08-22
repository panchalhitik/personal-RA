from pathlib import Path

import chromadb

from conftest import make_paper
from personal_ra.library import (
    COLLECTION,
    _is_section_header,
    chunk_paper,
    detect_year,
    ingest,
    ingest_paper,
    is_indexed,
    paper_id,
    year_from_arxiv_id,
)
from personal_ra.parse import EQUATION_HEADER, FIGURE_HEADER, Page, Paper

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
            "content_type",
        }
        assert c.metadata["paper_id"] == PID
        assert c.metadata["paper_title"] == "Synthetic"
        assert c.metadata["page"] in (1, 2)
        assert c.metadata["year"] == 2024
        assert c.metadata["content_type"] == "text"


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
    assert first["papers"] == len(list(FIXTURES.glob("*.pdf"))) and first["chunks"] > 0
    second = ingest(FIXTURES, tmp_path / "db", embed_fn=fake_embed)
    assert second["total_in_db"] == first["total_in_db"]  # re-run adds nothing


def test_figure_and_equation_chunks_labelled_by_content_type() -> None:
    enriched = make_paper(
        [
            f"{PAGE_1}\n{FIGURE_HEADER}\n[FIGURE 1: A bar chart of accuracy against steps.]",
            f"{PAGE_2}\n{EQUATION_HEADER}\n$$ E = mc^2 $$",
        ]
    )
    by_type: dict[str, list[str]] = {}
    for c in chunk_paper(enriched, PID, 2024):
        by_type.setdefault(c.metadata["content_type"], []).append(c.text)

    assert any("bar chart of accuracy" in t for t in by_type["figure"])
    assert any("E = mc^2" in t for t in by_type["equation"])
    # the headers are labels, never chunk content
    assert all(FIGURE_HEADER not in t for t in by_type["figure"])
    # and vision output never shares a chunk with the paper's own prose
    assert by_type["text"] and all("bar chart" not in t for t in by_type["text"])


def test_content_type_reaches_chroma(tmp_path: Path, monkeypatch) -> None:
    from personal_ra import vision

    def fake_enrich(paper, **kwargs):
        pages = list(paper.pages)
        pages[0] = Page(
            number=pages[0].number,
            text=f"{pages[0].text}\n{FIGURE_HEADER}\n[FIGURE 1: A described bar chart.]",
        )
        return Paper(
            path=paper.path,
            title=paper.title,
            pages=pages,
            full_text=paper.full_text,
            n_tokens=paper.n_tokens,
        )

    monkeypatch.setattr(vision, "enrich_paper", fake_enrich)
    db = tmp_path / "db"
    ingest(FIXTURES, db, embed_fn=fake_embed, figures=True)

    collection = chromadb.PersistentClient(path=str(db)).get_collection(COLLECTION)
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    types = {m["content_type"] for m in metadatas}
    assert types == {"text", "figure"}


def test_ingest_paper_indexes_one_without_touching_the_rest(tmp_path: Path) -> None:
    """A daily cron adding one paper shouldn't re-embed the whole library."""
    db = tmp_path / "db"
    ingest(FIXTURES, db, embed_fn=fake_embed)
    collection = chromadb.PersistentClient(path=str(db)).get_collection(COLLECTION)
    before = collection.count()
    ids_before = set(collection.get(include=[])["ids"])

    new_paper = tmp_path / "extra.pdf"
    new_paper.write_bytes((FIXTURES / "two_column.pdf").read_bytes() + b"%extra")
    result = ingest_paper(new_paper, db, embed_fn=fake_embed)

    after = set(collection.get(include=[])["ids"])
    assert result["chunks"] > 0
    assert result["total_in_db"] == before + result["chunks"]
    assert ids_before < after  # everything that was there is still there
    assert all(i.startswith(result["paper_id"]) for i in after - ids_before)


def test_ingest_paper_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "db"
    first = ingest_paper(FIXTURES / "two_column.pdf", db, embed_fn=fake_embed)
    second = ingest_paper(FIXTURES / "two_column.pdf", db, embed_fn=fake_embed)
    assert second["total_in_db"] == first["total_in_db"]


def test_is_indexed_follows_content_not_filename(tmp_path: Path) -> None:
    db = tmp_path / "db"
    paper = FIXTURES / "two_column.pdf"
    assert is_indexed(paper, db) is False
    ingest_paper(paper, db, embed_fn=fake_embed)
    assert is_indexed(paper, db) is True

    renamed = tmp_path / "downloaded-again.pdf"
    renamed.write_bytes(paper.read_bytes())
    assert is_indexed(renamed, db) is True  # same content, so still a duplicate


def test_ingest_rebuild_matches_fresh(tmp_path: Path) -> None:
    ingest(FIXTURES, tmp_path / "db", embed_fn=fake_embed)
    rebuilt = ingest(FIXTURES, tmp_path / "db", rebuild=True, embed_fn=fake_embed)
    fresh = ingest(FIXTURES, tmp_path / "db2", embed_fn=fake_embed)
    assert rebuilt["total_in_db"] == fresh["total_in_db"]


def test_plot_axis_ticks_are_not_section_headers() -> None:
    """A y-axis tick plus a legend entry matched the numbered-section pattern, so
    every chunk after a figure inherited a plot label — 41% of one paper's chunks.
    The label is embedded, so this cost retrieval, not just metadata."""
    for plot_text in ("1.0 Poison every 5 steps", "0.4 Claude Sonnet 4", "1.0 Monitor: Sonnet 3.7"):
        assert not _is_section_header(plot_text)
    for heading in ("3. Method", "3.1 Results", "4.2 EXPERIMENTAL RESULTS", "IV. Experiments"):
        assert _is_section_header(heading)


def test_a_figure_inside_a_paper_does_not_relabel_what_follows() -> None:
    page_one = "2. Method\nWe describe the approach in detail here."
    page_two = "1.0 Poison every 5 steps\nThe results continue in this later section."
    chunks = chunk_paper(make_paper([page_one, page_two]), PID, 2024)
    assert {c.metadata["section"] for c in chunks} == {"2. Method"}


def test_figure_chunks_embed_on_the_papers_own_caption() -> None:
    """Questions arrive in the paper's vocabulary; the caption is that vocabulary
    and the generated description is not."""
    caption = "Figure 9: The values of the probability p+ for each attacker model."
    page = f"{caption}\nSome surrounding prose.\n{FIGURE_HEADER}\n[FIGURE 9: A bar chart of p+.]"
    chunks = chunk_paper(make_paper([page]), PID, 2024)
    figure = next(c for c in chunks if c.metadata["content_type"] == "figure")

    assert caption in figure.embed_text  # the caption is what dense retrieval matches
    assert "A bar chart of p+." in figure.text  # the description is what gets returned
    assert "A bar chart" not in figure.embed_text  # ... and does not dilute the embedding
    # BM25 scores the text, not the embedding, so the caption has to be in both
    assert caption in figure.text


def test_a_figure_with_no_findable_caption_keeps_the_default_prefix() -> None:
    page = f"Some prose.\n{FIGURE_HEADER}\n[FIGURE 4: A scatter plot with no caption nearby.]"
    chunks = chunk_paper(make_paper([page]), PID, 2024)
    figure = next(c for c in chunks if c.metadata["content_type"] == "figure")
    assert figure.embed_text.startswith("From 'Synthetic', section ")
