from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from conftest import make_paper
from personal_ra.cite import verify_quote
from personal_ra.parse import FIGURE_HEADER, Paper, parse_pdf
from personal_ra.vision import (
    EQUATION_HEADER,
    describe_figures,
    detect_equation_pages,
    detect_figures,
    enrich_paper,
)

FIXTURES = Path(__file__).parent / "fixtures"

MATHY = (
    "We calculate\nX\nEp[f] =\nf(a)p(a)\na∈X\n=\nEr[fw]\nEr[w]\n"
    "Therefore the covariance Covr[f, w] ≤\nVr[f]Vr[w] holds\nby Cauchy-Schwarz\n"
    "α\n∆(X)\nσ\n≥\n0"
)
PROSE = (
    "We evaluate the model on three benchmark datasets and report accuracy. "
    "The results show consistent improvements over the baseline across all settings. "
    "Our ablation study confirms that each component contributes to the final score."
)


def make_client(text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)]
    )
    return client


@pytest.fixture()
def paper() -> Paper:
    return parse_pdf(FIXTURES / "two_column.pdf")


def test_detects_mathy_page_not_prose() -> None:
    paper = make_paper([PROSE, MATHY, PROSE])
    assert detect_equation_pages(paper) == [2]


def test_enrich_appends_transcription_to_page(paper: Paper, tmp_path: Path) -> None:
    client = make_client('(after: "we calculate")\n$$ E_p[f] = \\frac{E_r[fw]}{E_r[w]} $$')
    enriched = enrich_paper(paper, pages=[1], client=client, cache_dir=tmp_path)
    assert EQUATION_HEADER in enriched.pages[0].text
    assert "E_p[f]" in enriched.pages[0].text
    assert enriched.pages[0].text.startswith(paper.pages[0].text)  # original kept
    assert EQUATION_HEADER not in enriched.pages[1].text
    assert "E_p[f]" in enriched.full_text  # full_text rebuilt
    assert enriched.n_tokens > paper.n_tokens


def test_quote_from_transcription_verifies(paper: Paper, tmp_path: Path) -> None:
    client = make_client("$$ E_p[f] = \\frac{E_r[fw]}{E_r[w]} $$")
    enriched = enrich_paper(paper, pages=[2], client=client, cache_dir=tmp_path)
    citation = verify_quote("E_p[f] = \\frac{E_r[fw]}{E_r[w]}", enriched)
    assert citation.verified
    assert citation.page == 2


def test_transcription_cached_on_disk(paper: Paper, tmp_path: Path) -> None:
    client = make_client("$$ x = 1 $$")
    enrich_paper(paper, pages=[1], client=client, cache_dir=tmp_path)
    assert client.messages.create.call_count == 1
    assert len(list(tmp_path.glob("*_p001.txt"))) == 1

    fresh_client = make_client("$$ SHOULD NOT BE CALLED $$")
    again = enrich_paper(paper, pages=[1], client=fresh_client, cache_dir=tmp_path)
    fresh_client.messages.create.assert_not_called()
    assert "x = 1" in again.pages[0].text


def test_none_response_leaves_page_unchanged(paper: Paper, tmp_path: Path) -> None:
    client = make_client("NONE")
    enriched = enrich_paper(paper, pages=[1], client=client, cache_dir=tmp_path)
    assert enriched.pages[0].text == paper.pages[0].text


def test_request_contains_page_image(paper: Paper, tmp_path: Path) -> None:
    client = make_client("NONE")
    enrich_paper(paper, pages=[1], client=client, cache_dir=tmp_path)
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    image_blocks = [b for b in content if b["type"] == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["source"]["type"] == "base64"
    assert image_blocks[0]["source"]["media_type"] == "image/png"
    assert len(image_blocks[0]["source"]["data"]) > 1000


def test_no_flagged_pages_returns_paper_unchanged(tmp_path: Path) -> None:
    paper = make_paper([PROSE])
    client = make_client("SHOULD NOT BE CALLED")
    assert enrich_paper(paper, client=client, cache_dir=tmp_path) is paper
    client.messages.create.assert_not_called()


# --- figures ----------------------------------------------------------------

DESCRIPTION = (
    "A bar chart of accuracy against training steps. The y-axis is accuracy and the "
    "x-axis is the number of steps. The bars rise from left to right."
)


@pytest.fixture()
def figure_paper() -> Paper:
    return parse_pdf(FIXTURES / "figure_page.pdf")


def test_detects_figure_region_and_pairs_its_caption(figure_paper: Paper) -> None:
    figures = detect_figures(figure_paper)
    assert len(figures) == 1
    figure = figures[0]
    assert (figure.page, figure.number) == (1, "1")
    assert figure.caption.startswith("Figure 1: Accuracy on the held-out set")
    # the body sentence that cites the figure, not the caption repeated
    assert "As shown in Figure 1" in figure.context
    # the crop covers artwork plus caption, and stays on the page
    x0, y0, x1, y1 = figure.rect
    assert y0 < 320 < y1 and x1 - x0 > 100
    assert 0 <= x0 and y1 <= 842


def test_text_only_page_yields_no_figures(figure_paper: Paper) -> None:
    assert all(f.page != 2 for f in detect_figures(figure_paper))


def test_text_only_paper_makes_zero_vision_calls(tmp_path: Path) -> None:
    paper = make_paper([PROSE, PROSE])  # no path to a real PDF, so no detection at all
    client = make_client("SHOULD NOT BE CALLED")
    assert describe_figures(paper, figures=[], client=client, cache_dir=tmp_path) == []
    client.messages.create.assert_not_called()


def test_figure_description_spliced_into_page_text(figure_paper: Paper, tmp_path: Path) -> None:
    client = make_client(DESCRIPTION)
    enriched = enrich_paper(figure_paper, client=client, cache_dir=tmp_path, figures=True)
    page_one = enriched.pages[0].text
    assert FIGURE_HEADER in page_one
    assert "[FIGURE 1: A bar chart of accuracy" in page_one
    assert page_one.startswith(figure_paper.pages[0].text)  # original kept
    assert FIGURE_HEADER not in enriched.pages[1].text


def test_figure_prompt_gets_the_image_caption_and_context(
    figure_paper: Paper, tmp_path: Path
) -> None:
    client = make_client(DESCRIPTION)
    describe_figures(figure_paper, client=client, cache_dir=tmp_path)
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "text"]
    assert len(content[0]["source"]["data"]) > 1000
    prompt = content[1]["text"]
    assert "Figure 1: Accuracy on the held-out set" in prompt
    assert "As shown in Figure 1" in prompt


def test_figure_descriptions_cached_on_disk(figure_paper: Paper, tmp_path: Path) -> None:
    client = make_client(DESCRIPTION)
    describe_figures(figure_paper, client=client, cache_dir=tmp_path)
    assert client.messages.create.call_count == 1
    assert len(list(tmp_path.glob("*_p001_fig1.txt"))) == 1

    fresh = make_client("SHOULD NOT BE CALLED")
    again = describe_figures(figure_paper, client=fresh, cache_dir=tmp_path)
    fresh.messages.create.assert_not_called()
    assert again[0].cached is True
    assert "bar chart" in again[0].text


def test_none_response_splices_nothing(figure_paper: Paper, tmp_path: Path) -> None:
    client = make_client("NONE")
    enriched = enrich_paper(figure_paper, client=client, cache_dir=tmp_path, figures=True)
    assert enriched.pages[0].text == figure_paper.pages[0].text


def test_figures_are_opt_in(figure_paper: Paper, tmp_path: Path) -> None:
    """Every caller of enrich_paper would otherwise start paying for figures."""
    client = make_client("SHOULD NOT BE CALLED")
    enrich_paper(figure_paper, client=client, cache_dir=tmp_path)
    client.messages.create.assert_not_called()


def test_figure_quote_is_marked_as_figure_derived(figure_paper: Paper, tmp_path: Path) -> None:
    client = make_client(DESCRIPTION)
    enriched = enrich_paper(figure_paper, client=client, cache_dir=tmp_path, figures=True)

    from_figure = verify_quote("The bars rise from left to right", enriched)
    assert from_figure.verified and from_figure.source_type == "figure"

    from_prose = verify_quote("accuracy improves with more training steps", enriched)
    assert from_prose.verified and from_prose.source_type == "text"
