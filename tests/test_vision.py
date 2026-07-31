from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from conftest import make_paper
from personal_ra.cite import verify_quote
from personal_ra.parse import Paper, parse_pdf
from personal_ra.vision import EQUATION_HEADER, detect_equation_pages, enrich_paper

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
