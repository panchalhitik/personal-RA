from pathlib import Path

import pytest

from personal_ra.parse import Paper, parse_pdf

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def paper() -> Paper:
    return parse_pdf(FIXTURES / "two_column.pdf")


def test_two_column_reading_order(paper: Paper) -> None:
    text = paper.pages[0].text
    order = [text.index(w) for w in ("alpha", "charlie", "echo", "golf")]
    assert order == sorted(order), f"columns interleaved: {text!r}"
    # everything in the left column must come before the right column starts
    assert text.index("delta") < text.index("echo")


def test_pages_are_one_indexed(paper: Paper) -> None:
    assert [p.number for p in paper.pages] == [1, 2, 3]


def test_hyphenated_words_joined(paper: Paper) -> None:
    assert "transformer" in paper.pages[0].text
    assert "trans- former" not in paper.pages[0].text


def test_whitespace_normalized(paper: Paper) -> None:
    for page in paper.pages:
        assert "  " not in page.text
        assert "\t" not in page.text


def test_repeated_headers_stripped(paper: Paper) -> None:
    for page in paper.pages:
        assert "SynthConf" not in page.text


def test_page_number_footers_stripped(paper: Paper) -> None:
    for page in paper.pages:
        assert not any(line.strip().isdigit() for line in page.text.splitlines())


def test_full_text_has_page_markers(paper: Paper) -> None:
    for n in (1, 2, 3):
        assert f"[PAGE {n}]" in paper.full_text


def test_title_from_largest_font(paper: Paper) -> None:
    assert paper.title == "A Synthetic Two Column Paper"


def test_rotated_watermark_excluded(paper: Paper) -> None:
    assert "arXiv:0000" not in paper.full_text
    assert "arXiv" not in paper.title


def test_token_estimate_positive(paper: Paper) -> None:
    assert paper.n_tokens == len(paper.full_text) // 4 > 0
