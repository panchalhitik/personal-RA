from pathlib import Path

from personal_ra.locate import locate_quote

FIXTURES = Path(__file__).parent / "fixtures"
PDF = FIXTURES / "two_column.pdf"


def test_locates_exact_text() -> None:
    rects = locate_quote(PDF, 1, "echo foxtrot reports experimental results here")
    assert rects
    for x0, y0, x1, y1 in rects:
        assert x1 > x0 and y1 > y0


def test_falls_back_to_prefix_when_tail_missing() -> None:
    rects = locate_quote(PDF, 1, "echo foxtrot reports experimental results here zzz qqq www vvv")
    assert rects


def test_absent_text_returns_empty() -> None:
    assert locate_quote(PDF, 1, "completely absent phrase xyzzy plugh") == []


def test_out_of_range_page_returns_empty() -> None:
    assert locate_quote(PDF, 99, "echo foxtrot reports experimental results") == []


def test_empty_quote_returns_empty() -> None:
    assert locate_quote(PDF, 1, "   ") == []
