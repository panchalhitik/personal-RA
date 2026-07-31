"""Generate the synthetic test PDFs. Run once; the PDFs are committed.

Usage: python tests/fixtures/generate_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import fitz

FIXTURES = Path(__file__).parent
HEADER = "SynthConf 2026 Proceedings"


def make_two_column(path: Path) -> None:
    """3-page PDF, two columns, two blocks per column so a naive y-sort of
    blocks would interleave the columns. Includes a repeated header, bare
    page-number footers, a double space, and a word hyphenated across lines."""
    doc = fitz.open()
    for i in range(1, 4):
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(fitz.Rect(50, 25, 545, 45), HEADER, fontsize=9)
        if i == 1:
            page.insert_textbox(
                fitz.Rect(50, 48, 545, 78), "A Synthetic Two Column Paper", fontsize=16
            )
            # rotated margin stamp, like arXiv's — must be excluded from output
            page.insert_textbox(
                fitz.Rect(10, 100, 32, 700),
                "arXiv:0000.00000v1 [cs.ZZ] 1 Jan 2026",
                fontsize=12,
                rotate=90,
            )
        page.insert_textbox(
            fitz.Rect(50, 800, 545, 825), str(i), fontsize=9, align=fitz.TEXT_ALIGN_CENTER
        )
        # left column: two stacked blocks
        page.insert_textbox(
            fitz.Rect(50, 85, 290, 300),
            f"alpha  bravo studies attention mechanisms in depth on page {i}",
            fontsize=11,
        )
        page.insert_textbox(
            fitz.Rect(50, 320, 290, 560),
            f"charlie delta the model uses a trans-\nformer architecture on page {i}",
            fontsize=11,
        )
        # right column: two stacked blocks at the same heights
        page.insert_textbox(
            fitz.Rect(305, 85, 545, 300),
            f"echo foxtrot reports experimental results here on page {i}",
            fontsize=11,
        )
        page.insert_textbox(
            fitz.Rect(305, 320, 545, 560),
            f"golf hotel concludes with future work notes on page {i}",
            fontsize=11,
        )
    doc.save(path)
    doc.close()


if __name__ == "__main__":
    make_two_column(FIXTURES / "two_column.pdf")
    print("wrote", FIXTURES / "two_column.pdf")
