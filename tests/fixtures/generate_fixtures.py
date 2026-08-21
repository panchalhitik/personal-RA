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


def make_figure_page(path: Path) -> None:
    """2-page PDF. Page 1 has a vector 'chart' (axes plus bars) with a caption
    below it and a body sentence referring to it; page 2 is prose only, so a
    detector that fires on anything would be caught."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(50, 50, 545, 90),
        "As shown in Figure 1, accuracy improves with more training steps.",
        fontsize=11,
    )
    # the artwork: axes plus three bars
    page.draw_line(fitz.Point(90, 320), fitz.Point(90, 130), width=1.2)
    page.draw_line(fitz.Point(90, 320), fitz.Point(400, 320), width=1.2)
    for i, height in enumerate((60, 110, 160)):
        x = 130 + i * 90
        page.draw_rect(fitz.Rect(x, 320 - height, x + 55, 320), fill=(0.2, 0.4, 0.8))
    page.insert_textbox(
        fitz.Rect(50, 340, 545, 400),
        "Figure 1: Accuracy on the held-out set against the number of training steps.",
        fontsize=10,
    )
    page.insert_textbox(
        fitz.Rect(50, 420, 545, 700),
        "The remaining sections describe the experimental setup in detail.",
        fontsize=11,
    )

    prose = doc.new_page(width=595, height=842)
    prose.insert_textbox(
        fitz.Rect(50, 50, 545, 400),
        "This page contains only prose about the evaluation protocol and has no figures "
        "of any kind, so no figure region should ever be detected on it.",
        fontsize=11,
    )
    doc.save(path)
    doc.close()


def make_two_figures_one_page(path: Path) -> None:
    """1-page PDF with two figures stacked close together, each captioned below
    its own artwork. The gap between the two figures is smaller than the cluster
    gap, so a detector that doesn't treat captions as barriers merges them and
    hands both captions the whole page."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # figure 1: two bars, caption below
    for i, height in enumerate((80, 140)):
        x = 120 + i * 100
        page.draw_rect(fitz.Rect(x, 200 - height, x + 60, 200), fill=(0.8, 0.3, 0.2))
    page.insert_textbox(
        fitz.Rect(50, 203, 545, 223), "Figure 1: Throughput by configuration.", fontsize=10
    )

    # figure 2: axes and a trend line, starting 33pt below figure 1's artwork —
    # inside the cluster gap, so only figure 1's caption keeps them apart
    page.draw_line(fitz.Point(120, 330), fitz.Point(400, 233), width=1.5)
    page.draw_line(fitz.Point(120, 330), fitz.Point(400, 330), width=1.2)
    page.insert_textbox(
        fitz.Rect(50, 338, 545, 358), "Figure 2: Latency against request rate.", fontsize=10
    )
    doc.save(path)
    doc.close()


if __name__ == "__main__":
    make_two_column(FIXTURES / "two_column.pdf")
    print("wrote", FIXTURES / "two_column.pdf")
    make_figure_page(FIXTURES / "figure_page.pdf")
    print("wrote", FIXTURES / "figure_page.pdf")
    make_two_figures_one_page(FIXTURES / "two_figures.pdf")
    print("wrote", FIXTURES / "two_figures.pdf")
