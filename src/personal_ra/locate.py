"""Locate quoted text on a PDF page as highlight rectangles.

Best-effort: quotes come from normalized extracted text, so an exact search can
fail (hyphenation, ligatures). We fall back to progressively shorter prefixes
of the quote; an empty result means "couldn't locate", never an error.
"""

from __future__ import annotations

from pathlib import Path

import fitz

MIN_NEEDLE_LEN = 12

Rect = tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points, top-left origin


def _candidates(quote: str) -> list[str]:
    words = quote.split()
    candidates = [" ".join(words)]
    for n in (8, 5):
        if len(words) > n:
            candidates.append(" ".join(words[:n]))
    return [c for c in candidates if len(c) >= MIN_NEEDLE_LEN]


def locate_quote(pdf_path: Path, page_number: int, quote: str) -> list[Rect]:
    if not quote.strip():
        return []
    doc = fitz.open(pdf_path)
    try:
        if not 1 <= page_number <= len(doc):
            return []
        page = doc[page_number - 1]
        for needle in _candidates(quote):
            rects = page.search_for(needle)
            if rects:
                return [(r.x0, r.y0, r.x1, r.y1) for r in rects]
    finally:
        doc.close()
    return []
