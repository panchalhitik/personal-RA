"""PDF -> structured text via PyMuPDF block extraction.

Reading order is recovered per page by assigning each text block to a column
(left/right of the page midline) and sorting by (column, y, x), so two-column
academic layouts read down each column instead of interleaving across them.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class Page:
    number: int  # 1-indexed, matches what a PDF reader shows
    text: str


@dataclass
class Paper:
    path: Path
    title: str
    pages: list[Page]
    full_text: str  # pages joined with [PAGE N] markers
    n_tokens: int  # rough estimate, chars / 4


_PAGE_NUMBER_RE = re.compile(r"(page\s*)?\d{1,4}", re.IGNORECASE)


def _normalize(text: str) -> str:
    """Join words hyphenated across line breaks, collapse whitespace runs."""
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _ordered_text_blocks(page: fitz.Page) -> list[str]:
    mid = page.rect.width / 2
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]

    def key(b: tuple) -> tuple[int, float, float]:
        x0, y0 = b[0], b[1]
        column = 0 if x0 < mid else 1
        return (column, y0, x0)

    return [b[4] for b in sorted(blocks, key=key)]


def _strip_repeated_blocks(raw_pages: list[list[str]]) -> list[list[str]]:
    """Drop headers/footers repeated on most pages, and bare page numbers."""
    repeated: set[str] = set()
    n = len(raw_pages)
    if n >= 3:
        counts = Counter(t for blocks in raw_pages for t in set(blocks))
        threshold = max(2, math.ceil(0.6 * n))
        repeated = {t for t, c in counts.items() if c >= threshold and len(t) < 120}
    return [
        [t for t in blocks if t not in repeated and not _PAGE_NUMBER_RE.fullmatch(t)]
        for blocks in raw_pages
    ]


def _guess_title(doc: fitz.Document) -> str:
    """Best-effort title: largest-font text in the top half of page 1."""
    spans: list[tuple[float, float, float, str]] = []  # (size, y, x, text)
    page = doc[0]
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if text and span["bbox"][1] < page.rect.height / 2:
                    spans.append((span["size"], span["bbox"][1], span["bbox"][0], text))
    if spans:
        max_size = max(s[0] for s in spans)
        parts = [s[3] for s in sorted(spans, key=lambda s: (s[1], s[2])) if s[0] > max_size - 0.5]
        title = _normalize(" ".join(parts))
        if title:
            return title[:200]
    meta_title = (doc.metadata or {}).get("title", "").strip()
    return meta_title or Path(doc.name).stem


def parse_pdf(path: str | Path) -> Paper:
    path = Path(path)
    doc = fitz.open(path)
    try:
        raw_pages = [
            [t for t in (_normalize(b) for b in _ordered_text_blocks(page)) if t] for page in doc
        ]
        cleaned = _strip_repeated_blocks(raw_pages)
        pages = [Page(number=i + 1, text="\n".join(blocks)) for i, blocks in enumerate(cleaned)]
        title = _guess_title(doc)
    finally:
        doc.close()
    full_text = "\n\n".join(f"[PAGE {p.number}]\n\n{p.text}" for p in pages)
    return Paper(
        path=path,
        title=title,
        pages=pages,
        full_text=full_text,
        n_tokens=len(full_text) // 4,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Parse a PDF and show a summary.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--debug", action="store_true", help="dump per-page text to debug/")
    args = ap.parse_args(argv)

    paper = parse_pdf(args.pdf)
    print(f"Title:   {paper.title}")
    print(f"Pages:   {len(paper.pages)}")
    print(f"~Tokens: {paper.n_tokens}")
    if args.debug:
        out_dir = Path("debug") / paper.path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for page in paper.pages:
            (out_dir / f"page_{page.number:03d}.txt").write_text(page.text, encoding="utf-8")
        print(f"Debug dump written to {out_dir}")


if __name__ == "__main__":
    main()
