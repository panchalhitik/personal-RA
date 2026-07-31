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


def _is_horizontal(line: dict) -> bool:
    dx, dy = line["dir"]
    return abs(dx - 1) < 0.01 and abs(dy) < 0.01


def _ordered_text_blocks(page: fitz.Page) -> list[str]:
    """Text blocks in reading order. Rotated text (e.g. arXiv margin stamps)
    is dropped — it is never body text."""
    mid = page.rect.width / 2
    blocks: list[tuple[int, float, float, str]] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        lines = [
            "".join(span["text"] for span in line["spans"])
            for line in block.get("lines", [])
            if _is_horizontal(line)
        ]
        text = "\n".join(line for line in lines if line.strip())
        if not text.strip():
            continue
        x0, y0 = block["bbox"][0], block["bbox"][1]
        column = 0 if x0 < mid else 1
        blocks.append((column, y0, x0, text))
    return [text for _, _, _, text in sorted(blocks, key=lambda b: b[:3])]


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
    # Whole lines, not spans: small-caps titles mix two font sizes on one line,
    # so a line counts as title-sized if its *largest* span is title-sized.
    lines: list[tuple[float, float, float, str]] = []  # (max_span_size, y, x, text)
    page = doc[0]
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            if not _is_horizontal(line):
                continue
            text = "".join(span["text"] for span in line.get("spans", [])).strip()
            if text and line["bbox"][1] < page.rect.height / 2:
                size = max(span["size"] for span in line["spans"])
                lines.append((size, line["bbox"][1], line["bbox"][0], text))
    if lines:
        max_size = max(ln[0] for ln in lines)
        parts = [
            ln[3] for ln in sorted(lines, key=lambda ln: (ln[1], ln[2])) if ln[0] > max_size - 0.5
        ]
        title = _normalize(" ".join(parts))
        if title:
            return title[:200]
    meta_title = (doc.metadata or {}).get("title", "").strip()
    return meta_title or Path(doc.name).stem


def build_full_text(pages: list[Page]) -> str:
    return "\n\n".join(f"[PAGE {p.number}]\n\n{p.text}" for p in pages)


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
    full_text = build_full_text(pages)
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
