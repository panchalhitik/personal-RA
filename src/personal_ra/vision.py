"""Vision-based math transcription for equation-heavy pages.

Text extraction garbles display equations (PDF math has no linear reading
order). For pages that look equation-heavy, we render the page to an image,
ask Claude to transcribe the display math to LaTeX, and append the
transcription to that page's text. Appending (rather than replacing) keeps
the original text intact, so quote verification keeps working — and quotes
taken from the transcription verify too, since it becomes part of the page.

Transcriptions are cached on disk keyed by (file hash, page number), so each
paper pays the vision cost once.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from pathlib import Path

import anthropic
import fitz
from dotenv import load_dotenv

from personal_ra.parse import Page, Paper, build_full_text, parse_pdf

MODEL = "claude-sonnet-4-5"
CACHE_DIR = Path(".cache") / "vision"
RENDER_DPI = 110
EQUATION_HEADER = "[TRANSCRIBED EQUATIONS (from page image)]"

# Characters that almost never appear in prose but are common in extracted math.
_MATH_CHARS = set("∈∉∑∏∫≤≥≠≈∼≃∝∞∂∇√±·×÷→←↔⇒⇔⊆⊂∪∩∀∃⊤⊥⟨⟩̸αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩℓ")

TRANSCRIBE_PROMPT = """This is one page of an academic paper. Transcribe every display \
equation on this page (numbered or unnumbered) into LaTeX, in reading order.

Format each as:
(after: "<3-8 words of the prose immediately before the equation>")
$$ <latex> $$

Transcribe only display equations, not inline math or tables. If the page contains no \
display equations, reply with exactly: NONE"""


def _math_score(text: str) -> float:
    """Heuristic score for how equation-garbled a page's extracted text is."""
    if not text:
        return 0.0
    symbols = sum(1 for c in text if c in _MATH_CHARS)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Garbled display math shows up as many very short lines ("X", "=", "a∈X")
    short_lines = sum(1 for ln in lines if len(ln) <= 4)
    return 1000 * symbols / len(text) + 100 * short_lines / max(len(lines), 1)


def detect_equation_pages(paper: Paper, threshold: float = 8.0) -> list[int]:
    """Page numbers whose extracted text looks equation-heavy."""
    return [p.number for p in paper.pages if _math_score(p.text) >= threshold]


def _paper_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _render_page_png(doc: fitz.Document, page_number: int) -> bytes:
    return doc[page_number - 1].get_pixmap(dpi=RENDER_DPI).tobytes("png")


def _transcribe_page(client: anthropic.Anthropic, png: bytes) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.standard_b64encode(png).decode(),
                        },
                    },
                    {"type": "text", "text": TRANSCRIBE_PROMPT},
                ],
            }
        ],
    )
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()


def enrich_paper(
    paper: Paper,
    pages: list[int] | None = None,
    client: anthropic.Anthropic | None = None,
    cache_dir: Path = CACHE_DIR,
) -> Paper:
    """Append LaTeX transcriptions to equation-heavy pages. Returns a new Paper."""
    page_numbers = detect_equation_pages(paper) if pages is None else pages
    if not page_numbers:
        return paper

    file_hash = _paper_hash(paper.path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    new_pages = list(paper.pages)
    doc: fitz.Document | None = None
    try:
        for n in page_numbers:
            cache_file = cache_dir / f"{file_hash}_p{n:03d}.txt"
            if cache_file.exists():
                raw = cache_file.read_text(encoding="utf-8")
            else:
                client = client or anthropic.Anthropic()
                doc = doc or fitz.open(paper.path)
                raw = _transcribe_page(client, _render_page_png(doc, n))
                cache_file.write_text(raw, encoding="utf-8")
            if raw.strip().upper() == "NONE" or not raw.strip():
                continue
            old = new_pages[n - 1]
            new_pages[n - 1] = Page(
                number=old.number, text=f"{old.text}\n{EQUATION_HEADER}\n{raw.strip()}"
            )
    finally:
        if doc is not None:
            doc.close()

    full_text = build_full_text(new_pages)
    return Paper(
        path=paper.path,
        title=paper.title,
        pages=new_pages,
        full_text=full_text,
        n_tokens=len(full_text) // 4,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Detect and transcribe equation-heavy pages.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--detect-only", action="store_true", help="print page scores, no API calls")
    args = ap.parse_args(argv)

    load_dotenv()
    paper = parse_pdf(args.pdf)
    if args.detect_only:
        for p in paper.pages:
            score = _math_score(p.text)
            flag = " <-- equation-heavy" if score >= 8.0 else ""
            print(f"page {p.number:3d}  score {score:6.1f}{flag}")
        return

    flagged = detect_equation_pages(paper)
    print(f"Equation-heavy pages: {flagged or 'none'}")
    if flagged:
        enriched = enrich_paper(paper)
        for n in flagged:
            text = enriched.pages[n - 1].text
            idx = text.find(EQUATION_HEADER)
            if idx != -1:
                print(f"\n=== page {n} ===\n{text[idx:]}")


if __name__ == "__main__":
    main()
