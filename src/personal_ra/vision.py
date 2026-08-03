"""Vision for the two things text extraction cannot recover: display math and
figures.

Text extraction garbles display equations (PDF math has no linear reading
order). For pages that look equation-heavy, we render the page to an image,
ask Claude to transcribe the display math to LaTeX, and append the
transcription to that page's text. Appending (rather than replacing) keeps
the original text intact, so quote verification keeps working — and quotes
taken from the transcription verify too, since it becomes part of the page.

Figures work the same way, one rung less certain. We find each figure by its
caption, crop the region, and ask for a structured description that is spliced
back as "[FIGURE 3: ...]". A transcribed equation is *extracted* content; a
described figure is model *inference* about a picture, so it is marked
distinctly everywhere downstream (parse.FIGURE_HEADER, cite's source_type,
Chroma's content_type) and must never be read as if it were the paper's words.

Both are cached on disk keyed by (file hash, page number), so each paper pays
the vision cost once.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import fitz
from dotenv import load_dotenv

from personal_ra.parse import (
    EQUATION_HEADER,
    FIGURE_HEADER,
    Page,
    Paper,
    build_full_text,
    parse_pdf,
)

MODEL = "claude-sonnet-4-5"
CACHE_DIR = Path(".cache") / "vision"
RENDER_DPI = 110

# claude-sonnet-4-5 list price, USD per million tokens. Only used to report what
# a run cost — never to decide anything.
PRICE_IN_PER_MTOK = 3.0
PRICE_OUT_PER_MTOK = 15.0

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


# --- figures ----------------------------------------------------------------

# A caption block, not a body-text mention: "Figure 3:" / "Fig. 2 —" start the
# block, and a delimiter must follow the number so "Figure 3 shows that ..." in
# a paragraph is not mistaken for one.
_FIG_CAPTION_RE = re.compile(r"^(?:figure|fig\.?)\s*(\d{1,2}[a-z]?)\s*[:.–—-]\s*(\S.*)", re.I)
FIGURE_DPI = 150
_MAX_ABOVE = 420.0  # pt to search above a caption for its artwork
_MAX_BELOW = 70.0  # ... and below, for the rarer caption-on-top layout
_CLUSTER_GAP = 45.0  # vertical gap that separates one figure's parts from another's
_MIN_FIGURE_AREA = 4000.0  # pt², below which the "figure" is a rule or a glyph
_CROP_PAD = 4.0

DESCRIBE_PROMPT = """This image is a cropped region of a page from an academic paper: \
figure {number} and its caption.

Caption: {caption}
How the text refers to it: {context}

Describe the figure for a search index in 2-4 sentences of plain prose. Cover, in order:
1. What kind of figure it is (line plot, bar chart, scatter, heatmap, architecture \
diagram, screenshot, qualitative example) and what it shows.
2. The axes — what each measures, with units and range. Skip this for diagrams.
3. The main trend or comparison, naming the series or conditions by their legend labels.
4. Any numbers you can read directly off the figure (axis ticks, annotated data labels, \
values in a legend).

Rules:
- Never estimate a value from where a point sits. Report a number only if it is printed \
on the figure.
- Do not repeat the caption verbatim, and do not speculate about what the result implies.
- No markdown, no bullet points.

If this image is not a figure — a table, a page of prose, or blank — reply with exactly: \
NONE"""


@dataclass
class Figure:
    page: int  # 1-indexed
    number: str  # as printed: "3", "12b"
    caption: str
    rect: tuple[float, float, float, float]  # crop region in PDF points
    context: str = ""  # the sentence in the body text that refers to it


@dataclass
class FigureDescription:
    figure: Figure
    text: str  # "" when the model said NONE
    cached: bool
    usage: dict = field(default_factory=dict)  # input/output tokens, cost_usd


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _block_text(block: dict) -> str:
    return _norm(
        " ".join("".join(s["text"] for s in line["spans"]) for line in block.get("lines", []))
    )


def _hoverlap(a: fitz.Rect, b: fitz.Rect) -> float:
    """Shared horizontal span as a fraction of the narrower rect."""
    span = min(a.x1, b.x1) - max(a.x0, b.x0)
    return span / max(min(a.width, b.width), 1e-6)


def _graphic_rects(page: fitz.Page) -> list[fitz.Rect]:
    """Raster image blocks plus vector drawings — academic plots are usually
    vector, so image blocks alone find nothing on most papers."""
    page_area = page.rect.get_area()
    rects: list[fitz.Rect] = []
    candidates = [
        fitz.Rect(b["bbox"]) for b in page.get_text("dict")["blocks"] if b.get("type") == 1
    ]
    candidates += [fitz.Rect(d["rect"]) for d in page.get_drawings()]
    for rect in candidates:
        if rect.get_area() > 0.9 * page_area:  # page border or full-page background fill
            continue
        # Running-header/footer rules span the text width and would drag the crop
        # up into the header on any page where a figure sits near the top.
        if rect.height <= 2 and rect.width > 0.5 * page.rect.width:
            continue
        # Axis lines are zero-height; give them extent so unions still see them.
        if rect.width <= 0:
            rect.x1 = rect.x0 + 0.5
        if rect.height <= 0:
            rect.y1 = rect.y0 + 0.5
        rects.append(rect)
    return rects


def _artwork_rect(caption: fitz.Rect, graphics: list[fitz.Rect]) -> fitz.Rect | None:
    """The artwork belonging to one caption: graphics that share the caption's
    column and sit just above (or just below) it, clustered by vertical gaps so
    a stray rule further up the page can't inflate the crop."""
    near = [
        g
        for g in graphics
        if _hoverlap(g, caption) > 0.25
        and (
            caption.y0 - _MAX_ABOVE <= g.y1 <= caption.y0 + 2
            or caption.y1 - 2 <= g.y0 <= caption.y1 + _MAX_BELOW
        )
    ]
    if not near:
        return None

    clusters: list[fitz.Rect] = []
    for g in sorted(near, key=lambda r: r.y0):
        if clusters and g.y0 - clusters[-1].y1 <= _CLUSTER_GAP:
            clusters[-1] |= g
        else:
            clusters.append(fitz.Rect(g))

    best = min(clusters, key=lambda c: min(abs(caption.y0 - c.y1), abs(c.y0 - caption.y1)))
    return best if best.get_area() >= _MIN_FIGURE_AREA else None


def _figure_context(paper: Paper, page_number: int, number: str) -> str:
    """The sentence in the body text that refers to this figure. A caption alone
    often says less than the sentence that cites it."""
    pattern = re.compile(
        rf"([^.\n]{{0,220}}\b(?:figure|fig\.?)\s*{re.escape(number)}\b[^.\n]{{0,220}}\.)", re.I
    )
    for offset in (0, -1, 1):  # this page first, then its neighbours
        index = page_number - 1 + offset
        if not 0 <= index < len(paper.pages):
            continue
        for match in pattern.finditer(paper.pages[index].text):
            sentence = _norm(match.group(1))
            if not _FIG_CAPTION_RE.match(sentence):  # skip the caption itself
                return sentence
    return "(the body text does not refer to it directly)"


def detect_figures(paper: Paper) -> list[Figure]:
    """Figures found by caption, paired with the artwork above or below them."""
    figures: list[Figure] = []
    doc = fitz.open(paper.path)
    try:
        for page in doc:
            graphics = _graphic_rects(page)
            if not graphics:
                continue
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                match = _FIG_CAPTION_RE.match(_block_text(block))
                if not match:
                    continue
                caption_rect = fitz.Rect(block["bbox"])
                artwork = _artwork_rect(caption_rect, graphics)
                if artwork is None:
                    continue
                crop = (artwork | caption_rect) + (-_CROP_PAD, -_CROP_PAD, _CROP_PAD, _CROP_PAD)
                crop &= page.rect
                figures.append(
                    Figure(
                        page=page.number + 1,
                        number=match.group(1),
                        caption=_norm(match.group(0)),
                        rect=(crop.x0, crop.y0, crop.x1, crop.y1),
                    )
                )
    finally:
        doc.close()

    for figure in figures:
        figure.context = _figure_context(paper, figure.page, figure.number)
    return figures


def _render_clip_png(doc: fitz.Document, figure: Figure) -> bytes:
    page = doc[figure.page - 1]
    return page.get_pixmap(clip=fitz.Rect(*figure.rect), dpi=FIGURE_DPI).tobytes("png")


def _describe_figure(client: anthropic.Anthropic, png: bytes, figure: Figure) -> tuple[str, dict]:
    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
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
                    {
                        "type": "text",
                        "text": DESCRIBE_PROMPT.format(
                            number=figure.number,
                            caption=figure.caption,
                            context=figure.context,
                        ),
                    },
                ],
            }
        ],
    )
    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
    usage = getattr(response, "usage", None)
    tokens_in = getattr(usage, "input_tokens", 0) or 0
    tokens_out = getattr(usage, "output_tokens", 0) or 0
    return text, {
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "cost_usd": tokens_in / 1e6 * PRICE_IN_PER_MTOK + tokens_out / 1e6 * PRICE_OUT_PER_MTOK,
    }


def describe_figures(
    paper: Paper,
    figures: list[Figure] | None = None,
    client: anthropic.Anthropic | None = None,
    cache_dir: Path = CACHE_DIR,
) -> list[FigureDescription]:
    """Describe each detected figure, reusing the on-disk cache. Only figures
    missing from the cache cost anything."""
    figures = detect_figures(paper) if figures is None else figures
    if not figures:
        return []

    file_hash = _paper_hash(paper.path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[FigureDescription] = []
    doc: fitz.Document | None = None
    try:
        for figure in figures:
            cache_file = cache_dir / f"{file_hash}_p{figure.page:03d}_fig{figure.number}.txt"
            if cache_file.exists():
                raw, usage, cached = cache_file.read_text(encoding="utf-8"), {}, True
            else:
                client = client or anthropic.Anthropic()
                doc = doc or fitz.open(paper.path)
                raw, usage = _describe_figure(client, _render_clip_png(doc, figure), figure)
                cache_file.write_text(raw, encoding="utf-8")
                cached = False
            text = "" if raw.strip().upper() == "NONE" else _norm(raw)
            results.append(FigureDescription(figure=figure, text=text, cached=cached, usage=usage))
    finally:
        if doc is not None:
            doc.close()
    return results


def _splice(pages: list[Page], page_number: int, header: str, body: str) -> None:
    """Append vision output under its header, one header per page."""
    old = pages[page_number - 1]
    if header in old.text:
        pages[page_number - 1] = Page(number=old.number, text=f"{old.text}\n{body}")
    else:
        pages[page_number - 1] = Page(number=old.number, text=f"{old.text}\n{header}\n{body}")


def _rebuild(paper: Paper, pages: list[Page]) -> Paper:
    full_text = build_full_text(pages)
    return Paper(
        path=paper.path,
        title=paper.title,
        pages=pages,
        full_text=full_text,
        n_tokens=len(full_text) // 4,
    )


def enrich_paper(
    paper: Paper,
    pages: list[int] | None = None,
    client: anthropic.Anthropic | None = None,
    cache_dir: Path = CACHE_DIR,
    equations: bool = True,
    figures: bool = False,
) -> Paper:
    """Append LaTeX transcriptions to equation-heavy pages and, when asked,
    descriptions of detected figures. Returns a new Paper.

    Figures are opt-in: every caller of this function would otherwise start
    paying for figure vision the moment a paper is opened.
    """
    page_numbers = (detect_equation_pages(paper) if pages is None else pages) if equations else []
    figure_descriptions = (
        describe_figures(paper, client=client, cache_dir=cache_dir) if figures else []
    )
    if not page_numbers and not figure_descriptions:
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
            _splice(new_pages, n, EQUATION_HEADER, raw.strip())
    finally:
        if doc is not None:
            doc.close()

    for described in figure_descriptions:
        if described.text:
            _splice(
                new_pages,
                described.figure.page,
                FIGURE_HEADER,
                f"[FIGURE {described.figure.number}: {described.text}]",
            )

    return _rebuild(paper, new_pages)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Detect, transcribe, and describe page images.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--detect-only", action="store_true", help="print what was found, no API calls")
    ap.add_argument("--figures-only", action="store_true", help="figures only, skip equations")
    ap.add_argument("--figures", action="store_true", help="describe figures as well as equations")
    args = ap.parse_args(argv)

    # Captions carry Greek letters and typographic quotes; the Windows console
    # defaults to cp1252 and raises on them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    load_dotenv()
    paper = parse_pdf(args.pdf)
    want_figures = args.figures or args.figures_only
    want_equations = not args.figures_only

    if args.detect_only:
        if want_equations:
            for p in paper.pages:
                score = _math_score(p.text)
                flag = " <-- equation-heavy" if score >= 8.0 else ""
                print(f"page {p.number:3d}  score {score:6.1f}{flag}")
        if want_figures:
            figures = detect_figures(paper)
            print(f"\nFigures detected: {len(figures)}")
            for f in figures:
                w, h = f.rect[2] - f.rect[0], f.rect[3] - f.rect[1]
                print(
                    f"  p{f.page:<3d} Figure {f.number:<4s} {w:5.0f}x{h:5.0f}pt  {f.caption[:70]}"
                )
        return

    if want_equations:
        flagged = detect_equation_pages(paper)
        print(f"Equation-heavy pages: {flagged or 'none'}")
        if flagged:
            enriched = enrich_paper(paper, figures=False)
            for n in flagged:
                text = enriched.pages[n - 1].text
                idx = text.find(EQUATION_HEADER)
                if idx != -1:
                    print(f"\n=== page {n} ===\n{text[idx:]}")

    if want_figures:
        described = describe_figures(paper)
        cost = sum(d.usage.get("cost_usd", 0.0) for d in described)
        billed = sum(1 for d in described if not d.cached)
        for d in described:
            mark = "cached" if d.cached else f"${d.usage.get('cost_usd', 0.0):.4f}"
            body = d.text or "(model said this is not a figure)"
            print(f"\n=== p{d.figure.page} Figure {d.figure.number} [{mark}] ===\n{body}")
        print(f"\n{len(described)} figures, {billed} billed, ${cost:.4f}")


if __name__ == "__main__":
    main()
