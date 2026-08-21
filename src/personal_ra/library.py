"""Ingest papers into a Chroma vector store with section-aware chunking.

Chunks split on sentence boundaries within detected sections (~1000 chars,
200 overlap). Each chunk is embedded with a context prefix — paper title and
section — which is a cheap, large retrieval-quality win. Ingest is idempotent:
chunk IDs are deterministic ({paper_id}:{chunk_index}) and writes are upserts,
so re-running never duplicates.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
import fitz

from personal_ra.parse import EQUATION_HEADER, FIGURE_HEADER, Paper, parse_pdf

# (page, section, sentence, content_type) — content_type is "text", "figure", or
# "equation"; the latter two mark vision-derived content.
Unit = tuple[int, str, str, str]

PAPERS_DIR = Path("papers")
DB_PATH = Path("chroma_db")
COLLECTION = "papers"
EMBED_MODEL = "all-MiniLM-L6-v2"
TARGET_CHARS = 1000
OVERLAP_CHARS = 200

_SECTION_PATTERNS = [
    re.compile(r"^\d{1,2}(\.\d{1,2})*\.?\s+[A-Z][^\n]{0,70}$"),  # "3. Method", "3.1 Results"
    re.compile(r"^[IVXLC]+\.\s+[A-Z][^\n]{0,70}$"),  # "IV. Experiments"
    re.compile(
        r"^(abstract|introduction|related works?|background|preliminaries|"
        r"methods?|methodology|approach|experiments?|results?|evaluation|"
        r"discussion|limitations|conclusions?|references|acknowledge?ments?|"
        r"appendix(\s+\w+)?)\.?$",
        re.IGNORECASE,
    ),
]

# Split after sentence-ending punctuation followed by a plausible sentence start.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\d\"'])")
_YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
# arXiv IDs encode submission date as YYMM: 2506.05346 -> June 2025.
_ARXIV_ID_RE = re.compile(r"\b(\d{2})(0[1-9]|1[0-2])\.\d{4,5}(?!\d)")
# Dates in publication context, not just any year mentioned on the page.
_PUB_CONTEXT_RE = re.compile(
    r"(?:arxiv:\s*\d{4}\.\d{4,5}v?\d*\s*\[[^\]]+\]\s*\d{1,2}\s+\w+\s+|"
    r"(?:published|submitted|accepted|preprint|revised|updated)[^.\n]{0,40}?|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s*)"
    r"(19[89]\d|20[0-3]\d)",
    re.IGNORECASE,
)
_CONF_YEAR_RE = re.compile(
    r"(?:ICLR|NeurIPS|ICML|ACL|EMNLP|NAACL|AAAI|CVPR|ICCV|COLM|IJCAI)\s*'?\s*(20[0-3]\d)",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    id: str  # "{paper_id}:{chunk_index}" — deterministic, stable across runs
    text: str
    embed_text: str  # context-prefixed text, what actually gets embedded
    metadata: dict


def paper_id(path: Path) -> str:
    """Stable id from file content — renaming a file keeps its identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def year_from_arxiv_id(name: str) -> int | None:
    """arXiv filenames encode YYMM (2506.05346 -> 2025). Authoritative when present."""
    match = _ARXIV_ID_RE.search(name)
    if not match:
        return None
    year = 2000 + int(match.group(1))
    return year if 2007 <= year <= 2039 else None


def _year_from_pdf_metadata(path: Path) -> int | None:
    try:
        doc = fitz.open(path)
        try:
            meta = doc.metadata or {}
        finally:
            doc.close()
    except Exception:
        return None
    for field in ("creationDate", "modDate"):
        match = re.search(r"D:(\d{4})", meta.get(field) or "")
        if match:
            year = int(match.group(1))
            if 1990 <= year <= 2039:
                return year
    return None


def detect_year(paper: Paper) -> int | None:
    """Publication year, most reliable source first: arXiv ID in the filename,
    then a year in publication context on page 1 (submission line, conference
    venue, 'published/preprint'), then the PDF's own creation date.

    Deliberately does NOT fall back to 'largest year on page 1' — that reads
    citation years and produces confidently wrong values.
    """
    from_id = year_from_arxiv_id(paper.path.name)
    if from_id:
        return from_id

    if paper.pages:
        page1 = paper.pages[0].text
        candidates = [int(y) for y in _PUB_CONTEXT_RE.findall(page1)]
        candidates += [int(y) for y in _CONF_YEAR_RE.findall(page1)]
        # An arXiv ID printed in the page text works as well as one in the filename.
        in_text = year_from_arxiv_id(page1)
        if in_text:
            candidates.append(in_text)
        if candidates:
            return max(candidates)

    return _year_from_pdf_metadata(paper.path)


def _is_section_header(line: str) -> bool:
    line = line.strip()
    return bool(line) and len(line) <= 80 and any(p.match(line) for p in _SECTION_PATTERNS)


def _units(paper: Paper) -> list[Unit]:
    """(page, section, sentence, content_type) tuples in reading order.

    vision.py appends its output to the end of a page under a header, so
    everything after a header on that page carries that content type — which is
    what lets a chunk say whether it holds the paper's words or a model's
    reading of a picture.
    """
    units: list[Unit] = []
    section = ""
    for page in paper.pages:
        content_type = "text"
        for line in page.text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == FIGURE_HEADER:
                content_type = "figure"
                continue
            if stripped == EQUATION_HEADER:
                content_type = "equation"
                continue
            if _is_section_header(stripped):
                section = stripped
                continue
            for sentence in _SENT_SPLIT.split(stripped):
                if sentence.strip():
                    units.append((page.number, section, sentence.strip(), content_type))
    return units


def extract_sections(paper: Paper) -> dict[str, str]:
    """Section label -> that section's full text, in document order.

    Built from the same sentence units as chunking, so there is no chunk
    overlap duplication — this is what the MCP read_paper tool serves.
    """
    sections: dict[str, list[str]] = {}
    for _page, section, sentence, _content_type in _units(paper):
        sections.setdefault(section or "(no section)", []).append(sentence)
    return {label: " ".join(sentences) for label, sentences in sections.items()}


STRATEGIES = ("fixed", "section", "section_context")


def _fixed_windows(paper: Paper) -> list[Unit]:
    """Blind character windows (the eval baseline): TARGET_CHARS with
    OVERLAP_CHARS overlap, no section or sentence awareness."""
    units: list[Unit] = []
    for page in paper.pages:
        text = " ".join(page.text.split())
        start = 0
        while start < len(text):
            piece = text[start : start + TARGET_CHARS].strip()
            if piece:
                units.append((page.number, "", piece, "text"))
            if start + TARGET_CHARS >= len(text):
                break
            start += TARGET_CHARS - OVERLAP_CHARS
    return units


def chunk_paper(
    paper: Paper, pid: str, year: int | None, strategy: str = "section_context"
) -> list[Chunk]:
    """Chunks under one of three strategies (the eval matrix axis):
    fixed            — blind character windows, no context prefix
    section          — section-aware sentence chunks, no context prefix
    section_context  — section-aware chunks, embedded with a title/section prefix

    Section-aware chunks split on sentence boundaries, never span a section,
    and carry OVERLAP_CHARS of trailing-sentence overlap. The source_path
    metadata lets cross-paper answers verify quotes against the chunk's own
    parsed paper."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")
    if strategy == "fixed":
        return _finalize_chunks(paper, pid, year, [[u] for u in _fixed_windows(paper)], False)
    units = _units(paper)
    groups: list[list[Unit]] = []
    buf: list[Unit] = []
    buf_len = 0
    for page, section, sentence, content_type in units:
        # A content-type change breaks a chunk the same way a section change does:
        # a chunk that mixed prose with a figure description would have to be
        # labelled by the less certain half, and would embed as neither.
        section_changed = buf and (section != buf[0][1] or content_type != buf[0][3])
        if buf and (section_changed or buf_len + len(sentence) > TARGET_CHARS):
            groups.append(buf)
            if section_changed:
                buf, buf_len = [], 0
            else:  # overlap: carry trailing sentences into the next chunk
                tail: list[Unit] = []
                tail_len = 0
                for unit in reversed(buf):
                    if tail_len + len(unit[2]) > OVERLAP_CHARS:
                        break
                    tail.insert(0, unit)
                    tail_len += len(unit[2])
                buf, buf_len = tail, tail_len
        buf.append((page, section, sentence, content_type))
        buf_len += len(sentence)
    if buf:
        groups.append(buf)
    return _finalize_chunks(paper, pid, year, groups, prefix=(strategy == "section_context"))


def _finalize_chunks(
    paper: Paper,
    pid: str,
    year: int | None,
    groups: list[list[Unit]],
    prefix: bool,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, group in enumerate(groups):
        page, section = group[0][0], group[0][1]
        text = " ".join(u[2] for u in group)
        # Groups never mix content types (chunk_paper breaks on a change), and
        # fixed windows are all plain text.
        content_type = group[0][3]
        embed_text = (
            f"From '{paper.title}', section '{section or 'unknown'}': {text}" if prefix else text
        )
        chunks.append(
            Chunk(
                id=f"{pid}:{idx}",
                text=text,
                embed_text=embed_text,
                metadata={
                    "paper_id": pid,
                    "paper_title": paper.title,
                    "page": page,
                    "section": section,
                    "chunk_index": idx,
                    "year": year or 0,
                    "source_path": str(paper.path),
                    "content_type": content_type,
                },
            )
        )
    return chunks


_model = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """The single embedding entry point — swap providers here (spec §2)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL)
    return [[float(x) for x in v] for v in _model.encode(texts, normalize_embeddings=True)]


def _collection(db_path: Path, rebuild: bool = False):
    client = chromadb.PersistentClient(path=str(db_path))
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def _upsert_paper(collection, pdf: Path, embed_fn, strategy: str, figures: bool) -> tuple[str, int]:
    """Parse, chunk, embed and upsert one PDF. Returns (paper_id, chunk count)."""
    paper = parse_pdf(pdf)
    if figures:
        from personal_ra.vision import enrich_paper

        paper = enrich_paper(paper, equations=False, figures=True)
    pid = paper_id(pdf)
    chunks = chunk_paper(paper, pid, detect_year(paper), strategy=strategy)
    if not chunks:
        return pid, 0
    collection.upsert(
        ids=[c.id for c in chunks],
        embeddings=embed_fn([c.embed_text for c in chunks]),
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    return pid, len(chunks)


def is_indexed(pdf: Path, db_path: Path = DB_PATH) -> bool:
    """Whether this file's content is already in the store. Keyed on the content
    hash, so a paper re-downloaded under a different name is still a duplicate."""
    got = _collection(db_path).get(where={"paper_id": paper_id(pdf)}, limit=1)
    return bool(got["ids"])


def ingest_paper(
    pdf: Path,
    db_path: Path = DB_PATH,
    embed_fn=None,
    strategy: str = "section_context",
    figures: bool = False,
) -> dict:
    """Index exactly one PDF.

    `ingest` scans a directory, so adding today's arXiv paper re-embedded all
    4,807 chunks of the library. This does the one paper. Same deterministic IDs
    and upsert semantics, so the two remain interchangeable and idempotent.
    """
    collection = _collection(db_path)
    pid, n_chunks = _upsert_paper(collection, pdf, embed_fn or embed_texts, strategy, figures)
    return {
        "paper": pdf.name,
        "paper_id": pid,
        "chunks": n_chunks,
        "total_in_db": collection.count(),
    }


def ingest(
    papers_dir: Path = PAPERS_DIR,
    db_path: Path = DB_PATH,
    rebuild: bool = False,
    embed_fn=None,
    strategy: str = "section_context",
    figures: bool = False,
) -> dict:
    """Parse, chunk, embed, and upsert every PDF in papers_dir. Idempotent.

    figures=True describes each detected figure with Claude vision and indexes
    the descriptions. It costs money on the first run over a paper (cached after
    that), so it is off by default.
    """
    embed_fn = embed_fn or embed_texts
    collection = _collection(db_path, rebuild=rebuild)

    per_paper: dict[str, int] = {}
    for pdf in sorted(papers_dir.glob("*.pdf")):
        _pid, n_chunks = _upsert_paper(collection, pdf, embed_fn, strategy, figures)
        if n_chunks:
            per_paper[pdf.name] = n_chunks

    return {
        "papers": len(per_paper),
        "chunks": sum(per_paper.values()),
        "per_paper": per_paper,
        "total_in_db": collection.count(),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Ingest papers into the vector store.")
    ap.add_argument("--papers-dir", type=Path, default=PAPERS_DIR)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--rebuild", action="store_true", help="wipe the collection and reindex")
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="section_context")
    ap.add_argument(
        "--figures", action="store_true", help="describe and index figures (costs vision calls)"
    )
    args = ap.parse_args(argv)

    stats = ingest(
        args.papers_dir,
        args.db,
        rebuild=args.rebuild,
        strategy=args.strategy,
        figures=args.figures,
    )
    print(f"Ingested {stats['papers']} papers -> {stats['chunks']} chunks")
    for name, n in stats["per_paper"].items():
        print(f"  {name}: {n} chunks")
    print(f"Total chunks in DB: {stats['total_in_db']}")


if __name__ == "__main__":
    main()
