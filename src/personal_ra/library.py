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

from personal_ra.parse import Paper, parse_pdf

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


@dataclass
class Chunk:
    id: str  # "{paper_id}:{chunk_index}" — deterministic, stable across runs
    text: str
    embed_text: str  # context-prefixed text, what actually gets embedded
    metadata: dict


def paper_id(path: Path) -> str:
    """Stable id from file content — renaming a file keeps its identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def detect_year(paper: Paper) -> int | None:
    """Best-effort publication year: latest plausible year on page 1."""
    if not paper.pages:
        return None
    years = [int(y) for y in _YEAR_RE.findall(paper.pages[0].text)]
    return max(years) if years else None


def _is_section_header(line: str) -> bool:
    line = line.strip()
    return bool(line) and len(line) <= 80 and any(p.match(line) for p in _SECTION_PATTERNS)


def _units(paper: Paper) -> list[tuple[int, str, str]]:
    """(page, section, sentence) triples in reading order."""
    units: list[tuple[int, str, str]] = []
    section = ""
    for page in paper.pages:
        for line in page.text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _is_section_header(stripped):
                section = stripped
                continue
            for sentence in _SENT_SPLIT.split(stripped):
                if sentence.strip():
                    units.append((page.number, section, sentence.strip()))
    return units


def chunk_paper(paper: Paper, pid: str, year: int | None) -> list[Chunk]:
    """Section-aware chunks: sentence-boundary splits, no section spanning,
    ~TARGET_CHARS with OVERLAP_CHARS of trailing-sentence overlap. The
    source_path metadata lets cross-paper answers verify quotes against the
    chunk's own parsed paper."""
    units = _units(paper)
    chunks: list[Chunk] = []
    buf: list[tuple[int, str, str]] = []
    buf_len = 0

    def flush() -> None:
        if not buf:
            return
        page, section = buf[0][0], buf[0][1]
        text = " ".join(u[2] for u in buf)
        idx = len(chunks)
        chunks.append(
            Chunk(
                id=f"{pid}:{idx}",
                text=text,
                embed_text=f"From '{paper.title}', section '{section or 'unknown'}': {text}",
                metadata={
                    "paper_id": pid,
                    "paper_title": paper.title,
                    "page": page,
                    "section": section,
                    "chunk_index": idx,
                    "year": year or 0,
                    "source_path": str(paper.path),
                },
            )
        )

    for page, section, sentence in units:
        section_changed = buf and section != buf[0][1]
        if buf and (section_changed or buf_len + len(sentence) > TARGET_CHARS):
            flush()
            if section_changed:
                buf, buf_len = [], 0
            else:  # overlap: carry trailing sentences into the next chunk
                tail: list[tuple[int, str, str]] = []
                tail_len = 0
                for unit in reversed(buf):
                    if tail_len + len(unit[2]) > OVERLAP_CHARS:
                        break
                    tail.insert(0, unit)
                    tail_len += len(unit[2])
                buf, buf_len = tail, tail_len
        buf.append((page, section, sentence))
        buf_len += len(sentence)
    flush()
    return chunks


_model = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """The single embedding entry point — swap providers here (spec §2)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL)
    return [[float(x) for x in v] for v in _model.encode(texts, normalize_embeddings=True)]


def ingest(
    papers_dir: Path = PAPERS_DIR,
    db_path: Path = DB_PATH,
    rebuild: bool = False,
    embed_fn=None,
) -> dict:
    """Parse, chunk, embed, and upsert every PDF in papers_dir. Idempotent."""
    embed_fn = embed_fn or embed_texts
    client = chromadb.PersistentClient(path=str(db_path))
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    per_paper: dict[str, int] = {}
    for pdf in sorted(papers_dir.glob("*.pdf")):
        paper = parse_pdf(pdf)
        pid = paper_id(pdf)
        chunks = chunk_paper(paper, pid, detect_year(paper))
        if not chunks:
            continue
        collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embed_fn([c.embed_text for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        per_paper[pdf.name] = len(chunks)

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
    args = ap.parse_args(argv)

    stats = ingest(args.papers_dir, args.db, rebuild=args.rebuild)
    print(f"Ingested {stats['papers']} papers -> {stats['chunks']} chunks")
    for name, n in stats["per_paper"].items():
        print(f"  {name}: {n} chunks")
    print(f"Total chunks in DB: {stats['total_in_db']}")


if __name__ == "__main__":
    main()
