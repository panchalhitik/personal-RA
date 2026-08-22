"""Cross-paper retrieval and grounded answers over the whole library.

Hybrid retrieval: dense (Chroma cosine) and BM25 rankings fused with
Reciprocal Rank Fusion. Answers keep v0's verbatim-quote discipline — every
quote is verified against the parsed source of the paper it came from.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from personal_ra.ask import MODEL, _usage_dict
from personal_ra.cite import Citation, verify_quote
from personal_ra.library import COLLECTION, DB_PATH, embed_texts
from personal_ra.parse import Paper, parse_pdf, use_utf8_stdout

RRF_K = 60
LIBRARY_REFUSAL = "That isn't covered in my library."

LIBRARY_PROMPT = f"""You are a careful research assistant. You answer questions using only \
the paper excerpts provided in <paper> blocks — never general knowledge.

Rules:
- Support every substantive claim with a short verbatim quote from an excerpt, wrapped in \
<quote></quote> tags. Copy character-for-character; never paraphrase inside the tags.
- Name the paper(s) you are drawing from in your prose, and say which papers cover what \
when the question spans several.
- Each excerpt is labeled with its page number. Excerpts are fragments — if they don't \
contain enough to answer, say so rather than guessing.
- If the excerpts don't cover the question at all, reply exactly: {LIBRARY_REFUSAL}"""


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    score: float  # fused RRF score
    dense_rank: int | None = None
    bm25_rank: int | None = None


@dataclass
class LibraryCitation:
    paper_title: str
    citation: Citation


@dataclass
class LibraryAnswer:
    text: str
    citations: list[LibraryCitation]
    unverified: list[Citation]
    chunks: list[RetrievedChunk]  # what retrieval returned, scores included
    usage: dict = field(default_factory=dict)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1/(k + rank)."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return dict(scores)


def _build_where(paper_id: str | None, year_min: int | None, year_max: int | None) -> dict | None:
    clauses: list[dict] = []
    if paper_id:
        clauses.append({"paper_id": {"$eq": paper_id}})
    if year_min is not None:
        clauses.append({"year": {"$gte": year_min}})
    if year_max is not None:
        clauses.append({"year": {"$lte": year_max}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class Library:
    """Read-side handle on the ingested corpus. BM25 is built once per instance;
    create a fresh Library after re-ingesting."""

    def __init__(self, db_path: Path = DB_PATH, embed_fn=None) -> None:
        self.embed_fn = embed_fn or embed_texts
        client = chromadb.PersistentClient(path=str(db_path))
        self.collection = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        data = self.collection.get(include=["documents", "metadatas"])
        self._ids: list[str] = data["ids"]
        self._docs: dict[str, str] = dict(zip(data["ids"], data["documents"]))
        self._metas: dict[str, dict] = dict(zip(data["ids"], data["metadatas"]))
        self._bm25 = BM25Okapi([_tokenize(self._docs[i]) for i in self._ids]) if self._ids else None

    def _matches_filters(
        self, meta: dict, paper_id: str | None, year_min: int | None, year_max: int | None
    ) -> bool:
        if paper_id and meta["paper_id"] != paper_id:
            return False
        if year_min is not None and meta["year"] < year_min:
            return False
        if year_max is not None and meta["year"] > year_max:
            return False
        return True

    def _dense_ranking(self, query: str, n: int, where: dict | None) -> list[str]:
        result = self.collection.query(
            query_embeddings=self.embed_fn([query]),
            n_results=min(n, len(self._ids)),
            where=where,
        )
        return result["ids"][0]

    def _bm25_ranking(
        self,
        query: str,
        n: int,
        paper_id: str | None,
        year_min: int | None,
        year_max: int | None,
    ) -> list[str]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._ids, scores), key=lambda p: p[1], reverse=True)
        out = []
        for cid, score in ranked:
            if score <= 0:
                break
            if self._matches_filters(self._metas[cid], paper_id, year_min, year_max):
                out.append(cid)
            if len(out) >= n:
                break
        return out

    def search(
        self,
        query: str,
        k: int = 8,
        paper_id: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        mode: str = "hybrid",  # "dense" | "bm25" | "hybrid" — the eval matrix axis
    ) -> list[RetrievedChunk]:
        if not self._ids:
            return []
        n_fetch = max(k * 3, 20)
        rankings: list[list[str]] = []
        dense_ranks: dict[str, int] = {}
        bm25_ranks: dict[str, int] = {}
        if mode in ("dense", "hybrid"):
            dense = self._dense_ranking(query, n_fetch, _build_where(paper_id, year_min, year_max))
            dense_ranks = {cid: r for r, cid in enumerate(dense, start=1)}
            rankings.append(dense)
        if mode in ("bm25", "hybrid"):
            bm25 = self._bm25_ranking(query, n_fetch, paper_id, year_min, year_max)
            bm25_ranks = {cid: r for r, cid in enumerate(bm25, start=1)}
            rankings.append(bm25)

        fused = rrf_fuse(rankings)
        top = sorted(fused.items(), key=lambda p: p[1], reverse=True)[:k]
        return [
            RetrievedChunk(
                id=cid,
                text=self._docs[cid],
                metadata=self._metas[cid],
                score=round(score, 5),
                dense_rank=dense_ranks.get(cid),
                bm25_rank=bm25_ranks.get(cid),
            )
            for cid, score in top
        ]


_paper_cache: dict[str, Paper] = {}


def _parsed(source_path: str) -> Paper:
    if source_path not in _paper_cache:
        _paper_cache[source_path] = parse_pdf(source_path)
    return _paper_cache[source_path]


def _short_title(title: str, max_len: int = 40) -> str:
    return title if len(title) <= max_len else title[: max_len - 1].rstrip() + "…"


def _postprocess_library(
    raw: str, chunks: list[RetrievedChunk]
) -> tuple[str, list[LibraryCitation], list[Citation]]:
    """Verify each quote against the parsed source of every retrieved paper;
    the paper it matches in is the paper it gets attributed to."""
    papers: dict[str, str] = {}  # source_path -> title, insertion-ordered
    for ch in chunks:
        papers.setdefault(ch.metadata["source_path"], ch.metadata["paper_title"])

    citations: list[LibraryCitation] = []
    unverified: list[Citation] = []

    def replace(match: re.Match[str]) -> str:
        quote = match.group(1).strip()
        for path, title in papers.items():
            citation = verify_quote(quote, _parsed(path))
            if citation.verified:
                citations.append(LibraryCitation(paper_title=title, citation=citation))
                return f'"{quote}" [{_short_title(title)}, p. {citation.page}]'
        unverified.append(
            Citation(quote=quote, page=None, char_offset=None, verified=False, match_type="failed")
        )
        return "[unverified]"

    text = re.sub(r"<quote>(.*?)</quote>", replace, raw, flags=re.DOTALL).strip()
    return text, citations, unverified


def parsed_paper(source_path: str) -> Paper:
    """Public handle on the module's parse cache.

    Added for v3's single_paper node so it shares this cache instead of parsing the
    same PDF a second time. `answer_across_library` is untouched.
    """
    return _parsed(source_path)


def answer_from_chunks(
    question: str,
    chunks: list[RetrievedChunk],
    client: anthropic.Anthropic | None = None,
    system: str = LIBRARY_PROMPT,
    extra_context: str = "",
) -> LibraryAnswer:
    """The generation half of `answer_across_library`, over chunks retrieved elsewhere.

    v3's graph grades and rewrites between retrieval and generation, so it needs
    these two halves separable. Added as a new function rather than by refactoring
    `answer_across_library`, so that function's behaviour is bit-for-bit unchanged.

    `extra_context` carries already-formatted web results, which are labelled
    <web_result> rather than <paper> and so can never pick up a page citation.
    """
    if not chunks and not extra_context:
        return LibraryAnswer(text=LIBRARY_REFUSAL, citations=[], unverified=[], chunks=[], usage={})

    by_paper: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for ch in chunks:
        by_paper[ch.metadata["paper_title"]].append(ch)
    parts = []
    for title, paper_chunks in by_paper.items():
        excerpts = "\n\n".join(f"[page {ch.metadata['page']}] {ch.text}" for ch in paper_chunks)
        parts.append(f'<paper title="{title}">\n{excerpts}\n</paper>')
    if extra_context:
        parts.append(extra_context)
    context = "\n\n".join(parts)

    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
    )
    raw = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    text, citations, unverified = _postprocess_library(raw, chunks)
    return LibraryAnswer(
        text=text,
        citations=citations,
        unverified=unverified,
        chunks=chunks,
        usage=_usage_dict(response.usage),
    )


def answer_across_library(
    question: str,
    k: int = 8,
    library: Library | None = None,
    client: anthropic.Anthropic | None = None,
    paper_id: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> LibraryAnswer:
    library = library or Library()
    chunks = library.search(question, k=k, paper_id=paper_id, year_min=year_min, year_max=year_max)
    if not chunks:
        return LibraryAnswer(text=LIBRARY_REFUSAL, citations=[], unverified=[], chunks=[], usage={})

    by_paper: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for ch in chunks:
        by_paper[ch.metadata["paper_title"]].append(ch)
    context_parts = []
    for title, paper_chunks in by_paper.items():
        excerpts = "\n\n".join(f"[page {ch.metadata['page']}] {ch.text}" for ch in paper_chunks)
        context_parts.append(f'<paper title="{title}">\n{excerpts}\n</paper>')
    context = "\n\n".join(context_parts)

    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=0,
        system=LIBRARY_PROMPT,
        messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
    )
    raw = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    text, citations, unverified = _postprocess_library(raw, chunks)
    return LibraryAnswer(
        text=text,
        citations=citations,
        unverified=unverified,
        chunks=chunks,
        usage=_usage_dict(response.usage),
    )


def main(argv: list[str] | None = None) -> None:
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description="Search or ask across the whole library.")
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--paper-id", default=None)
    ap.add_argument("--year-min", type=int, default=None)
    ap.add_argument("--year-max", type=int, default=None)
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ap.add_argument(
        "--retrieve-only", action="store_true", help="show retrieved chunks, no API call"
    )
    args = ap.parse_args(argv)

    load_dotenv()
    library = Library()
    if args.retrieve_only:
        chunks = library.search(
            args.question,
            k=args.k,
            paper_id=args.paper_id,
            year_min=args.year_min,
            year_max=args.year_max,
            mode=args.mode,
        )
        for ch in chunks:
            m = ch.metadata
            print(
                f"[{ch.score:.5f} dense={ch.dense_rank} bm25={ch.bm25_rank}] "
                f"{_short_title(m['paper_title'])} p.{m['page']} ({m['section'] or 'no section'})"
            )
            print(f"    {ch.text[:200]}...\n")
        return

    answer = answer_across_library(
        args.question,
        k=args.k,
        library=library,
        paper_id=args.paper_id,
        year_min=args.year_min,
        year_max=args.year_max,
    )
    print(answer.text)
    if answer.citations:
        print("\nCitations:")
        for lc in answer.citations:
            c = lc.citation
            print(f'  [{lc.paper_title}, p. {c.page}] ({c.match_type}) "{c.quote}"')
    if answer.unverified:
        print("\nWARNING - unverified quotes:")
        for c in answer.unverified:
            print(f'  "{c.quote}"')
    if answer.usage:
        u = answer.usage
        print(f"\nUsage: in={u['input_tokens']} out={u['output_tokens']} cost=${u['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
