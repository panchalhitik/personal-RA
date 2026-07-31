"""MCP server exposing the paper library to Claude Code (stdio transport).

The tool *descriptions* are the load-bearing part of this file — they are what a
model reads to decide which tool to reach for. The distinction they encode:
search_library finds *which* papers say something and returns fragments;
read_paper returns one paper's complete argument. Fragments are for locating,
whole papers are for reasoning.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from personal_ra.cite import verify_quote as _verify_quote_impl
from personal_ra.eval import RESULTS_DIR, render_markdown_table
from personal_ra.library import DB_PATH, extract_sections
from personal_ra.parse import Paper, parse_pdf
from personal_ra.search import Library

SERVER_INSTRUCTIONS = """This server searches Hitik's personal library of ML/NLP research \
papers (PDFs he has read and ingested). Use it whenever a question concerns papers in that \
library rather than general knowledge, and cite paper title + page when you answer from it.

Start with list_papers or search_library. Never claim a paper says something without either \
retrieving it or checking the wording with verify_quote."""

_state: dict[str, Any] = {
    "db_path": DB_PATH,
    "results_dir": RESULTS_DIR,
    "library": None,
    "index": None,
    "papers": {},
}

server = MCPServer(
    name="personal-ra",
    version="2.0.0",
    instructions=SERVER_INSTRUCTIONS,
)


def configure(db_path: Path | None = None, results_dir: Path | None = None) -> None:
    """Point the server at a different index (used by tests). Clears caches."""
    if db_path is not None:
        _state["db_path"] = db_path
    if results_dir is not None:
        _state["results_dir"] = results_dir
    _state["library"] = None
    _state["index"] = None
    _state["papers"] = {}


def _library() -> Library:
    if _state["library"] is None:
        # stdio transport carries JSON-RPC on stdout: a stray print from the
        # embedding model would corrupt the protocol stream, so send it to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            _state["library"] = Library(db_path=_state["db_path"])
    return _state["library"]


def _index() -> dict[str, dict]:
    """paper_id -> {title, year, pages, chunks, sections, source_path}."""
    if _state["index"] is None:
        data = _library().collection.get(include=["metadatas"])
        papers: dict[str, dict] = {}
        for meta in data["metadatas"]:
            entry = papers.setdefault(
                meta["paper_id"],
                {
                    "paper_id": meta["paper_id"],
                    "title": meta["paper_title"],
                    "year": meta["year"] or None,
                    "source_path": meta["source_path"],
                    "pages": 0,
                    "chunks": 0,
                    "sections": [],
                },
            )
            entry["pages"] = max(entry["pages"], meta["page"])
            entry["chunks"] += 1
            if meta["section"] and meta["section"] not in entry["sections"]:
                entry["sections"].append(meta["section"])
        _state["index"] = papers
    return _state["index"]


def _require_paper(paper_id: str) -> dict:
    """Look up a paper, or raise a message that tells the model how to recover."""
    index = _index()
    if paper_id in index:
        return index[paper_id]
    known = ", ".join(sorted(index)[:5])
    raise ValueError(
        f"Unknown paper_id {paper_id!r}. Call list_papers to get valid ids "
        f"(there are {len(index)}; the first few are {known})."
    )


def _parsed(paper_id: str) -> Paper:
    if paper_id not in _state["papers"]:
        entry = _require_paper(paper_id)
        source = Path(entry["source_path"])
        if not source.exists():
            raise ValueError(
                f"The PDF for {entry['title']!r} is no longer at {source}. "
                f"It may have been moved or renamed since ingest; re-run ingest."
            )
        _state["papers"][paper_id] = parse_pdf(source)
    return _state["papers"][paper_id]


@server.tool(
    description=(
        "Search across ALL papers in the library and return the most relevant excerpts, "
        "each labelled with its paper title, page, and section.\n\n"
        "USE THIS WHEN: you don't yet know which paper holds the answer, the question spans "
        "several papers ('which of my papers use contrastive loss?'), or you want to locate "
        "a specific claim, number, or method by topic.\n\n"
        "DO NOT USE THIS WHEN: you already know the paper and need its full argument or a "
        "complete section — call read_paper instead. Excerpts are fragments of roughly 1000 "
        "characters, so an ablation, a proof, or a multi-step argument will be cut across "
        "several of them; repeatedly searching the same paper is a sign you should be "
        "reading it instead.\n\n"
        "Retrieval is hybrid (semantic embeddings + BM25 keyword matching fused by "
        "Reciprocal Rank Fusion), so both paraphrases and exact technical terms work. "
        "Narrow with paper_id (from list_papers) or a year range when the user names a "
        "paper or a time window. Raise k above 8 for broad survey questions; lower it for "
        "pinpoint lookups."
    ),
)
def search_library(
    query: str,
    k: int = 8,
    paper_id: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> dict:
    if paper_id:
        _require_paper(paper_id)
    hits = _library().search(query, k=k, paper_id=paper_id, year_min=year_min, year_max=year_max)
    return {
        "query": query,
        "n_results": len(hits),
        "results": [
            {
                "paper_id": h.metadata["paper_id"],
                "paper_title": h.metadata["paper_title"],
                "page": h.metadata["page"],
                "section": h.metadata["section"] or None,
                "year": h.metadata["year"] or None,
                "score": h.score,
                "text": h.text,
            }
            for h in hits
        ],
    }


@server.tool(
    description=(
        "Read one paper in full, or one of its sections, as clean text with page markers.\n\n"
        "USE THIS WHEN: you know which paper you need (from search_library or list_papers) "
        "and the question requires the whole argument rather than fragments — 'does the "
        "ablation support the main claim?', 'summarise this paper', 'what exactly is their "
        "method?'. A typical paper is 15-30k tokens and fits comfortably in context, so "
        "reading it whole is usually better than assembling it from search excerpts.\n\n"
        "Pass section to read just one part (e.g. '3. Method', 'Abstract') when the paper is "
        "long and you only need one region; call it with no section first if you don't know "
        "the section names, and the error will list them. Text comes from the same parse the "
        "search index was built from, so quotes taken here will verify."
    ),
)
def read_paper(paper_id: str, section: str | None = None) -> dict:
    entry = _require_paper(paper_id)
    paper = _parsed(paper_id)
    if section is None:
        return {
            "paper_id": paper_id,
            "title": entry["title"],
            "year": entry["year"],
            "pages": len(paper.pages),
            "approx_tokens": paper.n_tokens,
            "section": None,
            "text": paper.full_text,
        }

    sections = extract_sections(paper)
    match = next((label for label in sections if label.lower() == section.lower()), None)
    if match is None:
        available = ", ".join(repr(s) for s in list(sections)[:15])
        raise ValueError(
            f"No section {section!r} in {entry['title']!r}. Available sections include: "
            f"{available}. Omit the section argument to read the whole paper."
        )
    text = sections[match]
    return {
        "paper_id": paper_id,
        "title": entry["title"],
        "year": entry["year"],
        "pages": len(paper.pages),
        "approx_tokens": len(text) // 4,
        "section": match,
        "text": text,
    }


@server.tool(
    description=(
        "List every paper in the library with its id, title, year, page count, and chunk "
        "count.\n\n"
        "USE THIS WHEN: the user refers to a paper by name or topic and you need its "
        "paper_id for read_paper or a filtered search_library; the user asks what is in the "
        "library or how big it is; or you want to check whether a paper is present before "
        "claiming it is missing.\n\n"
        "This is cheap and takes no arguments — prefer it over guessing an id. It does not "
        "search paper contents; use search_library for that."
    ),
)
def list_papers() -> dict:
    papers = [
        {
            "paper_id": e["paper_id"],
            "title": e["title"],
            "year": e["year"],
            "pages": e["pages"],
            "chunks": e["chunks"],
        }
        for e in sorted(_index().values(), key=lambda e: e["title"].lower())
    ]
    return {"n_papers": len(papers), "papers": papers}


@server.tool(
    description=(
        "Check whether a quote really appears in a specific paper, and if so on which page.\n\n"
        "USE THIS WHEN: you are about to present text as a direct quotation from a paper, "
        "when the user asks whether a paper actually says something, or when you want a page "
        "number for a passage you already have.\n\n"
        "Matching ignores whitespace, curly vs straight quotes, dashes, ligatures, and case; "
        "it tries an exact match first, then a fuzzy match at a 95%% similarity threshold. "
        "verified=false means the wording does not occur in that paper — treat the quote as "
        "unreliable and do not present it as a quotation. match_type='fuzzy' means the "
        "wording is close but not exact, so quote the paper's own wording rather than yours."
    ),
)
def verify_quote(quote: str, paper_id: str) -> dict:
    entry = _require_paper(paper_id)
    citation = _verify_quote_impl(quote, _parsed(paper_id))
    return {
        "quote": citation.quote,
        "paper_id": paper_id,
        "paper_title": entry["title"],
        "verified": citation.verified,
        "match_type": citation.match_type,
        "page": citation.page,
        "char_offset": citation.char_offset,
    }


@server.resource(
    "library://index",
    name="Library index",
    description="Every paper in the library: id, title, year, pages, chunks.",
    mime_type="text/markdown",
)
def library_index() -> str:
    papers = sorted(_index().values(), key=lambda e: e["title"].lower())
    lines = [
        f"# Personal-RA library ({len(papers)} papers)",
        "",
        "| paper_id | year | pages | chunks | title |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| `{p['paper_id']}` | {p['year'] or '—'} | {p['pages']} | {p['chunks']} | {p['title']} |"
        for p in papers
    ]
    return "\n".join(lines)


@server.resource(
    "eval://latest",
    name="Latest retrieval metrics",
    description="Most recent retrieval evaluation: config matrix with recall@k and MRR.",
    mime_type="text/markdown",
)
def eval_latest() -> str:
    results_dir = Path(_state["results_dir"])
    files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
    if not files:
        return "No evaluation results yet. Run: python -m personal_ra.eval --matrix"
    payload = json.loads(files[-1].read_text(encoding="utf-8"))
    configs = payload.get("configs", [])
    header = (
        f"# Retrieval evaluation ({files[-1].stem})\n\n"
        f"Golden set: `{payload.get('golden_set')}` — k={payload.get('k')}\n\n"
    )
    return header + render_markdown_table(configs)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
