"""Retrieval for the graph, and resolving a paper_id back to a parsed Paper.

Both are thin wrappers over v1/v0 code. The only real work here is caching: a
`Library` loads every document and rebuilds BM25 on construction, and parsing a
PDF is seconds, so neither may happen per query.
"""

from __future__ import annotations

from functools import lru_cache

from personal_ra.graph.rerank import RETRIEVE_DEPTH, TOP_K
from personal_ra.parse import Paper
from personal_ra.search import Library, RetrievedChunk, parsed_paper


@lru_cache(maxsize=1)
def default_library() -> Library:
    """Cached because Library.__init__ reads the whole collection and builds BM25."""
    return Library()


def retrieve_chunks(
    question: str,
    library: Library | None = None,
    paper_id: str | None = None,
    rerank: bool = False,
) -> list[RetrievedChunk]:
    """Hybrid retrieval, deeper when reranking will follow.

    Depth is the only thing the rerank flag changes here — the reranking itself is
    the next node's job, so this stays a pure retrieval step.
    """
    library = library or default_library()
    k = RETRIEVE_DEPTH if rerank else TOP_K
    return library.search(question, k=k, mode="hybrid", paper_id=paper_id)


def resolve_paper(paper_id: str, library: Library | None = None) -> Paper | None:
    """paper_id -> parsed Paper, via the source_path stored on its chunks.

    Returns None rather than raising for an unknown id: the router can set a
    paper_id from a stale UI selection, and that should degrade to a library
    answer, not crash the graph.
    """
    library = library or default_library()
    found = library.collection.get(
        where={"paper_id": {"$eq": paper_id}}, limit=1, include=["metadatas"]
    )
    metadatas = found.get("metadatas") or []
    if not metadatas:
        return None
    return parsed_paper(metadatas[0]["source_path"])
