"""Cross-encoder reranking — the fix the v2 README named for `max_per_paper`.

Bi-encoder retrieval (what `search.py` does) embeds the query and the chunk
separately, so it never sees them together. A cross-encoder reads the pair in one
forward pass and scores the match directly. That is far more accurate and far too
slow to run over 4,807 chunks — so it reranks the top 30 that retrieval already
found, and keeps the best 8.

`max_per_paper` survives as an *optional* diversity cap applied after reranking,
not instead of it: reranking answers "is this chunk relevant?", the cap answers
"am I showing the user too much of one paper?". Those are different questions.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from personal_ra.search import Library, RetrievedChunk

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVE_DEPTH = 30  # retrieve deeper than needed so reranking has something to fix
TOP_K = 8


@lru_cache(maxsize=1)
def load_model(name: str = RERANK_MODEL):
    """Cached: loading the model takes seconds, so it must not happen per query.

    Imported lazily — `sentence_transformers.CrossEncoder` pulls in torch, and
    nothing that only does retrieval should pay that at import time.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)


def cap_per_paper(
    chunks: list[RetrievedChunk], max_per_paper: int | None, top_k: int
) -> list[RetrievedChunk]:
    """Keep at most `max_per_paper` chunks from any one paper, preserving rank order.

    Same rule `mcp_server.search_library` applies, but here it runs on a
    relevance-ordered list rather than an RRF-ordered one.
    """
    if max_per_paper is None:
        return chunks[:top_k]
    seen: dict[str, int] = {}
    kept: list[RetrievedChunk] = []
    for chunk in chunks:
        pid = chunk.metadata["paper_id"]
        if seen.get(pid, 0) >= max_per_paper:
            continue
        seen[pid] = seen.get(pid, 0) + 1
        kept.append(chunk)
        if len(kept) >= top_k:
            break
    return kept


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = TOP_K,
    max_per_paper: int | None = None,
    model=None,
) -> list[RetrievedChunk]:
    """Reorder `chunks` by cross-encoder relevance, then cap and truncate.

    The returned chunks carry the cross-encoder score in `.score`, replacing the
    RRF score they arrived with — `.dense_rank` and `.bm25_rank` are left intact so
    the pre-rerank provenance is still readable in a trace.
    """
    if not chunks:
        return []
    model = model or load_model()
    scores = model.predict([(query, chunk.text) for chunk in chunks])
    scored = [replace(c, score=round(float(s), 5)) for c, s in zip(chunks, scores)]
    # sorted() is stable, so equal scores keep their retrieval order.
    scored.sort(key=lambda c: c.score, reverse=True)
    return cap_per_paper(scored, max_per_paper, top_k)


def retrieve_and_rerank(
    library: Library,
    query: str,
    top_k: int = TOP_K,
    depth: int = RETRIEVE_DEPTH,
    max_per_paper: int | None = None,
    model=None,
    **filters,
) -> list[RetrievedChunk]:
    """Hybrid retrieval at `depth`, then rerank down to `top_k`.

    `depth` is raised to at least 3x top_k: reranking can only reorder what
    retrieval handed it, so a shallow first stage caps the achievable gain.
    """
    chunks = library.search(query, k=max(depth, top_k * 3), mode="hybrid", **filters)
    return rerank(query, chunks, top_k=top_k, max_per_paper=max_per_paper, model=model)
