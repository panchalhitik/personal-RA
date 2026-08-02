"""The graph's shared state.

Plain TypedDict fields: each node returns a partial dict and LangGraph merges it
last-write-wins. No reducers, because nothing fans out in parallel yet.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Literal, TypedDict

from personal_ra.search import RetrievedChunk

Route = Literal["single_paper", "library", "web", "direct"]

MAX_REWRITES = 2
MIN_GRADED_CHUNKS = 2  # fewer surviving chunks than this triggers a rewrite


class State(TypedDict, total=False):
    # input
    question: str
    original_question: str
    paper_id: str | None  # set when the user has a paper open
    thread_id: str
    # Not in the spec's State listing, but §3.2 requires conversation history in the
    # classification input ("what about the second one?" is unroutable without it).
    # Shape matches ask.py's Message: {"role": "user"|"assistant", "content": str}.
    history: list[dict]

    # routing
    route: Route
    route_reason: str  # why — surfaced in the UI and traces

    # retrieval
    chunks: list[dict]  # search.py output shape, unchanged
    graded_chunks: list[dict]
    rejected_chunks: list[dict]  # keep them — needed for rewrite and eval
    rewrite_count: int  # hard cap MAX_REWRITES
    # Why the query was rewritten, one sentence — the same job route_reason does for
    # routing, and needed by the UI and traces for the same reason.
    rewrite_reason: str
    web_results: list[dict]
    # The queries retrieval actually ran. A comparison question becomes several,
    # and a trace showing only the original question cannot explain the chunks.
    retrieval_queries: list[str]
    # Checkpoint 3.3: reranking is opt-in, not default — it adds ~1s at p50 and wins
    # precision@1 without winning recall@5. Per-query rather than per-graph so a UI
    # toggle or an MCP argument can turn it on without rebuilding the graph.
    rerank: bool

    # approval
    awaiting_approval: bool
    approved: bool | None

    # output
    answer: str
    citations: list[dict]  # cite.py Citation, serialized
    unverified: list[dict]
    grounding: dict  # verdict, unsupported claims
    usage: dict  # tokens, cost, per-node latency


def chunk_to_dict(chunk) -> dict:
    """search.py's RetrievedChunk -> the plain dict the state carries.

    State holds dicts rather than dataclasses so a checkpointer round-trip is
    lossless — `asdict` keeps the shape identical to what search.py returned.
    """
    return asdict(chunk)


def chunk_from_dict(data: dict) -> RetrievedChunk:
    """The inverse, tolerating keys the graph added along the way.

    Graded chunks carry `grade_reason`, which is not a RetrievedChunk field, so a
    bare `RetrievedChunk(**data)` raises. Filtering to the dataclass's own fields
    keeps the graph free to annotate chunks without breaking round-trips.
    """
    allowed = {f.name for f in fields(RetrievedChunk)}
    return RetrievedChunk(**{k: v for k, v in data.items() if k in allowed})


def initial_state(
    question: str,
    paper_id: str | None = None,
    thread_id: str = "default",
    route: Route | None = None,
    history: list[dict] | None = None,
    rerank: bool = False,
) -> State:
    """Every field populated, so no node has to guard against a missing key.

    `route` is a test/debug override: when set, the router honours it instead of
    classifying. Step 3.2 replaces the classifier, not this escape hatch.
    """
    state: State = {
        "question": question,
        "original_question": question,
        "paper_id": paper_id,
        "thread_id": thread_id,
        "history": history or [],
        "route": route,  # type: ignore[typeddict-item]
        "route_reason": "",
        "chunks": [],
        "graded_chunks": [],
        "rejected_chunks": [],
        "rewrite_count": 0,
        "rewrite_reason": "",
        "web_results": [],
        "retrieval_queries": [],
        "rerank": rerank,
        "awaiting_approval": False,
        "approved": None,
        "answer": "",
        "citations": [],
        "unverified": [],
        "grounding": {},
        "usage": {},
    }
    return state
