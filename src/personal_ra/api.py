"""HTTP wrapper around the v3 graph.

    POST /ask                     SSE: a node event per transition, then the answer
    POST /ask/{thread_id}/approve resume a graph halted at the approval gate
    GET  /ask/{thread_id}         status and the full node trace
    GET  /papers                  library inventory
    POST /ingest                  add a paper (v4's arXiv automation posts here)
    GET  /health

`/ask` streams **node transitions**, not just the answer. Watching
`route -> retrieve -> rerank -> grade -> rewrite -> retrieve -> generate ->
grounding` arrive live is what makes the architecture explain itself, and it is the
one thing a plain request/response API cannot show.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from personal_ra.graph.build import build_graph, pending_approval, sqlite_checkpointer
from personal_ra.graph.state import initial_state


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    paper_id: str | None = None
    thread_id: str | None = None
    rerank: bool = False
    history: list[dict] = Field(default_factory=list)


class ApproveRequest(BaseModel):
    # Defaults to False so a malformed body cannot spend money on a web search.
    # `web.interpret_decision` applies the same rule to the value itself.
    approved: bool = False


class IngestRequest(BaseModel):
    path: str
    dry_run: bool = False


def sse(event: str, payload: dict) -> str:
    """One Server-Sent Event. The blank line terminator is what flushes it."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _summary(values: dict) -> dict:
    """The bits of State an HTTP client actually needs, without the bulk."""
    grounding = values.get("grounding") or {}
    return {
        "question": values.get("original_question", ""),
        "route": values.get("route"),
        "route_reason": values.get("route_reason", ""),
        "answer": values.get("answer", ""),
        "citations": values.get("citations") or [],
        "unverified": values.get("unverified") or [],
        "grounding": {
            "verdict": grounding.get("verdict"),
            "unsupported": grounding.get("unsupported") or [],
        },
        "rewrite_count": values.get("rewrite_count", 0),
        "rewrite_reason": values.get("rewrite_reason", ""),
        "retrieval_queries": values.get("retrieval_queries") or [],
        "usage": values.get("usage") or {},
    }


def _inventory(library) -> list[dict]:
    """paper_id -> title, year, pages, chunks, from the chunk metadata."""
    data = library.collection.get(include=["metadatas"])
    papers: dict[str, dict] = {}
    for meta in data["metadatas"]:
        entry = papers.setdefault(
            meta["paper_id"],
            {
                "paper_id": meta["paper_id"],
                "title": meta["paper_title"],
                "year": meta.get("year") or None,
                "pages": 0,
                "chunks": 0,
            },
        )
        entry["pages"] = max(entry["pages"], meta.get("page") or 0)
        entry["chunks"] += 1
    return sorted(papers.values(), key=lambda p: p["title"])


def create_app(graph=None, library=None, ingest_fn=None) -> FastAPI:
    """Everything injectable, so the tests never touch Chroma or the network."""
    app = FastAPI(title="Personal-RA", version="3.0")

    def _graph():
        if graph is not None:
            return graph
        # Built once, lazily: constructing it opens the Chroma collection, which
        # should not happen just because someone imported this module.
        if not hasattr(app.state, "graph"):
            app.state.graph = build_graph(checkpointer=sqlite_checkpointer())
        return app.state.graph

    def _library():
        if library is not None:
            return library
        from personal_ra.graph.retrieve import default_library

        return default_library()

    def _require_thread(thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        values = _graph().get_state(config).values
        if not values:
            # 404, not a stack trace: an unknown thread is a client mistake, and
            # LangGraph answers for one with an empty snapshot rather than raising.
            raise HTTPException(status_code=404, detail=f"No thread {thread_id!r}")
        return values

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "personal-ra", "version": app.version}

    @app.get("/papers")
    def papers() -> dict:
        inventory = _inventory(_library())
        return {"n_papers": len(inventory), "papers": inventory}

    @app.post("/ask")
    def ask(request: AskRequest):
        from fastapi.responses import StreamingResponse

        thread_id = request.thread_id or f"api-{abs(hash(request.question)) % 10**10}"
        config = {"configurable": {"thread_id": thread_id}}
        state = initial_state(
            request.question,
            request.paper_id,
            thread_id,
            history=request.history,
            rerank=request.rerank,
        )

        def stream() -> Iterator[str]:
            graph_ = _graph()
            yield sse("start", {"thread_id": thread_id, "question": request.question})
            for update in graph_.stream(state, config, stream_mode="updates"):
                for node, delta in update.items():
                    yield sse("node", {"node": node, **_node_event(node, delta)})

            waiting = pending_approval(graph_, config)
            if waiting is not None:
                # Halted at the approval gate. The stream ends here on purpose; the
                # client resumes through /approve rather than holding the connection
                # open, which could be minutes.
                yield sse("approval_required", {"thread_id": thread_id, "request": waiting})
                return
            yield sse(
                "answer", {"thread_id": thread_id, **_summary(graph_.get_state(config).values)}
            )

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/ask/{thread_id}/approve")
    def approve(thread_id: str, request: ApproveRequest) -> dict:
        graph_ = _graph()
        config = {"configurable": {"thread_id": thread_id}}
        _require_thread(thread_id)
        if pending_approval(graph_, config) is None:
            raise HTTPException(
                status_code=409, detail=f"Thread {thread_id!r} is not waiting for approval"
            )
        graph_.invoke(Command(resume=request.approved), config)
        return {"thread_id": thread_id, **_summary(graph_.get_state(config).values)}

    @app.get("/ask/{thread_id}")
    def status(thread_id: str) -> dict:
        graph_ = _graph()
        config = {"configurable": {"thread_id": thread_id}}
        values = _require_thread(thread_id)
        waiting = pending_approval(graph_, config)
        return {
            "thread_id": thread_id,
            "status": "awaiting_approval" if waiting is not None else "complete",
            "approval_request": waiting,
            "trace": _trace(values),
            **_summary(values),
        }

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> dict:
        path = Path(request.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"No file at {request.path!r}")
        if path.suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="Only .pdf files can be ingested")
        if request.dry_run:
            # v4.2's n8n workflow needs to exercise the whole path without writing,
            # so a dry run reports what it would do and touches nothing.
            return {"path": str(path), "dry_run": True, "ingested": False}

        result = (ingest_fn or _default_ingest)(path)
        return {"path": str(path), "dry_run": False, "ingested": True, **result}

    return app


def _default_ingest(path: Path) -> dict:
    """Put the PDF in papers/ and reindex.

    `library.ingest` scans a directory rather than accepting one file, and it is
    idempotent, so the honest implementation is to copy the file in and re-run it.
    That rescans the whole library — fine for the occasional addition, and the
    obvious thing for v4.2 to make incremental once it is posting daily.
    """
    import shutil

    from personal_ra.library import PAPERS_DIR, ingest

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    destination = PAPERS_DIR / path.name
    already_there = destination.exists() and destination.resolve() == path.resolve()
    if not already_there:
        shutil.copy2(path, destination)
    return {"stored_at": str(destination), "already_present": already_there, "index": ingest()}


def _node_event(node: str, delta) -> dict:
    """A node transition, summarised. Deltas carry whole chunk bodies; a progress
    event carries counts."""
    if not isinstance(delta, dict):
        return {}
    event: dict = {}
    for key in ("route", "route_reason", "rewrite_count", "rewrite_reason", "approved"):
        if key in delta:
            event[key] = delta[key]
    for key in ("chunks", "graded_chunks", "rejected_chunks", "web_results", "citations"):
        if key in delta and delta[key] is not None:
            event[f"n_{key}"] = len(delta[key])
    if "grounding" in delta and isinstance(delta["grounding"], dict):
        event["verdict"] = delta["grounding"].get("verdict")
    return event


def _trace(values: dict) -> list[dict]:
    """Per-node cost and tokens, read back from the accumulated usage dict."""
    usage = values.get("usage") or {}
    return [
        {
            "node": node,
            "input_tokens": entry.get("input_tokens", 0),
            "output_tokens": entry.get("output_tokens", 0),
            "cost_usd": entry.get("cost_usd", 0.0),
        }
        for node, entry in usage.items()
        if isinstance(entry, dict)
    ]


app = create_app()
