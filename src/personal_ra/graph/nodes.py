"""Every node in the router graph.

Real so far: route (3.2), rerank (3.3), grade (3.4). The rest are still the
pass-through stubs from 3.1, each naming the step that replaces it. Stubs invent
no data — `retrieve` returns nothing, which is why the library route still
exercises the rewrite loop to its cap end to end.

Node bodies stay thin: anything with a prompt, a schema, or a model behind it
lives in its own module (router.py, rerank.py, grade.py) and is called from here.
"""

from __future__ import annotations

import anthropic

from personal_ra.graph.grade import grade_chunks
from personal_ra.graph.grounding import (
    MAX_ATTEMPTS,
    UNGROUNDED,
    build_context,
    check_grounding,
    refusal_for_route,
)
from personal_ra.graph.rerank import TOP_K, rerank
from personal_ra.graph.rewrite import rewrite_query
from personal_ra.graph.router import classify_route
from personal_ra.graph.state import (
    MAX_REWRITES,
    MIN_GRADED_CHUNKS,
    State,
    chunk_to_dict,
)
from personal_ra.search import RetrievedChunk


def route_node(state: State, client: anthropic.Anthropic | None = None) -> dict:
    """Classify the question into one of four routes (Haiku, forced tool use)."""
    route, reason, usage = classify_route(state, client=client)
    return {
        "route": route,
        "route_reason": reason,
        "usage": {**state.get("usage", {}), "route": usage},
    }


def single_paper_node(state: State) -> dict:
    """Whole paper in a cached system prompt via ask.py. Filled in after 3.2."""
    return {"answer": "", "citations": [], "unverified": []}


def retrieve_node(state: State) -> dict:
    """Hybrid retrieval via search.py, k=30 so 3.3 has depth to rerank. Filled in after 3.2."""
    return {"chunks": []}


def rerank_node(state: State, model=None) -> dict:
    """Cross-encoder rerank, keep top TOP_K — opt-in per Checkpoint 3.3.

    Off by default: it wins precision@1 but not recall@5, and costs ~1s at p50.
    """
    chunks = state["chunks"]
    if not state.get("rerank") or not chunks:
        return {"chunks": chunks}
    reranked = rerank(
        state["original_question"],
        [RetrievedChunk(**c) for c in chunks],
        top_k=TOP_K,
        model=model,
    )
    return {"chunks": [chunk_to_dict(c) for c in reranked]}


def grade_node(state: State, client=None) -> dict:
    """Binary relevance per chunk, graded concurrently against the ORIGINAL question.

    Grading the rewritten question would judge relevance against a retrieval device
    rather than against what the user asked.
    """
    kept, rejected, usage = grade_chunks(state["original_question"], state["chunks"], client=client)
    return {
        "graded_chunks": kept,
        "rejected_chunks": rejected,
        "usage": {**state.get("usage", {}), "grade": usage},
    }


def rewrite_node(state: State, client=None) -> dict:
    """Rewrite the query toward more specific vocabulary, using the rejected chunks.

    `original_question` is never touched — the rewrite is a retrieval device, and
    both the grader and the UI need the question the user actually asked. The
    counter increment is what makes the loop terminate; `after_grade` enforces the
    cap so this node can never be entered a third time.
    """
    new_query, rationale, usage = rewrite_query(
        state["original_question"],
        state["question"],
        state.get("rejected_chunks", []),
        client=client,
    )
    return {
        "question": new_query,
        "rewrite_reason": rationale,
        "rewrite_count": state["rewrite_count"] + 1,
        "usage": {**state.get("usage", {}), f"rewrite_{state['rewrite_count'] + 1}": usage},
    }


def approve_node(state: State) -> dict:
    """LangGraph interrupt: halt and wait for approval before any web search. Step 3.7."""
    return {"awaiting_approval": False, "approved": True}


def web_search_node(state: State) -> dict:
    """Tavily, max 5 results, marked external and never given page numbers. Step 3.7."""
    return {"web_results": []}


def generate_node(state: State) -> dict:
    """Grounded answer over graded chunks (and web results). Filled in after 3.2."""
    return {"answer": "", "citations": [], "unverified": []}


def grounding_node(state: State, client=None) -> dict:
    """Audit the answer: cite.py's quote verdict, then a claim check on the prose.

    On a second consecutive `ungrounded` the answer is replaced by the route's
    refusal string — having tried once with a stricter prompt, returning nothing is
    better than returning something that rests on nothing.
    """
    attempt = state.get("grounding", {}).get("attempt", 0) + 1
    context = build_context(state.get("graded_chunks") or [], state.get("web_results"))
    result = check_grounding(
        state.get("answer", ""),
        context,
        citations=state.get("citations"),
        unverified=state.get("unverified"),
        client=client,
    )
    result["attempt"] = attempt

    delta: dict = {
        "grounding": result,
        "usage": {**state.get("usage", {}), f"grounding_{attempt}": result.pop("usage", {})},
    }
    if result["verdict"] == UNGROUNDED and attempt >= MAX_ATTEMPTS:
        delta["answer"] = refusal_for_route(state.get("route"))
        result["refused_after_retry"] = True
    return delta


def after_grounding(state: State) -> str:
    """One stricter regeneration on `ungrounded`, then stop.

    The cap lives here rather than in the node so the loop cannot be re-entered,
    exactly as `after_grade` caps the rewrite loop.
    """
    grounding = state.get("grounding", {})
    if grounding.get("verdict") == UNGROUNDED and grounding.get("attempt", 1) < MAX_ATTEMPTS:
        # Regenerate with whichever generator produced the answer — sending a
        # single-paper answer to the library generator would change the question.
        return "single_paper" if state.get("route") == "single_paper" else "generate"
    return "end"


def direct_node(state: State) -> dict:
    """Conversational reply with no retrieval. Filled in after 3.2."""
    return {"answer": ""}


def after_grade(state: State) -> str:
    """Too few surviving chunks means retrieval missed — rewrite and try again.

    The cap is checked here rather than inside the rewrite node so the loop can
    never be entered a third time.
    """
    too_few = len(state["graded_chunks"]) < MIN_GRADED_CHUNKS
    if too_few and state["rewrite_count"] < MAX_REWRITES:
        return "rewrite"
    return "generate"
