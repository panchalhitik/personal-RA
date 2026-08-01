"""Every node in the router graph.

Step 3.1 fills these with pass-through stubs so the edges can be wired and
traversed first. Each stub names the step that replaces it. Stubs invent no data:
where a real node would produce chunks or an answer, the stub produces nothing,
which is why the library route currently exercises the rewrite loop to its cap.
"""

from __future__ import annotations

from personal_ra.graph.router import classify_route
from personal_ra.graph.state import MAX_REWRITES, MIN_GRADED_CHUNKS, State


def route_node(state: State) -> dict:
    """Classify the question into one of four routes. Step 3.2 makes this real."""
    route, reason = classify_route(state)
    return {"route": route, "route_reason": reason}


def single_paper_node(state: State) -> dict:
    """Whole paper in a cached system prompt via ask.py. Filled in after 3.2."""
    return {"answer": "", "citations": [], "unverified": []}


def retrieve_node(state: State) -> dict:
    """Hybrid retrieval via search.py, k=30 so 3.3 has depth to rerank. Filled in after 3.2."""
    return {"chunks": []}


def rerank_node(state: State) -> dict:
    """Cross-encoder rerank, keep top 8. Step 3.3."""
    return {"chunks": state["chunks"]}


def grade_node(state: State) -> dict:
    """Binary relevance per chunk against original_question, graded concurrently. Step 3.4."""
    return {"graded_chunks": state["chunks"], "rejected_chunks": []}


def rewrite_node(state: State) -> dict:
    """Rewrite toward more specific technical vocabulary using the rejected chunks. Step 3.5.

    The counter increment is real, not a stub: it is what makes the loop terminate.
    """
    return {
        "question": state["question"],
        "rewrite_count": state["rewrite_count"] + 1,
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


def grounding_node(state: State) -> dict:
    """Verify quotes via cite.py, then check unquoted claims against context. Step 3.6."""
    return {"grounding": {"verdict": "not_checked", "unsupported": []}}


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
