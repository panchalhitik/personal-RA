"""Route classification and the branch out of the route node.

Step 3.1 ships a placeholder classifier so the graph traverses. Step 3.2 replaces
`classify_route` with a claude-haiku-4-5 call that reads conversation history.
"""

from __future__ import annotations

from personal_ra.graph.state import Route, State


def classify_route(state: State) -> tuple[Route, str]:
    """Placeholder: honour an explicit route, else fall back on whether a paper is open.

    Returns (route, reason). `route_reason` is recorded on every path — it is what
    makes a trace readable and what §3.8 uses to debug misroutes.
    """
    forced = state.get("route")
    if forced:
        return forced, "route supplied by caller (classifier not yet wired)"
    if state.get("paper_id"):
        return "single_paper", "stub: a paper is open, so default to whole-paper context"
    return "library", "stub: no paper open, so search the library"


def after_route(state: State) -> Route:
    """Fan out to one of the four route branches."""
    return state["route"]
