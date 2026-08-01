"""Web search, and the approval gate in front of it.

Two things make this node different from every other one in the graph:

**It spends money the user did not explicitly authorise.** Every other node reads
the library the user already built. This one calls a paid third-party API, so the
graph halts and asks first — LangGraph's `interrupt`, persisted through the
checkpointer, so the pending question survives a process restart.

**Its results are not the library.** A web snippet has no page to cite and has not
been verified against anything the user owns. It is marked `source: "web"` and
carries a URL, never a page number. The whole point of this project is knowing
where an answer came from, so the two kinds of evidence must not be able to blur
together — `web_citation` structurally cannot produce a page.
"""

from __future__ import annotations

import os

MAX_RESULTS = 5
SEARCH_DEPTH = "advanced"
API_KEY_ENV = "TAVILY_API_KEY"

# Tavily bills in credits, not dollars: basic search is 1, advanced is 2. The dollar
# value depends on the plan (the free tier is 1,000 credits/month at no cost), so the
# approval payload reports credits — the unit that is actually true — and leaves USD
# to whoever knows the plan.
CREDITS_PER_SEARCH = 2


class MissingTavilyKey(RuntimeError):
    """Raised with the setup instructions rather than a bare KeyError."""


def approval_payload(state) -> dict:
    """What the user sees when the graph stops and asks.

    Query, why the router wanted the web, and what it costs — enough to answer
    without reading the trace.
    """
    return {
        "action": "web_search",
        "query": state.get("question", ""),
        "original_question": state.get("original_question", ""),
        "reason": state.get("route_reason", ""),
        "provider": "tavily",
        "search_depth": SEARCH_DEPTH,
        "max_results": MAX_RESULTS,
        "estimated_cost": f"{CREDITS_PER_SEARCH} Tavily credits (free tier: 1,000/month)",
    }


def interpret_decision(decision) -> bool:
    """Accept the shapes a UI, a CLI, or an HTTP client would each naturally send.

    Anything unrecognised is a denial: defaulting an unclear answer to "yes, spend
    money and leave the library" is the wrong way to be wrong.
    """
    if isinstance(decision, bool):
        return decision
    if isinstance(decision, str):
        return decision.strip().lower() in {"approve", "approved", "yes", "y", "true", "ok"}
    if isinstance(decision, dict):
        for key in ("approved", "approve", "decision"):
            if key in decision:
                return interpret_decision(decision[key])
    return False


def _client():
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise MissingTavilyKey(
            f"{API_KEY_ENV} is not set. Get a key at https://app.tavily.com "
            f"(free tier, no card) and add {API_KEY_ENV}=tvly-... to .env"
        )
    from tavily import TavilyClient

    return TavilyClient(api_key=key)


def search_web(query: str, client=None, max_results: int = MAX_RESULTS) -> list[dict]:
    """Tavily advanced search. Returns results tagged as external, with no page.

    Failures return an empty list rather than raising: a dead web search should
    degrade the answer, not destroy a run that was approved and already paid for.
    """
    client = client or _client()
    try:
        response = client.search(query, search_depth=SEARCH_DEPTH, max_results=max_results)
    except Exception:  # noqa: BLE001 — a failed search must not fail the graph
        return []

    results = []
    for item in (response or {}).get("results", [])[:max_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score"),
                "source": "web",
            }
        )
    return results


def web_citation(result: dict) -> dict:
    """A web result as a citation. Structurally incapable of carrying a page number.

    Built by whitelisting fields rather than copying and deleting, so a future
    Tavily field named `page` cannot leak through and make a web snippet look like
    a verified library quote.
    """
    return {
        "source": "web",
        "title": result.get("title", ""),
        "url": result.get("url", ""),
        "verified": False,  # nothing was string-matched against a source we hold
    }


def format_web_context(results: list[dict]) -> str:
    """Web results for the generator, labelled so they cannot pass as library text."""
    return "\n\n".join(
        f'<web_result title="{r.get("title", "")}" url="{r.get("url", "")}">\n'
        f"{r.get('content', '')}\n</web_result>"
        for r in results
    )
