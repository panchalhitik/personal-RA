"""Route classification: which of four paths should this question take?

Haiku with forced tool use — a classification job where speed matters, and the
schema is guaranteed valid because `tool_choice` names the tool and `strict`
validates its input. No prompt caching here: the router prompt is ~700 tokens and
Haiku's minimum cacheable prefix is 4096, so a cache_control marker would be a
no-op that still bills the write premium.
"""

from __future__ import annotations

import anthropic

from personal_ra.graph.state import Route, State

ROUTER_MODEL = "claude-haiku-4-5"

# $ per million tokens, claude-haiku-4-5. ask.py's _usage_dict hardcodes Sonnet
# prices; a local copy is cheaper than refactoring that module (v3 rule: the graph
# calls existing code, it doesn't rewrite it).
PRICE_INPUT = 1.00
PRICE_OUTPUT = 5.00

HISTORY_TURNS = 6  # recent context is what disambiguates follow-ups; older turns add noise

ROUTER_SYSTEM = """You route questions about a personal library of ML/NLP safety \
research papers to one of four handlers. You never answer the question — you only \
classify it.

The routes:

single_paper — a paper is currently open and the question is about *that* paper. \
The handler sends the whole paper in one context window, so it can answer about the \
argument as a whole. Choose this when a paper is open and the question uses deictic \
language ("this paper", "their method", "the ablation", "the authors", "Figure 3") or \
is otherwise clearly scoped to the open paper.

library — the question spans several papers, or no paper is open. The handler \
retrieves fragments from across the library. Choose this for "which of my papers...", \
comparisons between named works, and any substantive question asked with no paper open.

web — answering needs information that is not in the library at all: whether a \
follow-up has been published, whether a newer version exists, the current state of the \
art on a benchmark, who cited a paper. Anything about events or publications after the \
library was assembled. This route costs money and needs the user's approval, so only \
choose it when the library genuinely cannot hold the answer.

direct — conversational, needing no retrieval at all. Thanks, greetings, "summarize \
what you just told me", "rephrase that more simply", "what did I ask earlier". Anything \
answerable from the conversation so far.

Rules:

1. When torn between single_paper and library and a paper IS open, choose \
single_paper. Whole-paper context is strictly more informative than fragments of that \
same paper, so the cost of being wrong is much lower in that direction.
2. When no paper is open, single_paper is impossible. Choose library, web, or direct.
3. Use the conversation history. "What about the second one?" and "does that hold at \
scale?" only mean something relative to what came before, and they inherit the subject \
of the previous turn.
4. A question is only direct if it needs NOTHING from the papers. If it asks for a new \
fact, it is not direct even when phrased casually.
5. Do not choose web just because a question is hard. Choose it only when the answer \
lies outside the library by its nature.

Always give a reason: one sentence, naming the specific signal you routed on."""

ROUTER_TOOL = {
    "name": "classify_route",
    "description": "Record the route this question should take, and why.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "route": {
                "type": "string",
                "enum": ["single_paper", "library", "web", "direct"],
                "description": "Which handler should answer this question.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "One sentence naming the signal you routed on — the deictic phrase, "
                    "the cross-paper scope, the out-of-library need, or the conversational "
                    "cue. This is shown to the user and read in traces, so be specific."
                ),
            },
        },
        "required": ["route", "reason"],
        "additionalProperties": False,
    },
}


def _usage(response: object) -> dict:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cost = (input_tokens * PRICE_INPUT + output_tokens * PRICE_OUTPUT) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
    }


def build_classification_input(state: State) -> str:
    """The user turn: paper state, then recent history, then the question.

    Paper state comes first because rule 2 depends on it — the model needs to know
    single_paper is even available before it reads the question.
    """
    paper_id = state.get("paper_id")
    parts = [
        f"Paper currently open: {paper_id}"
        if paper_id
        else "Paper currently open: none — single_paper is not available."
    ]

    history = (state.get("history") or [])[-HISTORY_TURNS:]
    if history:
        turns = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        parts.append(f"Recent conversation:\n{turns}")

    parts.append(f"Question to route: {state['question']}")
    return "\n\n".join(parts)


def classify_route(
    state: State, client: anthropic.Anthropic | None = None
) -> tuple[Route, str, dict]:
    """Returns (route, reason, usage). An explicit `route` in state wins — that is the
    test/debug override, and Streamlit uses it when the user pins a route by hand."""
    forced = state.get("route")
    if forced:
        return forced, "route supplied by caller (classifier bypassed)", {}

    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=ROUTER_MODEL,
        max_tokens=256,
        temperature=0,
        system=ROUTER_SYSTEM,
        tools=[ROUTER_TOOL],
        tool_choice={"type": "tool", "name": "classify_route"},
        messages=[{"role": "user", "content": build_classification_input(state)}],
    )

    for block in response.content:
        if getattr(block, "type", "") == "tool_use" and block.name == "classify_route":
            return block.input["route"], block.input["reason"], _usage(response)

    # Forced tool_choice makes this unreachable in practice, but a router that raises
    # takes the whole graph down. Fall back to the same heuristic the 3.1 stub used.
    fallback: Route = "single_paper" if state.get("paper_id") else "library"
    return (
        fallback,
        "classifier returned no decision; fell back on whether a paper is open",
        _usage(response),
    )


def after_route(state: State) -> Route:
    """Fan out to one of the four route branches."""
    return state["route"]
