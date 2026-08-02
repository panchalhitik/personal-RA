"""Splitting a comparison question into one search per thing being compared.

The 60-run eval left q48 and q49 answering with zero citations after the grader
fix, and the cause was retrieval, not grading: "How do the OpenAI and DeepMind
approaches differ?" retrieves eight chunks from ONE of the two papers. One side of
the comparison never enters the pool, so no grading rule and no rewrite can
assemble an answer from it.

One query per side fixes the pool. Interleaving the results — rank 1 from each,
then rank 2 from each — is what stops the better-matching side from taking all
eight slots, which plain concatenation would allow.
"""

from __future__ import annotations

import re

import anthropic

SPLITTER_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 512
MAX_FACETS = 4

# $ per million tokens, claude-haiku-4-5.
PRICE_INPUT = 1.00
PRICE_OUTPUT = 5.00

# A cheap gate so single-subject questions never pay for a splitter call. It is
# deliberately loose: a false positive costs one Haiku call and returns one facet,
# while a false negative leaves the original bug in place.
MULTI_PART = re.compile(
    r"\b(differ|differs|differences?|compare[sd]?|comparison|contrast(?:s|ed)?|"
    r"versus|vs\.?|both|either|respectively)\b",
    re.IGNORECASE,
)

SPLITTER_SYSTEM = """You split a question about research papers into separate search \
queries, one per thing the question is asking about.

These queries are run as INDEPENDENT searches against a corpus of paper excerpts. \
Nothing carries over between them, so each query must name its own subject \
explicitly and stand completely alone. "the second one", "the other approach" and \
"how it differs" retrieve nothing.

Comparison questions get one query per side. "How do STAR-1 and RealSafe-R1 differ \
in aligning reasoning models?" becomes two queries: one about STAR-1's alignment \
method, one about RealSafe-R1's. Drop the comparison itself — no excerpt contains \
it, and the comparison is made later from what the searches return.

When the two sides are described rather than named ("the two DeepMind \
chain-of-thought papers"), write each query around the distinguishing description \
plus the shared subject, so the searches land in different places.

Keep the technical vocabulary from the original question: this corpus rewards \
distinctive terms, and a query stripped back to common words retrieves worse.

If the question is really about a single subject, return exactly one query — that \
is the normal case and there is nothing wrong with it."""

SPLIT_TOOL = {
    "name": "split_query",
    "description": "Record one standalone search query per subject in the question.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "description": (
                    "One standalone search query per subject, in the order the "
                    "question mentions them. Exactly one entry when the question has "
                    "a single subject."
                ),
                "items": {"type": "string"},
            }
        },
        "required": ["queries"],
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


def looks_multi_part(question: str) -> bool:
    """Cheap gate: is this worth a splitter call at all?"""
    return bool(MULTI_PART.search(question or ""))


def decompose_question(question: str, client=None) -> tuple[list[str], dict]:
    """Return (queries, usage). Always returns at least the original question.

    Every failure path degrades to a single-query search, which is exactly the
    behaviour before this module existed — a broken splitter must not be able to
    make retrieval worse than not having one.
    """
    if not looks_multi_part(question):
        return [question], {}

    client = client or anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=SPLITTER_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system=SPLITTER_SYSTEM,
            tools=[SPLIT_TOOL],
            tool_choice={"type": "tool", "name": "split_query"},
            messages=[{"role": "user", "content": question}],
        )
    except Exception:  # noqa: BLE001 — a failed split falls back to one query
        return [question], {}

    for block in response.content:
        if getattr(block, "type", "") == "tool_use" and block.name == "split_query":
            queries = [q.strip() for q in (block.input or {}).get("queries") or [] if q.strip()]
            return (queries or [question])[:MAX_FACETS], _usage(response)
    return [question], _usage(response)


def interleave(rankings: list[list], k: int, key=lambda item: item.id) -> list:
    """Round-robin merge: rank 1 from every ranking, then rank 2, and so on.

    Not concatenation. The two sides of a comparison are rarely equally well
    matched, and concatenating would let the stronger side fill all k slots — which
    is the exact failure this whole module exists to prevent. Round-robin guarantees
    every side is represented before any side gets a second slot.
    """
    merged, seen = [], set()
    for rank in range(max((len(r) for r in rankings), default=0)):
        for ranking in rankings:
            if rank >= len(ranking):
                continue
            item = ranking[rank]
            identity = key(item)
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(item)
            if len(merged) >= k:
                return merged
    return merged
