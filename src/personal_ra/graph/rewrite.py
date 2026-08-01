"""Query rewriting after a failed retrieval.

Fires only when grading leaves fewer than MIN_GRADED_CHUNKS survivors, and at most
MAX_REWRITES times. The useful signal is the *rejected* chunks: they are what
retrieval actually found near this query, so they reveal the corpus's vocabulary in
that neighbourhood — and the grader's reasons say what each one was about instead.

Direction matters and is counter-intuitive. The v1 eval found this corpus rewards
distinctive technical terms (BM25 beat dense retrieval on recall@1), so a rewrite
that *generalises* the query retrieves worse, not better. The prompt says so
explicitly.
"""

from __future__ import annotations

import anthropic

REWRITER_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 256

# $ per million tokens, claude-haiku-4-5.
PRICE_INPUT = 1.00
PRICE_OUTPUT = 5.00

MAX_REJECTED_IN_PROMPT = 8
REJECTED_EXCERPT_CHARS = 400

REWRITER_SYSTEM = """You repair a failed search query over a personal library of \
ML/NLP safety research papers.

Retrieval ran and every excerpt it found was judged irrelevant. Your job is to \
rewrite the query so retrieval finds different and better excerpts. You never answer \
the question.

Retrieval is hybrid: semantic embeddings fused with BM25 keyword matching. BM25 \
rewards distinctive terms, and this corpus has been measured to reward them heavily. \
That drives every rule below.

- Make the query MORE specific, never more general. Broadening is the instinctive \
move and it is wrong here — a general query pulls the same generic passages that \
were just rejected.
- Prefer distinctive technical vocabulary: named methods, benchmarks, metrics, and \
phenomena ("sandbagging", "emergent misalignment", "attack success rate", \
"Best-of-n"). One precise term is worth a dozen common words.
- Mine the rejected excerpts for vocabulary, then steer away from their subject. \
They show how this corpus words things near your query, which is useful even though \
the excerpts themselves were wrong.
- Preserve the user's intent exactly. A query that retrieves beautifully and answers \
a different question is a failure.
- Write keywords and noun phrases, not a polite sentence. Drop "what", "how", \
"does", "the paper" — they match everything and discriminate nothing.

Give a rationale of at most twenty words explaining what you changed and why."""

REWRITE_TOOL = {
    "name": "rewrite_query",
    "description": "Record the rewritten retrieval query.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The rewritten query: keywords and technical noun phrases, more "
                    "specific than the original, same intent."
                ),
            },
            "rationale": {
                "type": "string",
                "description": "At most twenty words on what changed and why.",
            },
        },
        "required": ["query", "rationale"],
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


def build_rewrite_input(
    original_question: str, current_query: str, rejected_chunks: list[dict]
) -> str:
    """Original intent, the query that just failed, and what it dragged back.

    Excerpts are truncated: the grader's reason carries most of the signal, and a
    full 1000-character chunk times eight would crowd out the instructions.
    """
    parts = [f"Original question: {original_question}"]
    if current_query != original_question:
        parts.append(f"Query that just failed: {current_query}")

    if rejected_chunks:
        lines = []
        for chunk in rejected_chunks[:MAX_REJECTED_IN_PROMPT]:
            meta = chunk.get("metadata", {})
            text = (chunk.get("text") or "")[:REJECTED_EXCERPT_CHARS]
            reason = chunk.get("grade_reason", "no reason recorded")
            lines.append(
                f'- from "{meta.get("paper_title", "unknown")}": {text}\n  rejected: {reason}'
            )
        parts.append("Excerpts retrieval found, and why each was wrong:\n" + "\n".join(lines))
    else:
        parts.append("Retrieval returned nothing at all — the query matched no excerpt.")

    parts.append("Rewrite the query.")
    return "\n\n".join(parts)


def rewrite_query(
    original_question: str,
    current_query: str,
    rejected_chunks: list[dict],
    client: anthropic.Anthropic | None = None,
) -> tuple[str, str, dict]:
    """Returns (rewritten_query, rationale, usage).

    Falls back to the current query on any failure: a rewrite that raises would take
    down a graph run that still has surviving chunks to answer from.
    """
    client = client or anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=REWRITER_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system=REWRITER_SYSTEM,
            tools=[REWRITE_TOOL],
            tool_choice={"type": "tool", "name": "rewrite_query"},
            messages=[
                {
                    "role": "user",
                    "content": build_rewrite_input(
                        original_question, current_query, rejected_chunks
                    ),
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 — a failed rewrite must not fail the run
        return current_query, f"rewrite failed ({type(exc).__name__}); query unchanged", {}

    for block in response.content:
        if getattr(block, "type", "") == "tool_use" and block.name == "rewrite_query":
            new_query = (block.input.get("query") or "").strip()
            if new_query:
                return new_query, block.input.get("rationale", ""), _usage(response)
            break
    return current_query, "rewriter returned no query; query unchanged", _usage(response)
