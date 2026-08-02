"""Binary relevance grading: does this chunk help answer the question?

Retrieval returns the closest chunks, not the useful ones — a query about an
ablation will happily surface the related-work paragraph that mentions the same
words. The grader is the filter that notices, and its output drives the rewrite
loop: fewer than MIN_GRADED_CHUNKS survivors means retrieval missed.

Chunks are graded **concurrently**. Eight serial Haiku round-trips is the obvious
latency trap in a node that exists to make the graph faster to trust.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import anthropic

GRADER_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 128

# $ per million tokens, claude-haiku-4-5.
PRICE_INPUT = 1.00
PRICE_OUTPUT = 5.00

GRADER_SYSTEM = """You judge whether a single excerpt from a research paper helps \
answer a question. You are a filter, not an answerer — never try to answer the \
question yourself.

Mark an excerpt relevant when it contains part of the answer, evidence bearing on \
it, or a definition the answer depends on. Partial help counts: an excerpt that \
supplies one of several needed numbers is relevant.

Many questions cannot be answered by any single excerpt. When the question compares \
or contrasts two things, spans several papers, or asks which papers do something, \
judge each excerpt on whether it supplies ONE PART of that answer. The answer is \
assembled from several excerpts, and rejecting each one for being incomplete leaves \
nothing to answer from at all.

Two rejection reasons are therefore never valid, and you must not give them:

- "does not compare X and Y" — no single excerpt does, and it is not the excerpt's \
job to. If the excerpt substantively covers X, or covers Y, it is relevant.
- "cannot confirm this is the paper by that author or lab" — you cannot tell who \
wrote an excerpt and you do not need to. Retrieval already established which papers \
these came from. Judge the subject matter alone.

None of this rescues genuine non-content. Keep rejecting author lists, \
bibliographies, citation lists, acknowledgements, impact statements and headers, \
whatever the question is — they carry no findings for any answer.

Mark it irrelevant when it merely shares vocabulary with the question — a related-work \
paragraph that name-drops the topic, a citation list, a header, boilerplate, or a \
passage about a different method that happens to use the same terms. Topic overlap is \
not relevance.

Excerpts are fragments of a larger paper, so judge generously on completeness: an \
excerpt that clearly begins an answer is relevant even if it is cut off. Judge \
strictly on subject: an excerpt about a different thing is irrelevant however \
well-written it is.

Give a reason of at most fifteen words. For an irrelevant excerpt, say what it is \
about instead — that is the signal used to rewrite the query, so name the excerpt's \
actual subject and vocabulary."""

GRADE_TOOL = {
    "name": "grade_excerpt",
    "description": "Record whether this excerpt helps answer the question.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "relevant": {
                "type": "boolean",
                "description": "True if the excerpt helps answer the question.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "At most fifteen words. When irrelevant, name what the excerpt is "
                    "actually about — the rewrite step reads this to learn the corpus's "
                    "vocabulary."
                ),
            },
        },
        "required": ["relevant", "reason"],
        "additionalProperties": False,
    },
}


def _usage(response: object) -> dict:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
    }


def _prompt(question: str, chunk: dict) -> str:
    meta = chunk.get("metadata", {})
    return (
        f'<excerpt paper="{meta.get("paper_title", "unknown")}" '
        f'page="{meta.get("page", "?")}">\n{chunk.get("text", "")}\n</excerpt>\n\n'
        f"Question: {question}"
    )


async def _grade_one(client, question: str, chunk: dict) -> tuple[bool, str, dict]:
    """Grade one chunk. Fails open: a chunk we could not grade is kept, not dropped.

    Failing closed would let a transient API error trigger a spurious rewrite, and
    the grounding check in 3.6 is the backstop for content that should not survive.
    """
    try:
        response = await client.messages.create(
            model=GRADER_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system=GRADER_SYSTEM,
            tools=[GRADE_TOOL],
            tool_choice={"type": "tool", "name": "grade_excerpt"},
            messages=[{"role": "user", "content": _prompt(question, chunk)}],
        )
    except Exception as exc:  # noqa: BLE001 — one bad grade must not fail the node
        return True, f"kept ungraded: {type(exc).__name__}", {}

    for block in response.content:
        if getattr(block, "type", "") == "tool_use" and block.name == "grade_excerpt":
            return bool(block.input["relevant"]), block.input["reason"], _usage(response)
    return True, "kept ungraded: grader returned no verdict", _usage(response)


async def grade_chunks_async(
    question: str, chunks: list[dict], client=None
) -> tuple[list[dict], list[dict], dict]:
    """Grade every chunk concurrently. Returns (kept, rejected, usage).

    `question` must be the *original* question, not a rewrite — the rewrite is a
    retrieval device, and relevance is judged against what the user actually asked.
    """
    if not chunks:
        return [], [], {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "n_graded": 0}

    client = client or anthropic.AsyncAnthropic()
    grades = await asyncio.gather(*(_grade_one(client, question, c) for c in chunks))

    kept, rejected = [], []
    tokens_in = tokens_out = 0
    for chunk, (relevant, reason, usage) in zip(chunks, grades):
        tokens_in += usage.get("input_tokens", 0)
        tokens_out += usage.get("output_tokens", 0)
        (kept if relevant else rejected).append({**chunk, "grade_reason": reason})

    cost = (tokens_in * PRICE_INPUT + tokens_out * PRICE_OUTPUT) / 1_000_000
    return (
        kept,
        rejected,
        {
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cost_usd": round(cost, 6),
            "n_graded": len(chunks),
        },
    )


def grade_chunks(
    question: str, chunks: list[dict], client=None
) -> tuple[list[dict], list[dict], dict]:
    """Sync entry point for the sync graph. Step 3.10's FastAPI layer, which already
    has a loop running, should await `grade_chunks_async` directly instead."""

    def run() -> tuple[list[dict], list[dict], dict]:
        return asyncio.run(grade_chunks_async(question, chunks, client=client))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return run()
    # A loop is already running on this thread; asyncio.run would raise. Hand the
    # whole batch to a worker thread that owns its own loop.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(run).result()
