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
# Four fields including a short free-text subject. At 128 the JSON truncated and
# came back missing `reason`, which crashed the node.
MAX_TOKENS = 256

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

Report `on_topic` separately from `relevant`. An excerpt is on topic when it \
concerns the same subject matter as the question — the same methods, systems, \
phenomena, benchmarks or papers — even when it does not answer the question by \
itself. This is the difference between "one piece of a multi-part answer" and "about \
something else entirely", and it is the more important of the two judgements when \
you are unsure. An excerpt about a different research area is not on topic.

Genuine non-content is never on topic and never relevant: author lists, \
bibliographies, citation lists, acknowledgements, impact statements, checklists and \
headers carry no findings for any question, whatever the excerpt is nominally about.

Mark it irrelevant when it merely shares vocabulary with the question — a related-work \
paragraph that name-drops the subject, boilerplate, or a passage about a different \
method that happens to use the same terms. Vocabulary overlap is not relevance, and \
it is not on_topic either: matching words is not the same as concerning the same \
subject matter.

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
        # Field order is generation order, and it is load-bearing. Asked for
        # `relevant` first, the model wrote one justification ("does not compare X
        # and Y") and made both booleans follow it, marking excerpts about the exact
        # subject as off-topic. Naming the subject first, without reference to the
        # question, forces the two judgements apart.
        "properties": {
            "subject": {
                "type": "string",
                "description": (
                    "In at most eight words, what this excerpt is about. Describe the "
                    "excerpt on its own terms; do not mention the question."
                ),
            },
            "on_topic": {
                "type": "boolean",
                "description": (
                    "Does that subject match the question's subject — the same methods, "
                    "systems, phenomena or papers? Answer about subject matter only, "
                    "ignoring whether the excerpt answers the question. False for a "
                    "different research area, and false for non-content."
                ),
            },
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
        "required": ["subject", "on_topic", "relevant", "reason"],
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


async def _grade_one(client, question: str, chunk: dict) -> tuple[bool, bool, str, dict]:
    """Grade one chunk -> (relevant, on_topic, reason, usage).

    Fails open: a chunk we could not grade is kept, not dropped. Failing closed
    would let a transient API error trigger a spurious rewrite, and the grounding
    check in 3.6 is the backstop for content that should not survive.
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
        return True, True, f"kept ungraded: {type(exc).__name__}", {}

    for block in response.content:
        if getattr(block, "type", "") == "tool_use" and block.name == "grade_excerpt":
            # .get throughout: strict tool use guarantees the schema, but a
            # truncated response can still arrive with fields missing, and a
            # KeyError here would take down the whole grading batch.
            fields = block.input or {}
            relevant = bool(fields.get("relevant", False))
            return (
                relevant,
                bool(fields.get("on_topic", relevant)),
                fields.get("reason") or fields.get("subject") or "no reason recorded",
                _usage(response),
            )
    return True, True, "kept ungraded: grader returned no verdict", _usage(response)


async def grade_chunks_async(
    question: str, chunks: list[dict], client=None
) -> tuple[list[dict], list[dict], dict]:
    """Grade every chunk concurrently. Returns (kept, rejected, usage).

    `question` must be the *original* question, not a rewrite — the rewrite is a
    retrieval device, and relevance is judged against what the user actually asked.
    """
    if not chunks:
        return (
            [],
            [],
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "n_graded": 0,
                "n_kept": 0,
                "n_partial": 0,
            },
        )

    client = client or anthropic.AsyncAnthropic()
    grades = await asyncio.gather(*(_grade_one(client, question, c) for c in chunks))

    kept, rejected, n_partial = [], [], 0
    tokens_in = tokens_out = 0
    for chunk, (relevant, on_topic, reason, usage) in zip(chunks, grades):
        tokens_in += usage.get("input_tokens", 0)
        tokens_out += usage.get("output_tokens", 0)
        graded = {**chunk, "grade_reason": reason, "partial": not relevant and on_topic}
        # On-topic-but-incomplete survives. A comparison question has no excerpt that
        # answers it alone, and dropping every partial leaves nothing to assemble
        # from — which is exactly how comparison questions came back with zero
        # citations. Off-topic chunks are still rejected, so an unanswerable question
        # still empties the pool and still ends in a refusal.
        if relevant or on_topic:
            n_partial += graded["partial"]
            kept.append(graded)
        else:
            rejected.append(graded)

    cost = (tokens_in * PRICE_INPUT + tokens_out * PRICE_OUTPUT) / 1_000_000
    return (
        kept,
        rejected,
        {
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cost_usd": round(cost, 6),
            "n_graded": len(chunks),
            "n_kept": len(kept),
            "n_partial": n_partial,
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
