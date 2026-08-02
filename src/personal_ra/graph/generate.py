"""Answer generation for the graph's two answering routes.

`single_paper` sends the whole paper in a cached system prompt (v0's core insight,
via ask.py). `library` sends graded excerpts (via search.py). Both keep the
verbatim-quote discipline, so cite.py can verify every quote before the grounding
node ever sees the answer.

The stricter variant exists for the one regeneration the grounding check is allowed
to ask for. It does not add rules — it removes the room the first attempt used to
overreach in.
"""

from __future__ import annotations

from personal_ra.ask import Message, ask
from personal_ra.graph.web import format_web_context
from personal_ra.search import LIBRARY_PROMPT, answer_from_chunks

# Appended on a regeneration, never on the first attempt. Phrased as a narrowing of
# what may be said rather than as a scolding — "be more careful" changes nothing,
# "say only what you can quote" changes the output.
STRICTER = """

The previous attempt at this answer made claims the excerpts did not support. Write \
it again under a harder rule: every substantive statement must be either a verbatim \
<quote> from the excerpts or an obvious restatement of one. Do not connect two \
excerpts into a claim neither makes on its own. Do not supply context from general \
knowledge, however standard. If the excerpts cannot answer the question, say so \
plainly instead of assembling an answer from fragments."""

NOTE_THIN_RETRIEVAL = (
    "\n\nRetrieval was rewritten twice and still returned little. Say so in one "
    "sentence at the start of your answer, then answer from what is here."
)


def library_system_prompt(stricter: bool = False, thin: bool = False) -> str:
    prompt = LIBRARY_PROMPT
    if thin:
        prompt += NOTE_THIN_RETRIEVAL
    if stricter:
        prompt += STRICTER
    return prompt


def generate_library_answer(
    question: str,
    chunks,
    web_results: list[dict] | None = None,
    client=None,
    stricter: bool = False,
    thin: bool = False,
):
    """Answer from graded excerpts, plus any approved web results.

    Web results go in as `extra_context`, which `answer_from_chunks` labels
    <web_result>. Only <paper> excerpts are quote-verified against a source, so a
    web snippet structurally cannot acquire a page citation.
    """
    return answer_from_chunks(
        question,
        chunks,
        client=client,
        system=library_system_prompt(stricter=stricter, thin=thin),
        extra_context=format_web_context(web_results or []),
    )


def generate_single_paper_answer(
    paper,
    question: str,
    history: list[dict] | None = None,
    client=None,
    stricter: bool = False,
):
    """Whole paper in a cached system prompt, via ask.py.

    ask.py owns the prompt and the cache_control block, so the stricter variant is
    appended to the question rather than to the system prompt — editing the system
    prompt would invalidate the very prompt cache that makes this route cheap.
    """
    messages = [Message(role=m["role"], content=m["content"]) for m in history or []]
    prompt = question + (STRICTER if stricter else "")
    return ask(paper, prompt, history=messages, client=client)
