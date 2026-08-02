"""Single-paper Q&A: whole paper in a cached system prompt, verified quotes out."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv

from personal_ra.cite import Citation, verify_quote
from personal_ra.parse import Paper, parse_pdf
from personal_ra.vision import enrich_paper

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 2048
REFUSAL = "That isn't covered in this paper."

# Distinct from REFUSAL on purpose. REFUSAL is the model reading the paper and
# telling you the answer isn't in it — a correct outcome. This is the request never
# being answered at all, which is a failure, and the two must never be counted
# together.
API_REFUSAL = "The API declined this request"

# $ per million tokens, claude-sonnet-4-5 (cache write = 1.25x input, read = 0.1x)
PRICE_INPUT = 3.00
PRICE_OUTPUT = 15.00
PRICE_CACHE_WRITE = 3.75
PRICE_CACHE_READ = 0.30

SYSTEM_PROMPT = f"""You are a careful research assistant. You answer questions strictly from \
the research paper provided in the <paper> block, never from general knowledge.

Rules:
- Support every substantive claim with a short verbatim quote from the paper, wrapped in \
<quote></quote> tags. Copy the quote character-for-character from the paper text; never \
paraphrase, alter, or abbreviate inside the tags. Keep each quote to one or two sentences.
- The paper text contains [PAGE N] markers showing where each page begins. Never include \
these markers inside a quote.
- If the question is not covered by the paper, reply exactly: {REFUSAL}"""

_QUOTE_RE = re.compile(r"<quote>(.*?)</quote>", re.DOTALL)


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class Answer:
    text: str  # quotes replaced with inline [p. N] markers
    citations: list[Citation]
    unverified: list[Citation]  # surfaced separately as a warning
    usage: dict = field(default_factory=dict)


def _postprocess(raw: str, paper: Paper) -> tuple[str, list[Citation], list[Citation]]:
    citations: list[Citation] = []
    unverified: list[Citation] = []

    def replace(match: re.Match[str]) -> str:
        citation = verify_quote(match.group(1).strip(), paper)
        if citation.verified:
            citations.append(citation)
            # Keep the quote visible inline; the page marker is a suffix, not a
            # replacement — otherwise answers that lean on quotes lose their substance.
            return f'"{citation.quote}" [p. {citation.page}]'
        unverified.append(citation)
        return "[unverified]"

    return _QUOTE_RE.sub(replace, raw).strip(), citations, unverified


def _usage_dict(usage: object) -> dict:
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cost = (
        input_tokens * PRICE_INPUT
        + output_tokens * PRICE_OUTPUT
        + cache_write * PRICE_CACHE_WRITE
        + cache_read * PRICE_CACHE_READ
    ) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_write_tokens": cache_write,
        "cache_read_tokens": cache_read,
        "cost_usd": round(cost, 6),
    }


def _refusal_answer(response) -> Answer:
    """Anthropic's safety classifiers declined the request.

    The response is a normal HTTP 200 with `stop_reason: "refusal"` and an EMPTY
    content list, so the usual text-joining produces "" and the caller shows a blank
    answer with no explanation. Whole-paper prompts about adversarial or
    safety-critical research trip this fairly often, and it is a false positive on
    that material — but silence is the worst possible way to report it.
    """
    details = getattr(response, "stop_details", None)
    category = getattr(details, "category", None) or "unspecified"
    return Answer(
        text=(
            f"{API_REFUSAL} — Anthropic's safety classifier declined it "
            f"(category: {category}). The question was not answered at all, so there "
            f"is nothing to cite. This is a known false positive on papers about "
            f"adversarial and safety-critical topics; asking about a narrower part of "
            f"the paper, or using library search instead of whole-paper mode, usually "
            f"gets through."
        ),
        citations=[],
        unverified=[],
        usage={**_usage_dict(response.usage), "api_refusal": True, "refusal_category": category},
    )


def ask(
    paper: Paper,
    question: str,
    history: list[Message] | None = None,
    client: anthropic.Anthropic | None = None,
) -> Answer:
    client = client or anthropic.Anthropic()
    system = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {
            "type": "text",
            "text": f'<paper title="{paper.title}">\n{paper.full_text}\n</paper>',
            "cache_control": {"type": "ephemeral"},
        },
    ]
    messages = [{"role": m.role, "content": m.content} for m in history or []]
    messages.append({"role": "user", "content": question})

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        system=system,
        messages=messages,
    )
    # Check stop_reason BEFORE reading content: on a refusal the list is empty, and
    # joining it silently yields "" rather than raising.
    if getattr(response, "stop_reason", None) == "refusal":
        return _refusal_answer(response)

    raw = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    text, citations, unverified = _postprocess(raw, paper)
    return Answer(
        text=text,
        citations=citations,
        unverified=unverified,
        usage=_usage_dict(response.usage),
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Ask a question about one paper.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("question")
    ap.add_argument(
        "--no-vision",
        action="store_true",
        help="skip vision transcription of equation-heavy pages",
    )
    args = ap.parse_args(argv)

    load_dotenv()
    paper = parse_pdf(args.pdf)
    if not args.no_vision:
        paper = enrich_paper(paper)
    print(f"Paper: {paper.title} ({len(paper.pages)} pages, ~{paper.n_tokens} tokens)\n")
    answer = ask(paper, args.question)

    print(answer.text)
    if answer.citations:
        print("\nCitations:")
        for c in answer.citations:
            print(f'  [p. {c.page}] ({c.match_type}) "{c.quote}"')
    if answer.unverified:
        print("\nWARNING - quotes that could not be verified against the paper:")
        for c in answer.unverified:
            print(f'  "{c.quote}"')
    u = answer.usage
    print(
        f"\nUsage: in={u['input_tokens']} out={u['output_tokens']} "
        f"cache_write={u['cache_write_tokens']} cache_read={u['cache_read_tokens']} "
        f"cost=${u['cost_usd']:.4f}"
    )


if __name__ == "__main__":
    main()
