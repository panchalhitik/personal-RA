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

# Retried on when the safety classifier declines the primary model. Measured on the
# six papers that refused: it answers two of them. Not made the default — it rejects
# temperature=0 (spec §2 pins that), tokenises ~30% heavier, and thinks by default.
FALLBACK_MODEL = "claude-sonnet-5"

# $ per million tokens: (input, output, cache write = 1.25x input, cache read = 0.1x).
# Sonnet 5 is on introductory $2/$10 until 2026-08-31; list price is used here so a
# cost estimate is never lower than the bill.
PRICES = {
    "claude-sonnet-4-5": (3.00, 15.00, 3.75, 0.30),
    "claude-sonnet-5": (3.00, 15.00, 3.75, 0.30),
}
PRICE_INPUT, PRICE_OUTPUT, PRICE_CACHE_WRITE, PRICE_CACHE_READ = PRICES[MODEL]

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


def _usage_dict(usage: object, model: str = MODEL) -> dict:
    price_in, price_out, price_write, price_read = PRICES.get(model, PRICES[MODEL])
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cost = (
        input_tokens * price_in
        + output_tokens * price_out
        + cache_write * price_write
        + cache_read * price_read
    ) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_write_tokens": cache_write,
        "cache_read_tokens": cache_read,
        "cost_usd": round(cost, 6),
    }


def _request_kwargs(model: str, system: list, messages: list) -> dict:
    """Per-model request shape.

    Sonnet 5 rejects any non-default `temperature` with a 400, and runs adaptive
    thinking unless told otherwise — which would eat the max_tokens budget the
    answer needs. Neither applies to the primary model.
    """
    kwargs = {"model": model, "max_tokens": MAX_TOKENS, "system": system, "messages": messages}
    if model == FALLBACK_MODEL:
        kwargs["thinking"] = {"type": "disabled"}
    else:
        kwargs["temperature"] = 0
    return kwargs


def _add_usage(first: dict, second: dict) -> dict:
    """Sum two attempts. A refused attempt still bills its input tokens, so dropping
    it would under-report what the question actually cost."""
    merged = {k: first.get(k, 0) + second.get(k, 0) for k in ("input_tokens", "output_tokens")}
    merged["cache_write_tokens"] = first.get("cache_write_tokens", 0) + second.get(
        "cache_write_tokens", 0
    )
    merged["cache_read_tokens"] = first.get("cache_read_tokens", 0) + second.get(
        "cache_read_tokens", 0
    )
    merged["cost_usd"] = round(first.get("cost_usd", 0.0) + second.get("cost_usd", 0.0), 6)
    return merged


def _refusal_answer(response, usage: dict, tried: list[str]) -> Answer:
    """Every model tried declined the request.

    The response is a normal HTTP 200 with `stop_reason: "refusal"`. Content is
    usually empty, but on a mid-stream refusal it holds a partial answer — which is
    discarded, because a truncated answer presented as complete is worse than none.
    """
    details = getattr(response, "stop_details", None)
    category = getattr(details, "category", None) or "unspecified"
    return Answer(
        text=(
            f"{API_REFUSAL} — Anthropic's safety classifier declined it "
            f"(category: {category}) on {' and then '.join(tried)}. The question was "
            f"not answered at all, so there is nothing to cite. This is a known false "
            f"positive on papers about adversarial and safety-critical topics; asking "
            f"about a narrower part of the paper, or using library search instead of "
            f"whole-paper mode, usually gets through."
        ),
        citations=[],
        unverified=[],
        usage={**usage, "api_refusal": True, "refusal_category": category, "models_tried": tried},
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

    response = client.messages.create(**_request_kwargs(MODEL, system, messages))
    answered_by = MODEL
    usage = _usage_dict(response.usage, MODEL)

    # Check stop_reason BEFORE reading content: on a refusal the content list is
    # usually empty, and joining it silently yields "" rather than raising.
    if getattr(response, "stop_reason", None) == "refusal":
        tried = [MODEL]
        try:
            retry = client.messages.create(**_request_kwargs(FALLBACK_MODEL, system, messages))
        except Exception:  # noqa: BLE001 — a failed retry reports the original refusal
            return _refusal_answer(response, usage, tried)

        tried.append(FALLBACK_MODEL)
        usage = _add_usage(usage, _usage_dict(retry.usage, FALLBACK_MODEL))
        if getattr(retry, "stop_reason", None) == "refusal":
            return _refusal_answer(retry, usage, tried)
        response, answered_by = retry, FALLBACK_MODEL

    raw = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    text, citations, unverified = _postprocess(raw, paper)
    return Answer(
        text=text,
        citations=citations,
        unverified=unverified,
        usage={
            **usage,
            "model": answered_by,
            "used_fallback": answered_by != MODEL,
        },
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
