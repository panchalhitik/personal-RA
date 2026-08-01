"""Grounding check — does the answer actually rest on the retrieved material?

Two passes, because answers fail in two different ways:

1. **Quotes.** cite.py already string-matches every <quote> back to its source, so
   a fabricated quote is caught exactly. That work happens inside ask.py/search.py
   during generation; this module reads its verdict rather than paying for it twice.
2. **Everything else.** The prose *between* the quotes is where invention actually
   hides — a correctly quoted number wrapped in a claim the paper never made. Haiku
   with structured output judges each such claim against the retrieved context.

This replaces the string-prefix refusal detection the v2 README flags as a
weakness. A hedge that invents nothing is a correct refusal even when it does not
use the exact refusal sentence, and `refused_by_prefix` vs `refused_by_grounding`
below is the comparison that shows the difference.
"""

from __future__ import annotations

import anthropic

from personal_ra.ask import REFUSAL as PAPER_REFUSAL
from personal_ra.search import LIBRARY_REFUSAL

CHECKER_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024

# $ per million tokens, claude-haiku-4-5.
PRICE_INPUT = 1.00
PRICE_OUTPUT = 5.00

MAX_ATTEMPTS = 2  # one generation, then at most one stricter regeneration

GROUNDED = "grounded"
PARTIALLY_GROUNDED = "partially_grounded"
UNGROUNDED = "ungrounded"
NOT_CHECKED = "not_checked"

CHECKER_SYSTEM = """You audit an answer written about research papers, against the \
excerpts it was allowed to use. You are a checker, not an answerer — never supply \
missing information yourself.

Material already inside quotation marks followed by a page or paper marker has been \
verified character-for-character against the source. Skip it. Your job is the prose \
around those quotes: the claims the writer made in their own words.

For each such claim, decide whether the excerpts support it. Report only the ones \
they do NOT support, including:
- facts, numbers, dates, or names that appear nowhere in the excerpts
- causal or comparative claims the excerpts do not make ("X outperforms Y", \
"because of Z") when they only state X and Y separately
- confident generalisations drawn from a single reported result

Do not report:
- ordinary connective and framing language ("the paper argues", "in short")
- restatements of quoted material in different words
- explicit statements of uncertainty or absence ("the excerpts do not say")

Set makes_substantive_claims to false when the answer asserts nothing about the \
papers at all — a refusal, a hedge, a request for clarification, or a statement that \
the material does not cover the question. This is judged independently of wording: \
an answer that declines without using any fixed refusal phrase still counts."""

CHECK_TOOL = {
    "name": "report_grounding",
    "description": "Report which claims in the answer the excerpts do not support.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "makes_substantive_claims": {
                "type": "boolean",
                "description": (
                    "False when the answer asserts nothing about the papers — a "
                    "refusal, hedge, or request for clarification, however worded."
                ),
            },
            "unsupported_claims": {
                "type": "array",
                "description": "Empty when every claim is supported by the excerpts.",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "The unsupported claim, quoted from the answer.",
                        },
                        "why": {
                            "type": "string",
                            "description": "At most fifteen words on what the excerpts lack.",
                        },
                    },
                    "required": ["claim", "why"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["makes_substantive_claims", "unsupported_claims"],
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


def build_context(chunks: list[dict], web_results: list[dict] | None = None) -> str:
    """The material the answer was allowed to use, as the checker sees it.

    Web results are labelled as external so the checker does not treat a web
    snippet as though it were library evidence.
    """
    parts = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        parts.append(
            f'<excerpt paper="{meta.get("paper_title", "unknown")}" '
            f'page="{meta.get("page", "?")}">\n{chunk.get("text", "")}\n</excerpt>'
        )
    for result in web_results or []:
        parts.append(
            f'<web_result source="{result.get("url", "unknown")}">\n'
            f"{result.get('content', '')}\n</web_result>"
        )
    return "\n\n".join(parts)


def refused_by_prefix(answer: str) -> bool:
    """The v2 method: does the answer start with one of the fixed refusal strings?

    This is the baseline the README calls a weakness — it scores a well-judged hedge
    as a failure purely because the wording differs.
    """
    stripped = (answer or "").strip()
    return stripped.startswith(PAPER_REFUSAL) or stripped.startswith(LIBRARY_REFUSAL)


def refused_by_grounding(grounding: dict) -> bool:
    """The v3 method: did the answer decline without inventing anything?

    An answer counts as a correct refusal when it asserts nothing substantive about
    the papers and leaves no unsupported claim behind — regardless of its wording.
    """
    return not grounding.get("makes_substantive_claims", True) and not grounding.get(
        "unsupported", []
    )


def verdict_for(
    unsupported: list[dict], unverified_quotes: int, verified_quotes: int, substantive: bool
) -> str:
    """grounded / partially_grounded / ungrounded.

    `ungrounded` is reserved for an answer resting on nothing at all: it asserts
    things, and none of them are backed by a verified quote. An answer with real
    citations plus one overreaching sentence is `partially_grounded` — the honest
    label, and the one that keeps regeneration for the cases that need it.
    """
    if not unsupported and not unverified_quotes:
        return GROUNDED
    if substantive and verified_quotes == 0:
        return UNGROUNDED
    return PARTIALLY_GROUNDED


def check_grounding(
    answer: str,
    context: str,
    citations: list[dict] | None = None,
    unverified: list[dict] | None = None,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """Audit `answer` against `context`. Returns the grounding dict for State.

    `citations` and `unverified` come from cite.py, already run during generation —
    pass 1 is reading its result, not repeating it.
    """
    citations = citations or []
    unverified = unverified or []

    if not (answer or "").strip():
        return {
            "verdict": NOT_CHECKED,
            "unsupported": [],
            "verified_quotes": 0,
            "unverified_quotes": 0,
            "makes_substantive_claims": False,
            "usage": {},
        }

    client = client or anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=CHECKER_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system=CHECKER_SYSTEM,
            tools=[CHECK_TOOL],
            tool_choice={"type": "tool", "name": "report_grounding"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"<excerpts>\n{context or '(no excerpts were retrieved)'}\n</excerpts>\n\n"
                        f"<answer>\n{answer}\n</answer>"
                    ),
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 — a failed audit must not fail the run
        return {
            "verdict": NOT_CHECKED,
            "unsupported": [],
            "verified_quotes": len(citations),
            "unverified_quotes": len(unverified),
            "makes_substantive_claims": True,
            "error": f"{type(exc).__name__}",
            "usage": {},
        }

    unsupported: list[dict] = []
    substantive = True
    for block in response.content:
        if getattr(block, "type", "") == "tool_use" and block.name == "report_grounding":
            unsupported = list(block.input.get("unsupported_claims") or [])
            substantive = bool(block.input.get("makes_substantive_claims", True))
            break

    return {
        "verdict": verdict_for(unsupported, len(unverified), len(citations), substantive),
        "unsupported": unsupported,
        "verified_quotes": len(citations),
        "unverified_quotes": len(unverified),
        "makes_substantive_claims": substantive,
        "usage": _usage(response),
    }


def refusal_for_route(route: str | None) -> str:
    """The refusal to fall back on when an answer stays ungrounded after a retry."""
    return PAPER_REFUSAL if route == "single_paper" else LIBRARY_REFUSAL
