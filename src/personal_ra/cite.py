"""Verify that quotes actually appear in a paper, and on which page.

Both the quote and the source pages are normalized character-by-character
(whitespace collapsed, curly quotes / long dashes unified, NFKC + casefold),
while keeping a map from every normalized character back to its original
position — so a match in normalized space resolves to a real page and offset.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz

from personal_ra.parse import Paper

FUZZY_THRESHOLD = 95.0

_UNIFY = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "′": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "″": '"',
    "‒": "-",
    "–": "-",
    "—": "-",
    "−": "-",
    "…": "...",
}


@dataclass
class Citation:
    quote: str
    page: int | None
    char_offset: int | None  # offset into Page.text where the match starts
    verified: bool
    match_type: Literal["exact", "fuzzy", "failed"]


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalized text plus, for each normalized char, its source index."""
    chars: list[str] = []
    mapping: list[int] = []

    def emit(c: str, src: int) -> None:
        if c.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                mapping.append(src)
        else:
            chars.append(c)
            mapping.append(src)

    for i, raw in enumerate(text):
        for c in unicodedata.normalize("NFKC", _UNIFY.get(raw, raw)).casefold():
            emit(c, i)
    while chars and chars[-1] == " ":
        chars.pop()
        mapping.pop()
    return "".join(chars), mapping


def _build_index(paper: Paper) -> tuple[str, list[tuple[int, int]]]:
    """Concatenated normalized page texts and a per-char (page, offset) map.

    Pages are joined with a single space so quotes spanning a page boundary
    still match; the separator maps to the following page's first character.
    """
    chars: list[str] = []
    cmap: list[tuple[int, int]] = []
    for page in paper.pages:
        norm, m = _normalize_with_map(page.text)
        if not norm:
            continue
        if chars:
            chars.append(" ")
            cmap.append((page.number, m[0]))
        chars.extend(norm)
        cmap.extend((page.number, src) for src in m)
    return "".join(chars), cmap


def verify_quote(quote: str, paper: Paper) -> Citation:
    norm_quote, _ = _normalize_with_map(quote)
    if not norm_quote:
        return Citation(
            quote=quote, page=None, char_offset=None, verified=False, match_type="failed"
        )

    source, cmap = _build_index(paper)
    if not source:
        return Citation(
            quote=quote, page=None, char_offset=None, verified=False, match_type="failed"
        )

    idx = source.find(norm_quote)
    if idx != -1:
        page, offset = cmap[idx]
        return Citation(
            quote=quote, page=page, char_offset=offset, verified=True, match_type="exact"
        )

    alignment = fuzz.partial_ratio_alignment(norm_quote, source)
    if alignment is not None and alignment.score >= FUZZY_THRESHOLD:
        page, offset = cmap[min(alignment.dest_start, len(cmap) - 1)]
        return Citation(
            quote=quote, page=page, char_offset=offset, verified=True, match_type="fuzzy"
        )

    return Citation(quote=quote, page=None, char_offset=None, verified=False, match_type="failed")
