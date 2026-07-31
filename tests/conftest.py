from pathlib import Path

from personal_ra.parse import Page, Paper


def make_paper(page_texts: list[str]) -> Paper:
    """Build a synthetic Paper directly, no PDF involved."""
    pages = [Page(number=i + 1, text=t) for i, t in enumerate(page_texts)]
    full = "\n\n".join(f"[PAGE {p.number}]\n\n{p.text}" for p in pages)
    return Paper(
        path=Path("synthetic.pdf"),
        title="Synthetic",
        pages=pages,
        full_text=full,
        n_tokens=len(full) // 4,
    )
