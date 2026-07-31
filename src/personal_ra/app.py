"""Streamlit UI: PDF reader with assistant panel. Run: streamlit run src/personal_ra/app.py

Layout: the PDF fills the main pane (selectable text, citation highlights);
the assistant/notes panel on the right pops in and out via a sidebar toggle.
The ask bar is always docked at the bottom. After an answer, the viewer jumps
to the first cited page and highlights the cited passages until the next
question replaces them.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from streamlit_pdf_viewer import pdf_viewer

from personal_ra.ask import Answer, Message, ask
from personal_ra.cite import Citation
from personal_ra.locate import locate_quote
from personal_ra.parse import parse_pdf
from personal_ra.vision import enrich_paper

PAPERS_DIR = Path("papers")
NOTES_DIR = Path("notes")
HIGHLIGHT_COLOR = "#ffb703"
VIEWER_HEIGHT = 780


def list_papers(papers_dir: Path = PAPERS_DIR) -> list[Path]:
    if not papers_dir.exists():
        return []
    return sorted(papers_dir.glob("*.pdf"))


def notes_path(pdf: Path, notes_dir: Path = NOTES_DIR) -> Path:
    return notes_dir / f"{pdf.stem}.md"


def history_to_messages(chat: list[dict]) -> list[Message]:
    """Convert stored chat entries into the Message history ask() expects."""
    return [Message(role=e["role"], content=e["content"]) for e in chat]


def session_cost(chat: list[dict]) -> float:
    return sum(e["answer"].usage.get("cost_usd", 0.0) for e in chat if e.get("answer"))


def quote_preview(quote: str, max_len: int = 60) -> str:
    return quote if len(quote) <= max_len else quote[: max_len - 1].rstrip() + "…"


def _scroll_chat_to_bottom() -> None:
    """The fixed-height chat box resets to the top on every rerun; scroll it back
    down so the conversation reads bottom-anchored like a normal chat."""
    components.html(
        """<script>
        const scroll = () => {
            const doc = window.parent.document;
            const boxes = [...doc.querySelectorAll('div')].filter(d =>
                d.querySelector('[data-testid="stChatMessage"]') &&
                d.scrollHeight > d.clientHeight + 10 &&
                /auto|scroll/.test(getComputedStyle(d).overflowY));
            const box = boxes[boxes.length - 1];
            if (box) box.scrollTop = box.scrollHeight;
        };
        setTimeout(scroll, 50);
        setTimeout(scroll, 300);
        </script>""",
        height=0,
    )


def build_annotations(citations: list[Citation], pdf_path: Path) -> list[dict]:
    """Highlight boxes for the PDF viewer; quotes we can't locate are skipped."""
    annotations: list[dict] = []
    for c in citations:
        if c.page is None:
            continue
        for x0, y0, x1, y1 in locate_quote(pdf_path, c.page, c.quote):
            annotations.append(
                {
                    "page": c.page,
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                    "color": HIGHLIGHT_COLOR,
                }
            )
    return annotations


def render_answer(answer: Answer) -> None:
    st.markdown(answer.text)
    for c in answer.citations:
        with st.expander(f'p. {c.page} — "{quote_preview(c.quote)}" ({c.match_type})'):
            st.markdown(f"> {c.quote}")
    if answer.unverified:
        quotes = "\n\n".join(f'- "{c.quote}"' for c in answer.unverified)
        st.warning(
            f"**Unverified quotes** — these did not match the paper text and may be "
            f"invented or paraphrased:\n\n{quotes}"
        )


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Personal-RA", page_icon="📄", layout="wide")
    # A question was just answered on the previous run: open the assistant panel.
    # (Widget-keyed state can only be set before the toggle is instantiated.)
    if st.session_state.pop("_open_assistant", False):
        st.session_state.assistant_open = True
    st.session_state.setdefault("assistant_open", True)

    papers = list_papers()
    with st.sidebar:
        st.title("Personal-RA")
        if not papers:
            st.warning(f"No PDFs found in {PAPERS_DIR}/")
            st.stop()
        selected = st.selectbox("Paper", papers, format_func=lambda p: p.name)
        st.toggle("Assistant panel", key="assistant_open")

    key = str(selected)
    if st.session_state.get("paper_key") != key:
        with st.spinner("Parsing (and transcribing equations on first load)..."):
            paper = enrich_paper(parse_pdf(selected))
        st.session_state.paper_key = key
        st.session_state.paper = paper
        st.session_state.chat = []
        st.session_state.annotations = []
        st.session_state.scroll_page = None
        st.session_state.hl_version = 0
    paper = st.session_state.paper

    with st.sidebar:
        st.caption(paper.title)
        col1, col2 = st.columns(2)
        col1.metric("~Tokens", f"{paper.n_tokens:,}")
        col2.metric("Session cost", f"${session_cost(st.session_state.chat):.4f}")
        if st.button("Clear chat"):
            st.session_state.chat = []
            st.session_state.annotations = []
            st.session_state.scroll_page = None
            st.rerun()

    if st.session_state.assistant_open:
        pdf_col, side_col = st.columns([3, 2], gap="medium")
    else:
        pdf_col, side_col = st.container(), None

    with pdf_col:
        pdf_viewer(
            str(paper.path),
            width="100%",
            height=VIEWER_HEIGHT,
            annotations=st.session_state.annotations,
            render_text=True,
            scroll_to_page=st.session_state.scroll_page,
            key=f"pdf::{key}::{st.session_state.hl_version}",
        )

    if side_col is not None:
        with side_col:
            chat_tab, notes_tab = st.tabs(["Assistant", "Notes"])
            with chat_tab:
                # Fixed-height scrollable box: the chat scrolls on its own instead
                # of stretching the page and dragging the PDF along with it.
                with st.container(height=VIEWER_HEIGHT - 80):
                    for entry in st.session_state.chat:
                        with st.chat_message(entry["role"]):
                            if entry.get("answer") is not None:
                                render_answer(entry["answer"])
                            else:
                                st.markdown(entry["content"])
                if st.session_state.chat:
                    _scroll_chat_to_bottom()
            with notes_tab:
                nfile = notes_path(selected)
                existing = nfile.read_text(encoding="utf-8") if nfile.exists() else ""
                text = st.text_area(
                    "Notes (markdown, saved when you click away)",
                    value=existing,
                    height=520,
                    key=f"notes::{key}",
                )
                if text != existing:
                    nfile.parent.mkdir(parents=True, exist_ok=True)
                    nfile.write_text(text, encoding="utf-8")
                    st.caption("Saved.")
                st.download_button(
                    "Download notes",
                    data=text,
                    file_name=f"{selected.stem}.txt",
                    mime="text/plain",
                    disabled=not text,
                )

    if question := st.chat_input("Ask about this paper..."):
        history = history_to_messages(st.session_state.chat)
        st.session_state.chat.append({"role": "user", "content": question, "answer": None})
        with st.spinner("Thinking..."):
            answer = ask(paper, question, history=history)
        st.session_state.chat.append(
            {"role": "assistant", "content": answer.text, "answer": answer}
        )
        if answer.citations:
            st.session_state.annotations = build_annotations(answer.citations, paper.path)
            st.session_state.scroll_page = answer.citations[0].page
            st.session_state.hl_version += 1  # remount the viewer so it jumps
        else:
            st.session_state.annotations = []
            st.session_state.scroll_page = None
        st.session_state._open_assistant = True
        st.rerun()


if __name__ == "__main__":
    main()
