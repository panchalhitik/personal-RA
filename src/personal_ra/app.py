"""Streamlit UI for single-paper Q&A. Run with: streamlit run src/personal_ra/app.py"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from personal_ra.ask import Answer, Message, ask
from personal_ra.parse import parse_pdf
from personal_ra.vision import enrich_paper

PAPERS_DIR = Path("papers")


def list_papers(papers_dir: Path = PAPERS_DIR) -> list[Path]:
    if not papers_dir.exists():
        return []
    return sorted(papers_dir.glob("*.pdf"))


def history_to_messages(chat: list[dict]) -> list[Message]:
    """Convert stored chat entries into the Message history ask() expects."""
    return [Message(role=e["role"], content=e["content"]) for e in chat]


def session_cost(chat: list[dict]) -> float:
    return sum(e["answer"].usage.get("cost_usd", 0.0) for e in chat if e.get("answer"))


def quote_preview(quote: str, max_len: int = 60) -> str:
    return quote if len(quote) <= max_len else quote[: max_len - 1].rstrip() + "…"


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
    st.set_page_config(page_title="Personal-RA", page_icon="📄")

    papers = list_papers()
    with st.sidebar:
        st.title("Personal-RA")
        if not papers:
            st.warning(f"No PDFs found in {PAPERS_DIR}/")
            st.stop()
        selected = st.selectbox("Paper", papers, format_func=lambda p: p.name)

    key = str(selected)
    if st.session_state.get("paper_key") != key:
        with st.spinner("Parsing (and transcribing equations on first load)..."):
            paper = enrich_paper(parse_pdf(selected))
        st.session_state.paper_key = key
        st.session_state.paper = paper
        st.session_state.chat = []
    paper = st.session_state.paper

    with st.sidebar:
        st.caption(paper.title)
        col1, col2 = st.columns(2)
        col1.metric("~Tokens", f"{paper.n_tokens:,}")
        col2.metric("Session cost", f"${session_cost(st.session_state.chat):.4f}")
        if st.button("Clear chat"):
            st.session_state.chat = []
            st.rerun()

    for entry in st.session_state.chat:
        with st.chat_message(entry["role"]):
            if entry.get("answer"):
                render_answer(entry["answer"])
            else:
                st.markdown(entry["content"])

    if question := st.chat_input("Ask about this paper..."):
        history = history_to_messages(st.session_state.chat)
        st.session_state.chat.append({"role": "user", "content": question, "answer": None})
        with st.spinner("Thinking..."):
            answer = ask(paper, question, history=history)
        st.session_state.chat.append(
            {"role": "assistant", "content": answer.text, "answer": answer}
        )
        st.rerun()


if __name__ == "__main__":
    main()
