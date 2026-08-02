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
from langgraph.types import Command
from streamlit_pdf_viewer import pdf_viewer

from personal_ra.ask import Message
from personal_ra.graph.build import build_graph, pending_approval, sqlite_checkpointer
from personal_ra.graph.state import initial_state
from personal_ra.library import paper_id
from personal_ra.locate import locate_quote
from personal_ra.parse import parse_pdf
from personal_ra.vision import enrich_paper

ROUTE_LABELS = {
    "single_paper": ("📄", "this paper"),
    "library": ("📚", "library"),
    "web": ("🌐", "web"),
    "direct": ("💬", "conversation"),
}
VERDICT_LABELS = {
    "grounded": ("✅", "every claim traced to the sources"),
    "partially_grounded": ("⚠️", "some claims not supported by the sources"),
    "ungrounded": ("⛔", "the answer rested on nothing — it was withdrawn"),
    "api_refused": ("🚫", "the API declined this request"),
    "not_checked": ("", ""),
}

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


def history_dicts(chat: list[dict]) -> list[dict]:
    """The shape the router reads for follow-ups like "what about the second one?"."""
    return [{"role": e["role"], "content": e["content"]} for e in chat]


def total_cost(usage: dict) -> float:
    """Every node's cost in one run — the graph spends across several models."""
    return sum(e.get("cost_usd", 0.0) for e in usage.values() if isinstance(e, dict))


def session_cost(chat: list[dict]) -> float:
    total = 0.0
    for entry in chat:
        result = entry.get("result")
        if isinstance(result, dict):
            total += total_cost(result.get("usage") or {})
        elif entry.get("answer") is not None:  # v0-shaped entry
            total += entry["answer"].usage.get("cost_usd", 0.0)
    return total


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


def citations_for_open_paper(citations: list, paper_title: str) -> list[dict]:
    """Only the citations that belong to the PDF on screen.

    A library answer cites several papers. Highlighting all of them in the open
    document would put boxes on pages that never contained those words — the exact
    kind of false provenance this project exists to avoid. Single-paper citations
    carry no paper_title and are by definition from the open paper.
    """
    out = []
    for c in citations:
        entry = c if isinstance(c, dict) else {"page": c.page, "quote": c.quote}
        title = entry.get("paper_title")
        if title is None or title == paper_title:
            out.append(entry)
    return out


def build_annotations(citations: list, pdf_path: Path) -> list[dict]:
    """Highlight boxes for the PDF viewer; quotes we can't locate are skipped."""
    annotations: list[dict] = []
    for c in citations:
        page = c["page"] if isinstance(c, dict) else c.page
        quote = c["quote"] if isinstance(c, dict) else c.quote
        if page is None:
            continue
        for x0, y0, x1, y1 in locate_quote(pdf_path, page, quote):
            annotations.append(
                {
                    "page": page,
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                    "color": HIGHLIGHT_COLOR,
                }
            )
    return annotations


def render_route(result: dict) -> None:
    """The route and why, above the answer — §3.11's first requirement."""
    icon, label = ROUTE_LABELS.get(result.get("route"), ("", result.get("route") or "?"))
    line = f"{icon} **{label}**"
    if result.get("rewrite_count"):
        line += f" · rewrote the search {result['rewrite_count']}x"
    if result.get("route_reason"):
        line += f" — {result['route_reason']}"
    st.caption(line)


def render_grounding(result: dict) -> None:
    grounding = result.get("grounding") or {}
    icon, label = VERDICT_LABELS.get(grounding.get("verdict"), ("", ""))
    if not label:
        return
    unsupported = grounding.get("unsupported") or []
    if unsupported:
        claims = "\n\n".join(f"- {u.get('claim', '')} — *{u.get('why', '')}*" for u in unsupported)
        st.warning(f"{icon} **{label}**\n\n{claims}")
    else:
        st.caption(f"{icon} {label}")


def render_answer(result: dict) -> None:
    render_route(result)
    st.markdown(result.get("answer") or "")
    for c in result.get("citations") or []:
        title = f" · {c['paper_title']}" if c.get("paper_title") else ""
        header = f'p. {c.get("page")}{title} — "{quote_preview(c.get("quote", ""))}"'
        with st.expander(header):
            st.markdown(f"> {c.get('quote', '')}")
    for w in result.get("web_results") or []:
        # Never a page number: a web snippet was not verified against anything held.
        st.caption(
            f"🌐 [{w.get('title', 'web result')}]({w.get('url', '')}) — external, unverified"
        )
    if result.get("unverified"):
        quotes = "\n\n".join(f'- "{c.get("quote", "")}"' for c in result["unverified"])
        st.warning(
            f"**Unverified quotes** — these did not match the source text and may be "
            f"invented or paraphrased:\n\n{quotes}"
        )
    render_grounding(result)


def paper_id_for(paper) -> str | None:
    """The open PDF's library id, so the router knows a paper is open.

    None when the file is not in the library yet — the router then treats it as
    "no paper open" and answers library-wide, which is the honest fallback.
    """
    try:
        return paper_id(paper.path)
    except OSError:
        return None


@st.cache_resource
def get_graph():
    """One graph for the session. Building it opens Chroma and builds BM25, which
    must not happen on every Streamlit rerun."""
    return build_graph(checkpointer=sqlite_checkpointer())


def run_graph(arg, config, status) -> None:
    """Drive the graph, ticking the status box as each node finishes.

    Streamed rather than invoked so the node sequence is visible while it happens —
    a spinner cannot show that a rewrite fired, and the rewrite is often the reason
    a slow answer was slow.
    """
    for update in get_graph().stream(arg, config, stream_mode="updates"):
        for node in update:
            if node.startswith("__"):
                continue
            status.update(label=f"{node}…")
            status.write(f"`{node}`")


def summarise(values: dict) -> dict:
    """The parts of State the UI renders, copied out of the checkpoint."""
    return {
        key: values.get(key)
        for key in (
            "route",
            "route_reason",
            "answer",
            "citations",
            "unverified",
            "grounding",
            "rewrite_count",
            "rewrite_reason",
            "web_results",
            "usage",
        )
    }


def render_approval(pending: dict, thread_id: str, config) -> None:
    """The interrupt, inline: what it wants to search, why, and what it costs."""
    st.info(
        "**Search the web?**\n\n"
        f"Query: `{pending.get('query', '')}`\n\n"
        f"Why: {pending.get('reason', '')}\n\n"
        f"Cost: {pending.get('estimated_cost', 'unknown')}"
    )
    approve, deny = st.columns(2)
    if approve.button("Search the web", key=f"approve::{thread_id}", type="primary"):
        _resume(True, thread_id, config)
    if deny.button("Answer from my library only", key=f"deny::{thread_id}"):
        _resume(False, thread_id, config)


def _resume(approved: bool, thread_id: str, config) -> None:
    with st.status("Resuming…", expanded=True) as status:
        run_graph(Command(resume=approved), config, status)
        status.update(label="Done", state="complete", expanded=False)
    _store_result(config)
    st.session_state.pending_approval = None
    st.rerun()


def _store_result(config) -> None:
    result = summarise(get_graph().get_state(config).values)
    st.session_state.chat.append(
        {"role": "assistant", "content": result.get("answer") or "", "result": result}
    )
    paper = st.session_state.paper
    mine = citations_for_open_paper(result.get("citations") or [], paper.title)
    if mine:
        st.session_state.annotations = build_annotations(mine, paper.path)
        st.session_state.scroll_page = mine[0]["page"]
        st.session_state.hl_version += 1  # remount the viewer so it jumps
    else:
        st.session_state.annotations = []
        st.session_state.scroll_page = None
    st.session_state._open_assistant = True


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Personal-RA", page_icon="📄", layout="wide")
    # A question was just answered on the previous run: open the assistant panel.
    # (Widget-keyed state can only be set before the toggle is instantiated.)
    if st.session_state.pop("_open_assistant", False):
        st.session_state.assistant_open = True
    st.session_state.setdefault("assistant_open", True)
    st.session_state.setdefault("pending_approval", None)

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
        st.session_state.pending_approval = None
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
                            if entry.get("result") is not None:
                                render_answer(entry["result"])
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

    # A run halted at the approval gate renders its approve/deny inline, in the
    # assistant panel, rather than blocking the whole page.
    if side_col is not None and st.session_state.get("pending_approval"):
        with side_col:
            waiting = st.session_state.pending_approval
            render_approval(waiting["request"], waiting["thread_id"], waiting["config"])

    if question := st.chat_input("Ask about this paper..."):
        thread_id = f"{key}::{len(st.session_state.chat)}"
        config = {"configurable": {"thread_id": thread_id}}
        state = initial_state(
            question,
            paper_id=paper_id_for(paper),
            thread_id=thread_id,
            history=history_dicts(st.session_state.chat),
        )
        st.session_state.chat.append({"role": "user", "content": question, "result": None})

        with st.status("Routing…", expanded=True) as status:
            run_graph(state, config, status)
            status.update(label="Done", state="complete", expanded=False)

        waiting = pending_approval(get_graph(), config)
        if waiting is not None:
            st.session_state.pending_approval = {
                "request": waiting,
                "thread_id": thread_id,
                "config": config,
            }
            st.session_state._open_assistant = True
        else:
            _store_result(config)
        st.rerun()


if __name__ == "__main__":
    main()
