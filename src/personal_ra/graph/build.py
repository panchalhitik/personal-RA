"""Graph assembly and checkpointer wiring.

The shape is the diagram in the v3 spec:

    route ─┬─ single_paper ──────────────────────────────► grounding ─► END
           ├─ retrieve → rerank → grade ─┬─ rewrite ──┐
           │                             │      ▲     │ (loop to retrieve)
           │                             │      └─────┘
           │                             └─ generate ──► grounding ─► END
           ├─ approve → web_search ──────► generate
           └─ direct ────────────────────────────────────────────────► END

Deviation from the spec diagram, agreed with Hitik: the diagram sends
single_paper straight to END. It goes through grounding instead, so §3.6's
refusal-verdict comparison can be recomputed on the single-paper path — which is
where the README's 5/6 prefix-matched refusal number came from.
"""

from __future__ import annotations

import sqlite3
from functools import partial
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from personal_ra.graph import nodes
from personal_ra.graph.router import after_route
from personal_ra.graph.state import State

DB_PATH = Path(".cache/graph.db")

NODE_NAMES = [
    "route",
    "single_paper",
    "retrieve",
    "rerank",
    "grade",
    "rewrite",
    "approve",
    "web_search",
    "generate",
    "grounding",
    "direct",
]


def sqlite_checkpointer(db_path: Path | str = DB_PATH) -> SqliteSaver:
    """A long-lived checkpointer. Threads are keyed by thread_id in the run config.

    `check_same_thread=False` because Streamlit and FastAPI both touch the graph
    from worker threads.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def build_graph(checkpointer: SqliteSaver | None = None, client=None):
    """Assemble and compile the graph.

    `checkpointer` persists threads; `client` injects an Anthropic client into the
    nodes that call the API (tests pass a mock, so the suite never hits the network).
    """
    g = StateGraph(State)

    # partial, not a lambda: LangGraph inspects the signature to decide whether to
    # pass a RunnableConfig as the second argument, and `client` is not that.
    g.add_node("route", partial(nodes.route_node, client=client))
    g.add_node("single_paper", nodes.single_paper_node)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("rerank", nodes.rerank_node)
    g.add_node("grade", nodes.grade_node)
    g.add_node("rewrite", nodes.rewrite_node)
    g.add_node("approve", nodes.approve_node)
    g.add_node("web_search", nodes.web_search_node)
    g.add_node("generate", nodes.generate_node)
    g.add_node("grounding", nodes.grounding_node)
    g.add_node("direct", nodes.direct_node)

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        after_route,
        {
            "single_paper": "single_paper",
            "library": "retrieve",
            "web": "approve",
            "direct": "direct",
        },
    )

    # library branch
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "grade")
    g.add_conditional_edges(
        "grade",
        nodes.after_grade,
        {"rewrite": "rewrite", "generate": "generate"},
    )
    g.add_edge("rewrite", "retrieve")

    # web branch — approval gate, then the same generator
    g.add_edge("approve", "web_search")
    g.add_edge("web_search", "generate")

    g.add_edge("generate", "grounding")

    # terminals
    g.add_edge("single_paper", "grounding")
    g.add_edge("grounding", END)
    g.add_edge("direct", END)

    return g.compile(checkpointer=checkpointer)
