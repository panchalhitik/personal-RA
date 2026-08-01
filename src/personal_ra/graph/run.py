"""CLI entry point: run one question through the graph, printing node transitions.

python -m personal_ra.graph.run "what is sandbagging?" --route library
"""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv

from personal_ra.graph.build import DB_PATH, build_graph, sqlite_checkpointer
from personal_ra.graph.state import initial_state


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run a question through the v3 router graph.")
    ap.add_argument("question")
    ap.add_argument("--paper-id", default=None)
    ap.add_argument("--thread-id", default=None, help="reuse to continue a conversation")
    ap.add_argument(
        "--route",
        choices=["single_paper", "library", "web", "direct"],
        default=None,
        help="force a route instead of classifying (the classifier lands in step 3.2)",
    )
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args(argv)

    load_dotenv()
    thread_id = args.thread_id or uuid.uuid4().hex[:12]
    graph = build_graph(checkpointer=sqlite_checkpointer(args.db))
    state = initial_state(args.question, args.paper_id, thread_id, args.route)
    config = {"configurable": {"thread_id": thread_id}}

    print(f"thread {thread_id}\n")
    for update in graph.stream(state, config, stream_mode="updates"):
        for node, delta in update.items():
            print(f"  -> {node}", f"({delta['route_reason']})" if "route_reason" in delta else "")

    final = graph.get_state(config).values
    print(f"\nroute:    {final['route']} - {final['route_reason']}")
    print(f"rewrites: {final['rewrite_count']}")
    print(f"answer:   {final['answer'] or '(stub - nodes land in later steps)'}")


if __name__ == "__main__":
    main()
