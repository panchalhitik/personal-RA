"""Retrieval evaluation harness.

Cheap metrics (recall@k, MRR) run offline in seconds with no API calls, so
retrieval configs can be iterated on freely. Refusal rate on the unanswerable
subset is reported separately — a confident wrong answer there is worse than
no answer. RAGAS faithfulness/relevancy costs money and is gated behind --full.

Config matrix: 3 chunking strategies x 4 retrieval modes. The fourth mode,
"rerank", was added in v3 Step 3.3 so cross-encoder reranking is measured on the
same footing as the v1 modes rather than asserted to be better.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

GOLDEN_SET = Path("eval") / "golden_set.yaml"
RESULTS_DIR = Path("eval") / "results"
CHUNKING_STRATEGIES = ("fixed", "section", "section_context")
RETRIEVAL_MODES = ("dense", "bm25", "hybrid", "rerank")
CATEGORIES = ("factual", "cross_paper", "comparison", "unanswerable")


@dataclass
class Question:
    id: str
    question: str
    category: str
    expected_paper_ids: list[str] = field(default_factory=list)
    ground_truth: str = ""

    @property
    def is_unanswerable(self) -> bool:
        return self.category == "unanswerable"


def load_golden_set(path: Path = GOLDEN_SET) -> list[Question]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    questions = [
        Question(
            id=item["id"],
            question=item["question"],
            category=item.get("category", "factual"),
            expected_paper_ids=list(item.get("expected_paper_ids") or []),
            ground_truth=item.get("ground_truth", "") or "",
        )
        for item in raw
    ]
    validate_golden_set(questions)
    return questions


def validate_golden_set(questions: list[Question]) -> None:
    """Fail loudly on a malformed set — a silently broken golden set produces
    metrics that look fine and mean nothing."""
    ids = [q.id for q in questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate question ids: {sorted(duplicates)}")
    for q in questions:
        if q.category not in CATEGORIES:
            raise ValueError(f"{q.id}: unknown category {q.category!r} (expected {CATEGORIES})")
        if q.is_unanswerable and q.expected_paper_ids:
            raise ValueError(f"{q.id}: unanswerable questions must have no expected_paper_ids")
        if not q.is_unanswerable and not q.expected_paper_ids:
            raise ValueError(f"{q.id}: answerable questions need at least one expected_paper_id")


def recall_at_k(retrieved_paper_ids: list[str], expected: list[str], k: int) -> float:
    """Fraction of expected papers that appear in the top-k retrieved papers."""
    if not expected:
        return 0.0
    top_k = set(retrieved_paper_ids[:k])
    return sum(1 for pid in set(expected) if pid in top_k) / len(set(expected))


def reciprocal_rank(retrieved_paper_ids: list[str], expected: list[str]) -> float:
    """1/rank of the first expected paper; 0 if none appear."""
    expected_set = set(expected)
    for rank, pid in enumerate(retrieved_paper_ids, start=1):
        if pid in expected_set:
            return 1.0 / rank
    return 0.0


def dedup_preserving_order(items: list[str]) -> list[str]:
    """Chunk-level hits -> paper-level ranking (first appearance wins)."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def aggregate(per_question: list[dict], k_values: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """Mean metrics over answerable questions; refusal rate over unanswerable ones."""
    answerable = [r for r in per_question if not r["is_unanswerable"]]
    unanswerable = [r for r in per_question if r["is_unanswerable"]]

    metrics: dict = {
        "n_questions": len(per_question),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
    }
    for k in k_values:
        key = f"recall@{k}"
        metrics[key] = round(statistics.mean(r[key] for r in answerable), 4) if answerable else 0.0
    metrics["mrr"] = round(statistics.mean(r["rr"] for r in answerable), 4) if answerable else 0.0

    refusals = [r for r in unanswerable if r.get("refused") is not None]
    metrics["refusal_rate"] = (
        round(sum(1 for r in refusals if r["refused"]) / len(refusals), 4) if refusals else None
    )
    return metrics


def evaluate_config(
    questions: list[Question],
    search_fn,
    k: int = 10,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Run retrieval for every question. search_fn(question, k) -> list of paper_ids
    in rank order (chunk-level, duplicates allowed)."""
    per_question = []
    for q in questions:
        papers = dedup_preserving_order(search_fn(q.question, k))
        row = {
            "id": q.id,
            "category": q.category,
            "is_unanswerable": q.is_unanswerable,
            "retrieved_papers": papers[:k],
            "expected": q.expected_paper_ids,
            "rr": reciprocal_rank(papers, q.expected_paper_ids),
        }
        for kv in k_values:
            row[f"recall@{kv}"] = recall_at_k(papers, q.expected_paper_ids, kv)
        per_question.append(row)
    return {"metrics": aggregate(per_question, k_values), "per_question": per_question}


def build_search_fn(library, mode: str, rerank_model=None):
    """(question, k) -> paper_ids in rank order, for one retrieval mode.

    "rerank" is the odd one out: it retrieves deep with hybrid and lets the
    cross-encoder pick the top k, so its k means something different from the
    other three modes' k. That is the point of the comparison.
    """
    if mode == "rerank":
        from personal_ra.graph.rerank import retrieve_and_rerank

        def search_fn(question: str, k: int) -> list[str]:
            hits = retrieve_and_rerank(library, question, top_k=k, model=rerank_model)
            return [h.metadata["paper_id"] for h in hits]
    else:

        def search_fn(question: str, k: int) -> list[str]:
            hits = library.search(question, k=k, mode=mode)
            return [h.metadata["paper_id"] for h in hits]

    return search_fn


def render_markdown_table(results: list[dict]) -> str:
    """Config-matrix results as a markdown table for the README."""
    header = (
        "| Chunking | Retrieval | recall@1 | recall@3 | recall@5 | recall@10 | MRR |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    best = max((r["metrics"]["recall@5"] for r in results), default=0.0)
    for r in sorted(results, key=lambda r: r["metrics"]["recall@5"], reverse=True):
        m = r["metrics"]
        is_best = best > 0 and m["recall@5"] == best
        label = f"**{r['chunking']}**" if is_best else r["chunking"]
        rows.append(
            f"| {label} | {r['retrieval']} | "
            f"{m['recall@1']:.3f} | {m['recall@3']:.3f} | {m['recall@5']:.3f} | "
            f"{m['recall@10']:.3f} | {m['mrr']:.3f} |"
        )
    return header + "\n".join(rows)


def write_results(payload: dict, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_route_eval(golden_set: Path, routes_file: Path) -> dict:
    """Re-run the router over every labelled question in both paper states.

    The router is re-run rather than read from the checkpoint-3.2 draft, so this is
    an actual harness rather than a replay. Temperature is 0, so a rerun should
    reproduce; where it does not, that instability is itself worth seeing.
    """
    import anthropic

    from personal_ra.graph.router import classify_route
    from personal_ra.graph.state import initial_state
    from personal_ra.route_eval import load_route_labels

    questions = {q.id: q for q in load_golden_set(golden_set)}
    labels = load_route_labels(routes_file)
    client = anthropic.Anthropic()

    rows = []
    for qid, label in labels.items():
        question = questions.get(qid)
        if question is None:
            continue
        for arm in ("with_paper", "no_paper"):
            paper_id = label["paper_open"] if arm == "with_paper" else None
            started = time.perf_counter()
            route, reason, usage = classify_route(
                initial_state(question.question, paper_id), client=client
            )
            rows.append(
                {
                    "id": qid,
                    "arm": arm,
                    "question": question.question,
                    "category": question.category,
                    "is_unanswerable": question.is_unanswerable,
                    "expected": label[arm],
                    "predicted": route,
                    "reason": reason,
                    "route": route,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "cost_usd": usage.get("cost_usd", 0.0),
                }
            )
    return {"rows": rows}


def render_route_report(rows: list[dict]) -> str:
    from personal_ra import route_eval as re_

    out: list[str] = []
    overall = route_eval_pairs = [(r["expected"], r["predicted"]) for r in rows]
    out.append(
        f"## Route accuracy\n\n**{re_.route_accuracy(overall):.1%}** over {len(rows)} "
        f"classifications ({len(rows) // 2} questions x 2 paper states)\n"
    )

    for arm in ("with_paper", "no_paper"):
        subset = [r for r in rows if r["arm"] == arm]
        pairs = [(r["expected"], r["predicted"]) for r in subset]
        out.append(f"- `{arm}`: {re_.route_accuracy(pairs):.1%} ({len(subset)} questions)")
    out.append("")

    matrix = re_.confusion_matrix(route_eval_pairs)
    out.append("### Confusion matrix\n")
    out.append(re_.render_confusion_matrix(matrix))
    out.append("\n### Per route\n")
    out.append(re_.render_per_route(re_.per_route_scores(matrix)))

    wrong = re_.misroutes(rows)
    out.append(f"\n### Misroutes ({len(wrong)})\n")
    if wrong:
        for r in wrong:
            out.append(
                f"- **{r['id']}** ({r['arm']}): expected `{r['expected']}`, "
                f"got `{r['predicted']}` — {r['reason']}"
            )
    else:
        out.append("None.")

    cl = re_.cost_latency_by_route(rows)
    out.append("\n### Router cost and latency\n")
    out.append("| route | n | p50 | p95 | median cost |")
    out.append("|---|---|---|---|---|")
    for route, stats in cl.items():
        lat = stats["latency_ms"]
        out.append(
            f"| {route} | {stats['n']} | {lat['p50']}ms | {lat['p95']}ms | "
            f"${stats['median_cost_usd']:.6f} |"
        )
    out.append(f"\nTotal spend for this run: ${cl['all']['total_cost_usd']:.4f}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval configurations.")
    ap.add_argument("--golden-set", type=Path, default=GOLDEN_SET)
    ap.add_argument("--db", type=Path, default=None, help="Chroma path (default: chroma_db)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=list(RETRIEVAL_MODES),
        default=list(RETRIEVAL_MODES),
    )
    ap.add_argument("--full", action="store_true", help="also run paid RAGAS metrics")
    ap.add_argument(
        "--routes",
        action="store_true",
        help="evaluate routing against eval/routes.yaml instead of retrieval "
        "(re-runs the router over 63 questions x 2 states; costs cents)",
    )
    ap.add_argument("--routes-file", type=Path, default=Path("eval") / "routes.yaml")
    ap.add_argument(
        "--matrix",
        action="store_true",
        help="run all 3 chunking strategies x retrieval modes (builds eval indexes "
        "under eval/indexes/ on first run — several minutes of local embedding)",
    )
    args = ap.parse_args(argv)

    if not args.golden_set.exists():
        raise SystemExit(
            f"No golden set at {args.golden_set}. Run "
            f"`python -m personal_ra.eval_prep --template` to start one."
        )
    questions = load_golden_set(args.golden_set)
    print(
        f"Loaded {len(questions)} questions "
        f"({sum(1 for q in questions if q.is_unanswerable)} unanswerable)"
    )

    if args.routes:
        load_dotenv()
        if not args.routes_file.exists():
            raise SystemExit(f"No route labels at {args.routes_file}.")
        outcome = run_route_eval(args.golden_set, args.routes_file)
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "golden_set": str(args.golden_set),
            "routes_file": str(args.routes_file),
            "rows": outcome["rows"],
        }
        path = write_results(payload, RESULTS_DIR / "routes")
        print(f"\nWrote {path}\n")
        print(render_route_report(outcome["rows"]))
        return

    from personal_ra.library import DB_PATH, ingest
    from personal_ra.search import Library

    if args.matrix:
        indexes = {}
        for strategy in CHUNKING_STRATEGIES:
            db = Path("eval") / "indexes" / strategy
            library = Library(db_path=db)
            if not library.collection.count():
                print(f"Building {strategy} index (one-time, local embedding)...")
                ingest(db_path=db, strategy=strategy)
                library = Library(db_path=db)
            indexes[strategy] = library
    else:
        indexes = {"section_context": Library(db_path=args.db or DB_PATH)}

    rerank_model = None
    if "rerank" in args.modes:
        from personal_ra.graph.rerank import load_model

        print("Loading cross-encoder (one-time download on first run)...")
        rerank_model = load_model()

    results = []
    for strategy, library in indexes.items():
        for mode in args.modes:
            outcome = evaluate_config(
                questions, build_search_fn(library, mode, rerank_model), k=args.k
            )
            results.append({"chunking": strategy, "retrieval": mode, **outcome})
            m = outcome["metrics"]
            print(f"  {strategy:16}/{mode:7} recall@5={m['recall@5']:.3f}  MRR={m['mrr']:.3f}")

    if args.full:
        print("RAGAS metrics are not wired up yet (Step 1.3 follow-up).")

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "golden_set": str(args.golden_set),
        "k": args.k,
        "configs": results,
    }
    path = write_results(payload)
    print(f"\nWrote {path}\n")
    print(render_markdown_table(results))


if __name__ == "__main__":
    main()
