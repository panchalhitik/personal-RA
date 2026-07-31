"""Retrieval evaluation harness.

Cheap metrics (recall@k, MRR) run offline in seconds with no API calls, so
retrieval configs can be iterated on freely. Refusal rate on the unanswerable
subset is reported separately — a confident wrong answer there is worse than
no answer. RAGAS faithfulness/relevancy costs money and is gated behind --full.

Config matrix: 3 chunking strategies x 3 retrieval modes (spec §4 Step 1.3).
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

GOLDEN_SET = Path("eval") / "golden_set.yaml"
RESULTS_DIR = Path("eval") / "results"
CHUNKING_STRATEGIES = ("fixed", "section", "section_context")
RETRIEVAL_MODES = ("dense", "bm25", "hybrid")
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

    results = []
    for strategy, library in indexes.items():
        for mode in args.modes:

            def search_fn(question: str, k: int, _lib=library, _mode=mode) -> list[str]:
                hits = _lib.search(question, k=k, mode=_mode)
                return [h.metadata["paper_id"] for h in hits]

            outcome = evaluate_config(questions, search_fn, k=args.k)
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
