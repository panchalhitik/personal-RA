"""Route-level evaluation metrics for the v3 graph.

Separate from eval.py's retrieval metrics because they answer a different question:
eval.py asks "did retrieval find the right papers", this asks "did the graph make
the right decisions, and what did they cost". `personal_ra.eval --routes` drives it.

Everything here is a pure function over rows, so the metrics are testable against
hand-computed cases without a library, a graph, or an API key.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import yaml

ROUTES_FILE = Path("eval") / "routes.yaml"
ROUTES = ("single_paper", "library", "web", "direct")
ARMS = ("with_paper", "no_paper")
VERDICTS = ("grounded", "partially_grounded", "ungrounded", "not_checked")


def load_route_labels(path: Path = ROUTES_FILE) -> dict[str, dict]:
    """id -> {with_paper, no_paper, paper_open, notes}."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    labels = {}
    for item in raw:
        labels[item["id"]] = {
            "with_paper": item["expected_route_with_paper"],
            "no_paper": item["expected_route_no_paper"],
            "paper_open": item.get("paper_open"),
            "notes": item.get("notes", ""),
        }
    return labels


# --- routing ----------------------------------------------------------------------


def confusion_matrix(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """expected -> predicted -> count, over every route including unused ones.

    Kept dense rather than sparse: an all-zero row is itself information — it says
    the golden set never exercised that route.
    """
    matrix = {expected: dict.fromkeys(ROUTES, 0) for expected in ROUTES}
    for expected, predicted in pairs:
        if expected in matrix and predicted in matrix[expected]:
            matrix[expected][predicted] += 1
    return matrix


def route_accuracy(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(1 for expected, predicted in pairs if expected == predicted) / len(pairs)


def per_route_scores(matrix: dict[str, dict[str, int]]) -> dict[str, dict]:
    """Recall, precision and support per route.

    Both, not just accuracy: they fail in different directions. Low recall on `web`
    means the router misses questions that need the internet; low precision means it
    sends library questions out to a paid API. The second is the expensive mistake.
    """
    scores = {}
    for route in ROUTES:
        support = sum(matrix[route].values())
        hits = matrix[route][route]
        predicted_total = sum(matrix[other][route] for other in ROUTES)
        scores[route] = {
            "support": support,
            "correct": hits,
            "recall": round(hits / support, 4) if support else None,
            "precision": round(hits / predicted_total, 4) if predicted_total else None,
        }
    return scores


def misroutes(rows: list[dict]) -> list[dict]:
    """Every row where predicted != expected — the list worth reading by hand."""
    return [r for r in rows if r["expected"] != r["predicted"]]


# --- the rewrite loop -------------------------------------------------------------


def rewrite_metrics(rows: list[dict]) -> dict:
    """Trigger rate, and the recall@5 gap between questions that fired and didn't.

    This is the number that says whether the loop earns its latency. A high trigger
    rate with no recall gain means the grader is rejecting good chunks; a gain only
    on fired questions means the loop is doing real work.
    """
    answerable = [r for r in rows if not r.get("is_unanswerable")]
    fired = [r for r in answerable if r.get("rewrite_count", 0) > 0]
    not_fired = [r for r in answerable if r.get("rewrite_count", 0) == 0]

    def mean_recall(subset):
        values = [r["recall@5"] for r in subset if r.get("recall@5") is not None]
        return round(statistics.mean(values), 4) if values else None

    def mean_latency(subset):
        values = [r["latency_ms"] for r in subset if r.get("latency_ms") is not None]
        return round(statistics.mean(values), 1) if values else None

    fired_recall, unfired_recall = mean_recall(fired), mean_recall(not_fired)
    return {
        "n_answerable": len(answerable),
        "n_fired": len(fired),
        "trigger_rate": round(len(fired) / len(answerable), 4) if answerable else None,
        "recall@5_when_fired": fired_recall,
        "recall@5_when_not_fired": unfired_recall,
        "recall@5_delta": (
            round(fired_recall - unfired_recall, 4)
            if fired_recall is not None and unfired_recall is not None
            else None
        ),
        "latency_ms_when_fired": mean_latency(fired),
        "latency_ms_when_not_fired": mean_latency(not_fired),
    }


def rewrite_recall_within_question(rows: list[dict]) -> dict:
    """First-pass vs final-pass recall@5, on the questions where the loop fired.

    The between-groups comparison `rewrite_metrics` reports is confounded and should
    be read alongside this one, never instead of it: the loop fires *because*
    retrieval was bad, so "questions where it fired" is "the hard questions" by
    construction, and a negative delta there says nothing about whether rewriting
    helped. Comparing a question to itself removes the selection entirely.
    """
    fired = [
        r
        for r in rows
        if r.get("rewrite_count", 0) > 0
        and not r.get("is_unanswerable")
        and r.get("recall@5_first_pass") is not None
        and r.get("recall@5_final_pass") is not None
    ]
    if not fired:
        return {"n_fired": 0}

    deltas = [r["recall@5_final_pass"] - r["recall@5_first_pass"] for r in fired]
    return {
        "n_fired": len(fired),
        "mean_first_pass": round(statistics.mean(r["recall@5_first_pass"] for r in fired), 4),
        "mean_final_pass": round(statistics.mean(r["recall@5_final_pass"] for r in fired), 4),
        "mean_delta": round(statistics.mean(deltas), 4),
        "improved": sum(1 for d in deltas if d > 0),
        "unchanged": sum(1 for d in deltas if d == 0),
        "worsened": sum(1 for d in deltas if d < 0),
        # A question the loop rescued from nothing is the case it exists for.
        "rescued_from_zero": sum(
            1 for r in fired if r["recall@5_first_pass"] == 0 and r["recall@5_final_pass"] > 0
        ),
    }


def fired_despite_good_retrieval(rows: list[dict], threshold: float = 1.0) -> list[dict]:
    """Questions where the loop fired even though the first pass already found
    everything — the signature of the grader rejecting good chunks rather than of
    retrieval having missed."""
    return [
        r
        for r in rows
        if r.get("rewrite_count", 0) > 0
        and not r.get("is_unanswerable")
        and (r.get("recall@5_first_pass") or 0) >= threshold
    ]


# --- grounding and refusals -------------------------------------------------------


def grounding_summary(rows: list[dict]) -> dict:
    counts = dict.fromkeys(VERDICTS, 0)
    for row in rows:
        verdict = row.get("verdict")
        if verdict in counts:
            counts[verdict] += 1
    total = sum(counts.values())
    return {
        "counts": counts,
        "share": {k: round(v / total, 4) for k, v in counts.items()} if total else {},
        "n": total,
    }


def refusal_correctness(rows: list[dict]) -> dict:
    """The unanswerable-set number, computed both ways.

    Prefix matching scores an answer correct only if it opens with a fixed sentence.
    The grounding method scores it correct if it invented nothing. The gap between
    them is the point — it is how much of v2's refusal number was measurement error
    rather than model behaviour.
    """
    unanswerable = [r for r in rows if r.get("is_unanswerable")]
    if not unanswerable:
        return {"n": 0}

    by_prefix = sum(1 for r in unanswerable if r.get("refused_by_prefix"))
    by_grounding = sum(1 for r in unanswerable if r.get("refused_by_grounding"))
    disagreements = [
        {
            "id": r["id"],
            "prefix": bool(r.get("refused_by_prefix")),
            "grounding": bool(r.get("refused_by_grounding")),
            "answer": (r.get("answer") or "")[:160],
        }
        for r in unanswerable
        if bool(r.get("refused_by_prefix")) != bool(r.get("refused_by_grounding"))
    ]
    n = len(unanswerable)
    return {
        "n": n,
        "prefix_correct": by_prefix,
        "prefix_rate": round(by_prefix / n, 4),
        "grounding_correct": by_grounding,
        "grounding_rate": round(by_grounding / n, 4),
        "disagreements": disagreements,
    }


# --- cost and latency -------------------------------------------------------------


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None, "mean": None}
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "p50": round(statistics.median(ordered), 2),
        "p95": round(ordered[idx], 2),
        "mean": round(statistics.mean(ordered), 2),
    }


def cost_latency_by_route(rows: list[dict]) -> dict:
    """Per-route p50/p95 latency and median cost, plus an overall row."""
    out = {}
    for route in (*ROUTES, "all"):
        subset = rows if route == "all" else [r for r in rows if r.get("route") == route]
        if not subset:
            continue
        costs = [r["cost_usd"] for r in subset if r.get("cost_usd") is not None]
        out[route] = {
            "n": len(subset),
            "latency_ms": _percentiles([r["latency_ms"] for r in subset if r.get("latency_ms")]),
            "median_cost_usd": round(statistics.median(costs), 6) if costs else None,
            "total_cost_usd": round(sum(costs), 4) if costs else None,
        }
    return out


def total_cost(usage: dict) -> float:
    """Sum every per-node cost in a State `usage` dict."""
    return round(
        sum(entry.get("cost_usd", 0.0) for entry in usage.values() if isinstance(entry, dict)), 6
    )


# --- rendering --------------------------------------------------------------------


def render_confusion_matrix(matrix: dict[str, dict[str, int]]) -> str:
    header = "| expected \\ predicted | " + " | ".join(ROUTES) + " | total |"
    sep = "|---" * (len(ROUTES) + 2) + "|"
    lines = [header, sep]
    for expected in ROUTES:
        row = matrix[expected]
        total = sum(row.values())
        cells = " | ".join(
            f"**{row[p]}**" if p == expected and row[p] else str(row[p]) for p in ROUTES
        )
        lines.append(f"| {expected} | {cells} | {total} |")
    return "\n".join(lines)


def render_per_route(scores: dict[str, dict]) -> str:
    lines = ["| route | support | correct | recall | precision |", "|---|---|---|---|---|"]
    for route in ROUTES:
        s = scores[route]
        recall = "—" if s["recall"] is None else f"{s['recall']:.3f}"
        precision = "—" if s["precision"] is None else f"{s['precision']:.3f}"
        lines.append(f"| {route} | {s['support']} | {s['correct']} | {recall} | {precision} |")
    return "\n".join(lines)
