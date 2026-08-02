"""Step 3.8 — route-eval metrics, verified against hand-computed cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from personal_ra.route_eval import (
    ROUTES,
    confusion_matrix,
    cost_latency_by_route,
    grounding_summary,
    load_route_labels,
    misroutes,
    per_route_scores,
    refusal_correctness,
    render_confusion_matrix,
    render_per_route,
    rewrite_metrics,
    route_accuracy,
    total_cost,
)

# Hand-computed fixture. 10 classifications:
#   single_paper: 3 expected, 2 predicted right, 1 leaked to library
#   library:      5 expected, 4 predicted right, 1 leaked to web
#   web:          2 expected, 1 predicted right, 1 leaked to library
#   direct:       0 expected, 0 predicted
# accuracy = 7/10 = 0.70
PAIRS = [
    ("single_paper", "single_paper"),
    ("single_paper", "single_paper"),
    ("single_paper", "library"),
    ("library", "library"),
    ("library", "library"),
    ("library", "library"),
    ("library", "library"),
    ("library", "web"),
    ("web", "web"),
    ("web", "library"),
]


def test_route_accuracy_hand_computed():
    assert route_accuracy(PAIRS) == 0.7
    assert route_accuracy([]) == 0.0
    assert route_accuracy([("library", "library")]) == 1.0


def test_confusion_matrix_hand_computed():
    matrix = confusion_matrix(PAIRS)
    assert matrix["single_paper"] == {"single_paper": 2, "library": 1, "web": 0, "direct": 0}
    assert matrix["library"] == {"single_paper": 0, "library": 4, "web": 1, "direct": 0}
    assert matrix["web"] == {"single_paper": 0, "library": 1, "web": 1, "direct": 0}
    assert matrix["direct"] == dict.fromkeys(ROUTES, 0)


def test_matrix_is_dense_so_an_unused_route_is_visible():
    """An all-zero row says the golden set never exercised that route, which is
    information the headline accuracy hides."""
    matrix = confusion_matrix(PAIRS)
    assert "direct" in matrix
    assert sum(matrix["direct"].values()) == 0


def test_per_route_scores_hand_computed():
    scores = per_route_scores(confusion_matrix(PAIRS))

    # single_paper: 2 of 3 expected found; 2 of 2 predictions correct
    assert scores["single_paper"]["support"] == 3
    assert scores["single_paper"]["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert scores["single_paper"]["precision"] == 1.0

    # library: 4 of 5 expected found; 6 predictions made, 4 correct
    assert scores["library"]["recall"] == 0.8
    assert scores["library"]["precision"] == pytest.approx(4 / 6, abs=1e-4)

    # web: 1 of 2 found; 2 predicted, 1 correct
    assert scores["web"]["recall"] == 0.5
    assert scores["web"]["precision"] == 0.5

    # direct: never expected and never predicted — undefined, not zero
    assert scores["direct"]["recall"] is None
    assert scores["direct"]["precision"] is None


def test_precision_and_recall_are_reported_separately():
    """They fail in different directions: low `web` precision means library
    questions get sent to a paid API, which is the expensive mistake."""
    scores = per_route_scores(confusion_matrix(PAIRS))
    assert scores["web"]["recall"] != scores["library"]["precision"]


def test_misroutes_lists_exactly_the_disagreements():
    rows = [
        {"id": "q1", "expected": "library", "predicted": "library"},
        {"id": "q2", "expected": "library", "predicted": "web"},
        {"id": "q3", "expected": "web", "predicted": "library"},
    ]
    assert [r["id"] for r in misroutes(rows)] == ["q2", "q3"]


# --- rewrite loop -----------------------------------------------------------------


def test_rewrite_metrics_hand_computed():
    rows = [
        {"rewrite_count": 0, "recall@5": 1.0, "latency_ms": 100},
        {"rewrite_count": 0, "recall@5": 0.8, "latency_ms": 200},
        {"rewrite_count": 1, "recall@5": 0.6, "latency_ms": 900},
        {"rewrite_count": 2, "recall@5": 0.4, "latency_ms": 1500},
        {"rewrite_count": 0, "recall@5": None, "is_unanswerable": True},
    ]
    m = rewrite_metrics(rows)
    assert m["n_answerable"] == 4  # the unanswerable row is excluded
    assert m["n_fired"] == 2
    assert m["trigger_rate"] == 0.5
    assert m["recall@5_when_fired"] == 0.5  # (0.6 + 0.4) / 2
    assert m["recall@5_when_not_fired"] == 0.9  # (1.0 + 0.8) / 2
    assert m["recall@5_delta"] == -0.4
    assert m["latency_ms_when_fired"] == 1200.0
    assert m["latency_ms_when_not_fired"] == 150.0


def test_rewrite_metrics_on_an_empty_set():
    m = rewrite_metrics([])
    assert m["n_answerable"] == 0 and m["trigger_rate"] is None


# --- grounding and refusals -------------------------------------------------------


def test_grounding_summary_hand_computed():
    rows = [
        {"verdict": "grounded"},
        {"verdict": "grounded"},
        {"verdict": "partially_grounded"},
        {"verdict": "ungrounded"},
    ]
    s = grounding_summary(rows)
    assert s["n"] == 4
    assert s["counts"]["grounded"] == 2
    assert s["share"]["grounded"] == 0.5
    assert s["counts"]["not_checked"] == 0


def test_refusal_correctness_computed_both_ways():
    """The headline comparison: same six answers, two scoring methods."""
    rows = [
        # fixed string — both methods agree it is correct
        {
            "id": "q57",
            "is_unanswerable": True,
            "refused_by_prefix": True,
            "refused_by_grounding": True,
            "answer": "That isn't covered in my library.",
        },
        # hedged non-answer — prefix says miss, grounding says correct
        {
            "id": "q58",
            "is_unanswerable": True,
            "refused_by_prefix": False,
            "refused_by_grounding": True,
            "answer": "Nothing in the excerpts speaks to this.",
        },
        # invented — both agree it is wrong
        {
            "id": "q59",
            "is_unanswerable": True,
            "refused_by_prefix": False,
            "refused_by_grounding": False,
            "answer": "They report 91.2% top-1.",
        },
        # answerable rows are excluded entirely
        {
            "id": "q01",
            "is_unanswerable": False,
            "refused_by_prefix": False,
            "refused_by_grounding": False,
        },
    ]
    r = refusal_correctness(rows)
    assert r["n"] == 3
    assert r["prefix_correct"] == 1 and r["prefix_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert r["grounding_correct"] == 2 and r["grounding_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert [d["id"] for d in r["disagreements"]] == ["q58"]


def test_refusal_correctness_with_no_unanswerable_rows():
    assert refusal_correctness([{"is_unanswerable": False}])["n"] == 0


# --- cost and latency -------------------------------------------------------------


def test_cost_latency_by_route_hand_computed():
    rows = [
        {"route": "library", "latency_ms": 100, "cost_usd": 0.001},
        {"route": "library", "latency_ms": 300, "cost_usd": 0.003},
        {"route": "direct", "latency_ms": 50, "cost_usd": 0.0005},
    ]
    out = cost_latency_by_route(rows)
    assert out["library"]["n"] == 2
    assert out["library"]["latency_ms"]["p50"] == 200.0
    assert out["library"]["median_cost_usd"] == 0.002
    assert out["all"]["n"] == 3
    assert out["all"]["total_cost_usd"] == 0.0045
    assert "web" not in out  # routes with no rows are omitted, not zero-filled


def test_total_cost_sums_every_node():
    usage = {
        "route": {"cost_usd": 0.0004},
        "grade": {"cost_usd": 0.0031},
        "rewrite_1": {"cost_usd": 0.0009},
        "n_graded": 8,  # a non-dict entry must not break the sum
    }
    assert total_cost(usage) == 0.0044


# --- rendering and the labels file ------------------------------------------------


def test_confusion_matrix_renders_as_markdown():
    table = render_confusion_matrix(confusion_matrix(PAIRS))
    assert table.startswith("| expected \\ predicted |")
    assert table.count("\n") == len(ROUTES) + 1  # header + separator + one row per route


def test_per_route_table_marks_undefined_scores():
    table = render_per_route(per_route_scores(confusion_matrix(PAIRS)))
    assert "| direct | 0 | 0 | — | — |" in table


def test_routes_file_is_complete_and_well_formed():
    """63 questions x 2 states, every label a real route."""
    labels = load_route_labels(Path("eval") / "routes.yaml")
    assert len(labels) == 63
    for qid, label in labels.items():
        assert label["with_paper"] in ROUTES, qid
        assert label["no_paper"] in ROUTES, qid
        assert label["paper_open"], qid


def test_routes_file_ids_match_the_golden_set():
    from personal_ra.eval import load_golden_set

    golden = {q.id for q in load_golden_set()}
    labels = set(load_route_labels(Path("eval") / "routes.yaml"))
    assert labels == golden


def test_single_paper_is_impossible_without_a_paper_open():
    """A label saying single_paper with no paper open would be unsatisfiable."""
    labels = load_route_labels(Path("eval") / "routes.yaml")
    assert all(label["no_paper"] != "single_paper" for label in labels.values())


# --- within-question rewrite measurement ------------------------------------------


def test_within_question_delta_hand_computed():
    """The between-groups delta is confounded; this one compares each question to
    itself, so it is the number that can actually say whether rewriting helped."""
    from personal_ra.route_eval import rewrite_recall_within_question

    rows = [
        # fired and rescued from nothing
        {"rewrite_count": 1, "recall@5_first_pass": 0.0, "recall@5_final_pass": 1.0},
        # fired and improved partially
        {"rewrite_count": 2, "recall@5_first_pass": 0.5, "recall@5_final_pass": 1.0},
        # fired and got worse
        {"rewrite_count": 1, "recall@5_first_pass": 1.0, "recall@5_final_pass": 0.5},
        # fired and nothing changed
        {"rewrite_count": 1, "recall@5_first_pass": 1.0, "recall@5_final_pass": 1.0},
        # never fired — excluded
        {"rewrite_count": 0, "recall@5_first_pass": 1.0, "recall@5_final_pass": 1.0},
        # unanswerable — excluded, it can never have recall
        {
            "rewrite_count": 2,
            "is_unanswerable": True,
            "recall@5_first_pass": 0.0,
            "recall@5_final_pass": 0.0,
        },
    ]
    m = rewrite_recall_within_question(rows)
    assert m["n_fired"] == 4
    assert m["mean_first_pass"] == 0.625  # (0 + 0.5 + 1 + 1) / 4
    assert m["mean_final_pass"] == 0.875  # (1 + 1 + 0.5 + 1) / 4
    assert m["mean_delta"] == 0.25
    assert (m["improved"], m["unchanged"], m["worsened"]) == (2, 1, 1)
    assert m["rescued_from_zero"] == 1


def test_within_question_delta_with_nothing_fired():
    from personal_ra.route_eval import rewrite_recall_within_question

    assert rewrite_recall_within_question([{"rewrite_count": 0}])["n_fired"] == 0


def test_spurious_rewrites_are_isolated():
    """Firing on a question whose first pass already found everything is the grader
    rejecting good chunks, not retrieval missing — a different bug with a different fix."""
    from personal_ra.route_eval import fired_despite_good_retrieval

    rows = [
        {"id": "q48", "rewrite_count": 2, "recall@5_first_pass": 1.0},
        {"id": "q37", "rewrite_count": 1, "recall@5_first_pass": 0.0},
        {"id": "q12", "rewrite_count": 0, "recall@5_first_pass": 1.0},
        {"id": "q57", "rewrite_count": 2, "is_unanswerable": True, "recall@5_first_pass": 1.0},
    ]
    assert [r["id"] for r in fired_despite_good_retrieval(rows)] == ["q48"]


def test_grader_prompt_covers_multi_source_questions():
    """A comparison question has no single excerpt that answers it. Without this rule
    the grader rejects every chunk and the answer arrives with no citations at all —
    which is exactly what the first sample eval run produced."""
    from personal_ra.graph.grade import GRADER_SYSTEM

    assert "ONE PART" in GRADER_SYSTEM
    # Naming the two forbidden reasons is what actually changed behaviour; asking
    # for partial credit in the abstract did not.
    assert "never valid, and you must not give them" in GRADER_SYSTEM
    assert "does not compare X and Y" in GRADER_SYSTEM
    assert "cannot confirm this is the paper by that author or lab" in GRADER_SYSTEM
    # ...but the fix must not swing the other way and start keeping bibliographies.
    assert "Keep rejecting author lists" in GRADER_SYSTEM
