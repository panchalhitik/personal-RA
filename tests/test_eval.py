from pathlib import Path

import pytest

from personal_ra.eval import (
    Question,
    aggregate,
    dedup_preserving_order,
    evaluate_config,
    load_golden_set,
    recall_at_k,
    reciprocal_rank,
    render_markdown_table,
    validate_golden_set,
    write_results,
)

# Hand-computed fixture: expected papers are A and B.
#   ranking [C, A, D, B, E] -> A at rank 2, B at rank 4
#   recall@1 = 0/2, recall@3 = 1/2, recall@5 = 2/2, RR = 1/2
RANKING = ["C", "A", "D", "B", "E"]
EXPECTED = ["A", "B"]


def test_recall_at_k_hand_computed() -> None:
    assert recall_at_k(RANKING, EXPECTED, 1) == 0.0
    assert recall_at_k(RANKING, EXPECTED, 2) == 0.5
    assert recall_at_k(RANKING, EXPECTED, 3) == 0.5
    assert recall_at_k(RANKING, EXPECTED, 4) == 1.0
    assert recall_at_k(RANKING, EXPECTED, 5) == 1.0


def test_recall_ignores_duplicate_expectations() -> None:
    assert recall_at_k(["A", "B"], ["A", "A"], 2) == 1.0
    assert recall_at_k([], ["A"], 5) == 0.0
    assert recall_at_k(["A"], [], 5) == 0.0


def test_reciprocal_rank_hand_computed() -> None:
    assert reciprocal_rank(RANKING, EXPECTED) == 0.5  # A at rank 2
    assert reciprocal_rank(RANKING, ["B"]) == 0.25  # B at rank 4
    assert reciprocal_rank(RANKING, ["C"]) == 1.0
    assert reciprocal_rank(RANKING, ["Z"]) == 0.0  # not retrieved


def test_dedup_preserving_order() -> None:
    assert dedup_preserving_order(["A", "B", "A", "C", "B"]) == ["A", "B", "C"]
    assert dedup_preserving_order([]) == []


def test_aggregate_means_and_refusal_rate() -> None:
    per_question = [
        {"is_unanswerable": False, "recall@5": 1.0, "rr": 1.0},
        {"is_unanswerable": False, "recall@5": 0.0, "rr": 0.0},
        {"is_unanswerable": False, "recall@5": 0.5, "rr": 0.5},
        {"is_unanswerable": True, "recall@5": 0.0, "rr": 0.0, "refused": True},
        {"is_unanswerable": True, "recall@5": 0.0, "rr": 0.0, "refused": False},
    ]
    metrics = aggregate(per_question, k_values=(5,))
    assert metrics["n_questions"] == 5
    assert metrics["n_answerable"] == 3 and metrics["n_unanswerable"] == 2
    assert metrics["recall@5"] == 0.5  # mean of 1.0, 0.0, 0.5 — unanswerable excluded
    assert metrics["mrr"] == 0.5
    assert metrics["refusal_rate"] == 0.5


def test_aggregate_refusal_rate_none_when_not_measured() -> None:
    metrics = aggregate([{"is_unanswerable": True, "recall@5": 0.0, "rr": 0.0}], k_values=(5,))
    assert metrics["refusal_rate"] is None


def test_evaluate_config_end_to_end() -> None:
    questions = [
        Question(id="q1", question="about A", category="factual", expected_paper_ids=["A"]),
        Question(
            id="q2", question="about B and C", category="cross_paper", expected_paper_ids=["B", "C"]
        ),
        Question(id="q3", question="nothing", category="unanswerable"),
    ]

    def fake_search(question: str, k: int) -> list[str]:
        if "about A" in question:
            return ["A", "A", "D"]  # duplicates collapse to one paper
        if "B and C" in question:
            return ["D", "B", "E", "C"]
        return ["X", "Y"]

    outcome = evaluate_config(questions, fake_search, k=10, k_values=(1, 3, 5))
    rows = {r["id"]: r for r in outcome["per_question"]}
    assert rows["q1"]["retrieved_papers"] == ["A", "D"]
    assert rows["q1"]["recall@1"] == 1.0 and rows["q1"]["rr"] == 1.0
    assert rows["q2"]["recall@1"] == 0.0  # only D at rank 1
    assert rows["q2"]["recall@5"] == 1.0  # both B and C within top 5
    assert rows["q2"]["rr"] == 0.5  # B at rank 2
    assert outcome["metrics"]["n_answerable"] == 2
    assert outcome["metrics"]["recall@5"] == 1.0


def test_load_and_validate_golden_set(tmp_path: Path) -> None:
    path = tmp_path / "golden.yaml"
    path.write_text(
        """
- id: q01
  question: "What dataset?"
  ground_truth: "MNIST."
  expected_paper_ids: ["abc123"]
  category: factual
- id: q02
  question: "Anything about quantum gravity?"
  ground_truth: "Not covered."
  expected_paper_ids: []
  category: unanswerable
""",
        encoding="utf-8",
    )
    questions = load_golden_set(path)
    assert len(questions) == 2
    assert questions[1].is_unanswerable and questions[1].expected_paper_ids == []


def test_validate_rejects_malformed_sets() -> None:
    dup = [
        Question(id="q1", question="a", category="factual", expected_paper_ids=["A"]),
        Question(id="q1", question="b", category="factual", expected_paper_ids=["B"]),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        validate_golden_set(dup)

    with pytest.raises(ValueError, match="unknown category"):
        validate_golden_set([Question(id="q1", question="a", category="typo")])

    with pytest.raises(ValueError, match="must have no expected_paper_ids"):
        validate_golden_set(
            [Question(id="q1", question="a", category="unanswerable", expected_paper_ids=["A"])]
        )

    with pytest.raises(ValueError, match="need at least one expected_paper_id"):
        validate_golden_set([Question(id="q1", question="a", category="factual")])


def test_render_markdown_table_marks_best() -> None:
    results = [
        {
            "chunking": "fixed",
            "retrieval": "dense",
            "metrics": {
                "recall@1": 0.1,
                "recall@3": 0.2,
                "recall@5": 0.3,
                "recall@10": 0.4,
                "mrr": 0.15,
            },
        },
        {
            "chunking": "section_context",
            "retrieval": "hybrid",
            "metrics": {
                "recall@1": 0.5,
                "recall@3": 0.6,
                "recall@5": 0.7,
                "recall@10": 0.8,
                "mrr": 0.55,
            },
        },
    ]
    table = render_markdown_table(results)
    lines = table.strip().splitlines()
    assert lines[0].startswith("| Chunking |")
    assert "**section_context**" in lines[2]  # best recall@5, sorted first
    assert "fixed" in lines[3] and "**fixed**" not in lines[3]


def test_write_results_creates_timestamped_json(tmp_path: Path) -> None:
    path = write_results({"configs": []}, results_dir=tmp_path)
    assert path.exists() and path.suffix == ".json"
    assert path.parent == tmp_path
