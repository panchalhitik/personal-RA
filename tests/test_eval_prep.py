from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import chromadb

from personal_ra.eval_prep import _renumber_ids, draft_questions, list_papers
from personal_ra.library import COLLECTION

CHUNKS = [
    ("p1:0", "Contrastive loss over augmented pairs.", "aaa111", "Paper A", 2024),
    ("p1:1", "We evaluate on three benchmarks.", "aaa111", "Paper A", 2024),
    ("p2:0", "Reinforcement learning from human feedback.", "bbb222", "Paper B", 2025),
]


def make_db(tmp_path: Path) -> Path:
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.upsert(
        ids=[c[0] for c in CHUNKS],
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        documents=[c[1] for c in CHUNKS],
        metadatas=[
            {
                "paper_id": c[2],
                "paper_title": c[3],
                "page": 1,
                "section": "1. Intro",
                "chunk_index": i,
                "year": c[4],
                "source_path": f"papers/{c[2]}.pdf",
            }
            for i, c in enumerate(CHUNKS)
        ],
    )
    return tmp_path / "db"


def test_list_papers_aggregates_chunk_counts(tmp_path: Path) -> None:
    papers = list_papers(make_db(tmp_path))
    assert [p["paper_id"] for p in papers] == ["aaa111", "bbb222"]
    assert papers[0]["chunks"] == 2 and papers[1]["chunks"] == 1
    assert papers[0]["year"] == 2024 and papers[0]["title"] == "Paper A"


def test_renumber_ids_makes_them_unique() -> None:
    # The model restarts numbering per paper; ids must be globally unique or the
    # golden set fails validation.
    text = (
        "# --- draft from: A\n"
        '- id: d01\n  question: "one"\n'
        '- id: d02\n  question: "two"\n'
        "# --- draft from: B\n"
        '- id: d01\n  question: "three"\n'
    )
    out = _renumber_ids(text)
    ids = [line.split("id:")[1].strip() for line in out.splitlines() if "- id:" in line]
    assert ids == ["d01", "d02", "d03"]
    assert len(set(ids)) == len(ids)
    assert "# --- draft from: B" in out  # comments preserved


def test_draft_questions_uses_corpus_and_strips_fences(tmp_path: Path) -> None:
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text='```yaml\n- id: d01\n  question: "What loss?"\n```',
            )
        ]
    )
    out = draft_questions(n_papers=1, per_paper=2, db_path=make_db(tmp_path), client=client)
    assert "```" not in out
    assert "- id: d01" in out
    assert out.startswith("# --- draft from:")
    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Contrastive loss" in prompt or "Reinforcement learning" in prompt
