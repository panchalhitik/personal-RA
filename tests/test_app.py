from pathlib import Path

from personal_ra.app import (
    build_annotations,
    group_exchanges,
    history_to_messages,
    list_papers,
    notes_path,
    quote_preview,
    session_cost,
)
from personal_ra.ask import Answer, Message
from personal_ra.cite import Citation

FIXTURES = Path(__file__).parent / "fixtures"


def test_list_papers_sorted(tmp_path: Path) -> None:
    (tmp_path / "b.pdf").touch()
    (tmp_path / "a.pdf").touch()
    (tmp_path / "notes.txt").touch()
    assert [p.name for p in list_papers(tmp_path)] == ["a.pdf", "b.pdf"]


def test_list_papers_missing_dir(tmp_path: Path) -> None:
    assert list_papers(tmp_path / "nope") == []


def test_history_to_messages() -> None:
    chat = [
        {"role": "user", "content": "Hi", "answer": None},
        {"role": "assistant", "content": "Hello", "answer": None},
    ]
    assert history_to_messages(chat) == [
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello"),
    ]


def test_session_cost_sums_answer_costs() -> None:
    citation = Citation(quote="q", page=1, char_offset=0, verified=True, match_type="exact")
    chat = [
        {"role": "user", "content": "Q1", "answer": None},
        {
            "role": "assistant",
            "content": "A1",
            "answer": Answer(
                text="A1", citations=[citation], unverified=[], usage={"cost_usd": 0.01}
            ),
        },
        {
            "role": "assistant",
            "content": "A2",
            "answer": Answer(text="A2", citations=[], unverified=[], usage={"cost_usd": 0.005}),
        },
    ]
    assert session_cost(chat) == 0.015
    assert session_cost([]) == 0.0


def test_group_exchanges_pairs_question_with_answer() -> None:
    chat = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "pending"},
    ]
    groups = group_exchanges(chat)
    assert [[e["content"] for e in g] for g in groups] == [["Q1", "A1"], ["Q2", "A2"], ["pending"]]
    assert group_exchanges([]) == []


def test_notes_path_uses_paper_stem(tmp_path: Path) -> None:
    assert notes_path(Path("papers/foo.pdf"), tmp_path) == tmp_path / "foo.md"


def test_quote_preview_truncates() -> None:
    assert quote_preview("short") == "short"
    long = "x" * 100
    preview = quote_preview(long)
    assert len(preview) <= 61 and preview.endswith("…")


def test_build_annotations_from_fixture() -> None:
    located = Citation(
        quote="echo foxtrot reports experimental results here",
        page=1,
        char_offset=0,
        verified=True,
        match_type="exact",
    )
    unlocatable = Citation(
        quote="completely absent phrase xyzzy plugh",
        page=1,
        char_offset=0,
        verified=True,
        match_type="fuzzy",
    )
    annotations = build_annotations([located, unlocatable], FIXTURES / "two_column.pdf")
    assert annotations  # located quote produced boxes; unlocatable one was skipped
    for a in annotations:
        assert a["page"] == 1
        assert a["width"] > 0 and a["height"] > 0
        assert set(a) == {"page", "x", "y", "width", "height", "color"}
