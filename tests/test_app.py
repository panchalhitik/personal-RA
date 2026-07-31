from pathlib import Path

from personal_ra.app import history_to_messages, list_papers, session_cost
from personal_ra.ask import Answer, Message
from personal_ra.cite import Citation


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
