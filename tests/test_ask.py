from types import SimpleNamespace
from unittest.mock import MagicMock

from conftest import make_paper
from personal_ra.ask import API_REFUSAL, MODEL, REFUSAL, Answer, Message, ask

PAGE_1 = (
    "We trained the model on eight GPUs for twelve hours. "
    "The dataset contains two million labeled examples."
)
PAGE_2 = "Our ablation study removes the positional encoding entirely."

PAPER = make_paper([PAGE_1, PAGE_2])


def fake_response(text: str, cache_write: int = 0, cache_read: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
        ),
    )


def make_client(text: str, **kwargs) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = fake_response(text, **kwargs)
    return client


def test_quotes_extracted_and_verified() -> None:
    client = make_client(
        "They used a large dataset <quote>The dataset contains two million labeled "
        "examples.</quote> and ran an ablation <quote>Our ablation study removes the "
        "positional encoding entirely.</quote>"
    )
    answer = ask(PAPER, "What data did they use?", client=client)
    assert len(answer.citations) == 2
    assert [c.page for c in answer.citations] == [1, 2]
    assert all(c.verified for c in answer.citations)


def test_verified_quotes_become_page_markers() -> None:
    client = make_client(
        "Training took a while <quote>We trained the model on eight GPUs for twelve "
        "hours.</quote> in total."
    )
    answer = ask(PAPER, "How long was training?", client=client)
    assert "[p. 1]" in answer.text
    assert "<quote>" not in answer.text and "</quote>" not in answer.text


def test_quote_text_stays_visible_inline() -> None:
    # Regression: an answer that is *only* a heading plus a quote must not collapse
    # to a bare page marker — the quote text stays in the answer body.
    client = make_client(
        "## The Hypothesis\n<quote>We trained the model on eight GPUs for twelve hours.</quote>"
    )
    answer = ask(PAPER, "What is the hypothesis?", client=client)
    assert "We trained the model on eight GPUs for twelve hours." in answer.text
    assert answer.text.count("[p. 1]") == 1


def test_unverified_quote_flagged_not_rendered() -> None:
    client = make_client(
        "It scores highly <quote>The model achieves 99.9% accuracy on ImageNet.</quote>"
    )
    answer = ask(PAPER, "What accuracy?", client=client)
    assert len(answer.unverified) == 1
    assert not answer.unverified[0].verified
    assert answer.citations == []
    assert "[p." not in answer.text
    assert "[unverified]" in answer.text


def test_cache_control_on_paper_block() -> None:
    client = make_client("No quotes here.")
    ask(PAPER, "Anything?", client=client)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["temperature"] == 0
    system = kwargs["system"]
    paper_blocks = [b for b in system if PAPER.full_text in b["text"]]
    assert len(paper_blocks) == 1
    assert paper_blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_no_quotes_does_not_crash() -> None:
    client = make_client(REFUSAL)
    answer = ask(PAPER, "What about quantum gravity?", client=client)
    assert isinstance(answer, Answer)
    assert answer.text == REFUSAL
    assert answer.citations == [] and answer.unverified == []


def test_history_passed_through() -> None:
    client = make_client("Sure.")
    history = [Message(role="user", content="Hi"), Message(role="assistant", content="Hello")]
    ask(PAPER, "Follow-up?", history=history, client=client)
    messages = client.messages.create.call_args.kwargs["messages"]
    assert messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Follow-up?"},
    ]


def test_usage_and_cost_computed() -> None:
    client = make_client("No quotes.", cache_write=4000, cache_read=0)
    answer = ask(PAPER, "Anything?", client=client)
    u = answer.usage
    assert u["cache_write_tokens"] == 4000
    assert u["cache_read_tokens"] == 0
    # 100*3 + 50*15 + 4000*3.75 per MTok
    assert u["cost_usd"] == round((100 * 3 + 50 * 15 + 4000 * 3.75) / 1_000_000, 6)


# --- API refusals (found by the v3 eval: ~19% of single-paper runs) ----------------


def _refusal_response(category="bio"):
    """HTTP 200, stop_reason 'refusal', and an EMPTY content list — the shape that
    made ask() silently return a blank answer."""
    return SimpleNamespace(
        content=[],
        stop_reason="refusal",
        stop_details=SimpleNamespace(type="refusal", category=category, explanation="..."),
        usage=SimpleNamespace(
            input_tokens=25,
            output_tokens=1,
            cache_creation_input_tokens=17152,
            cache_read_input_tokens=0,
        ),
    )


def test_api_refusal_is_reported_rather_than_returned_as_a_blank_answer():
    client = MagicMock()
    client.messages.create.return_value = _refusal_response()
    answer = ask(make_paper(["some text"]), "a question", client=client)

    assert answer.text.startswith(API_REFUSAL)
    assert "bio" in answer.text  # the category, so the cause is visible
    assert answer.text != ""
    assert answer.citations == [] and answer.unverified == []


def test_api_refusal_is_flagged_in_usage_so_the_eval_can_count_it():
    client = MagicMock()
    client.messages.create.return_value = _refusal_response("cyber")
    usage = ask(make_paper(["some text"]), "a question", client=client).usage
    assert usage["api_refusal"] is True
    assert usage["refusal_category"] == "cyber"
    # The tokens were still billed — a refusal is not free.
    assert usage["cache_write_tokens"] == 17152


def test_api_refusal_is_not_the_paper_refusal_string():
    """REFUSAL means the model read the paper and said the answer isn't there — a
    correct outcome. An API refusal means it never answered. Conflating them would
    inflate the refusal-correctness number in the eval."""
    client = MagicMock()
    client.messages.create.return_value = _refusal_response()
    answer = ask(make_paper(["some text"]), "a question", client=client)
    assert not answer.text.startswith(REFUSAL)


def test_a_normal_response_is_unaffected():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="A normal answer.")],
        stop_reason="end_turn",
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=10,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )
    answer = ask(make_paper(["some text"]), "a question", client=client)
    assert answer.text == "A normal answer."
    assert "api_refusal" not in answer.usage
