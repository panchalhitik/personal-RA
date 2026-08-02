from types import SimpleNamespace
from unittest.mock import MagicMock

from conftest import make_paper
from personal_ra.ask import (
    API_REFUSAL,
    FALLBACK_MODEL,
    MODEL,
    REFUSAL,
    Answer,
    Message,
    ask,
)

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
    assert usage["models_tried"] == [MODEL, FALLBACK_MODEL]
    # Both attempts are billed for their input tokens even though neither answered;
    # reporting only one would under-state what the question cost.
    assert usage["cache_write_tokens"] == 2 * 17152


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


# --- the fallback model -------------------------------------------------------------


def _ok_response(text="A fallback answer.", cache_write=20000):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=25,
            output_tokens=300,
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=0,
        ),
    )


def test_a_refusal_is_retried_on_the_fallback_model():
    """Measured: the fallback answers 2 of the 6 papers the primary declines."""
    client = MagicMock()
    client.messages.create.side_effect = [_refusal_response(), _ok_response()]
    answer = ask(make_paper(["some text"]), "a question", client=client)

    assert answer.text == "A fallback answer."
    assert answer.usage["model"] == FALLBACK_MODEL
    assert answer.usage["used_fallback"] is True
    assert "api_refusal" not in answer.usage

    models = [c.kwargs["model"] for c in client.messages.create.call_args_list]
    assert models == [MODEL, FALLBACK_MODEL]


def test_the_fallback_request_omits_temperature_and_disables_thinking():
    """Sonnet 5 rejects a non-default temperature with a 400, and thinks by default —
    which would consume the max_tokens budget the answer needs."""
    client = MagicMock()
    client.messages.create.side_effect = [_refusal_response(), _ok_response()]
    ask(make_paper(["some text"]), "a question", client=client)

    primary, fallback = (c.kwargs for c in client.messages.create.call_args_list)
    assert primary["temperature"] == 0
    assert "thinking" not in primary
    assert "temperature" not in fallback
    assert fallback["thinking"] == {"type": "disabled"}


def test_the_fallback_sees_the_same_paper_and_question():
    client = MagicMock()
    client.messages.create.side_effect = [_refusal_response(), _ok_response()]
    ask(make_paper(["a distinctive sentence"]), "the question", client=client)

    primary, fallback = (c.kwargs for c in client.messages.create.call_args_list)
    assert fallback["system"] == primary["system"]
    assert fallback["messages"] == primary["messages"]
    assert "a distinctive sentence" in fallback["system"][1]["text"]


def test_usage_sums_both_attempts_when_the_fallback_answers():
    client = MagicMock()
    client.messages.create.side_effect = [_refusal_response(), _ok_response()]
    usage = ask(make_paper(["some text"]), "a question", client=client).usage
    assert usage["cache_write_tokens"] == 17152 + 20000
    assert usage["output_tokens"] == 1 + 300


def test_both_models_refusing_reports_both():
    client = MagicMock()
    client.messages.create.side_effect = [_refusal_response(), _refusal_response()]
    answer = ask(make_paper(["some text"]), "a question", client=client)
    assert answer.text.startswith(API_REFUSAL)
    assert MODEL in answer.text and FALLBACK_MODEL in answer.text
    assert answer.usage["models_tried"] == [MODEL, FALLBACK_MODEL]


def test_a_failing_fallback_reports_the_original_refusal_rather_than_raising():
    client = MagicMock()
    client.messages.create.side_effect = [_refusal_response(), RuntimeError("model unavailable")]
    answer = ask(make_paper(["some text"]), "a question", client=client)
    assert answer.text.startswith(API_REFUSAL)
    assert answer.usage["models_tried"] == [MODEL]


def test_a_normal_answer_never_calls_the_fallback():
    client = MagicMock()
    client.messages.create.return_value = _ok_response("A primary answer.")
    answer = ask(make_paper(["some text"]), "a question", client=client)
    assert client.messages.create.call_count == 1
    assert answer.usage["used_fallback"] is False
    assert answer.usage["model"] == MODEL


def test_mid_stream_refusals_discard_the_partial_answer():
    """Observed on q07: stop_reason 'refusal' WITH 727 chars of content. A truncated
    answer presented as complete is worse than no answer."""
    partial = _refusal_response()
    partial.content = [SimpleNamespace(type="text", text="According to the paper, 12 defenses")]
    client = MagicMock()
    client.messages.create.side_effect = [partial, partial]
    answer = ask(make_paper(["some text"]), "a question", client=client)
    assert "12 defenses" not in answer.text
    assert answer.text.startswith(API_REFUSAL)
