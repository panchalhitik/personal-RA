from rapidfuzz import fuzz

from conftest import make_paper
from personal_ra.cite import FUZZY_THRESHOLD, _normalize_with_map, verify_quote

PAGE_1 = (
    "Attention mechanisms have become central to sequence modeling. "
    "We trained the model on eight GPUs for twelve hours. "
    "The experiment used seven distinct random seeds"
)
PAGE_2 = (
    "and averaged all results across those runs. "
    "Our ablation study removes the positional encoding entirely. "
    "The quick brown fox jumps over the lazy dog near the river bank."
)


PAPER = make_paper([PAGE_1, PAGE_2])


def test_exact_match_correct_page() -> None:
    c = verify_quote("Our ablation study removes the positional encoding", PAPER)
    assert c.verified and c.match_type == "exact"
    assert c.page == 2


def test_char_offset_points_at_match() -> None:
    c = verify_quote("ablation study removes", PAPER)
    assert c.page == 2 and c.char_offset is not None
    assert PAGE_2[c.char_offset :].casefold().startswith("ablation")


def test_match_despite_whitespace_and_linebreaks() -> None:
    c = verify_quote("We  trained the\nmodel   on eight\n\nGPUs", PAPER)
    assert c.verified and c.match_type == "exact"
    assert c.page == 1


def test_match_despite_curly_quotes_and_dashes() -> None:
    paper = make_paper(['The model - dubbed "TinyNet" - is Bob\'s baseline.'])
    c = verify_quote("The model — dubbed “TinyNet” — is Bob’s baseline.", paper)
    assert c.verified and c.match_type == "exact"
    assert c.page == 1


def test_hallucinated_quote_fails() -> None:
    c = verify_quote("The model achieves 99.9% accuracy on ImageNet", PAPER)
    assert not c.verified
    assert c.match_type == "failed"
    assert c.page is None and c.char_offset is None


def test_quote_spanning_page_boundary_resolves_to_start_page() -> None:
    c = verify_quote("seven distinct random seeds and averaged all results", PAPER)
    assert c.verified
    assert c.page == 1


def test_fuzzy_just_above_threshold() -> None:
    # one substitution in a 44-char quote ≈ 97.7, above the 95 threshold
    quote = "The quick brown fox jumps over the hazy dog"
    norm_q, _ = _normalize_with_map(quote)
    norm_s, _ = _normalize_with_map(PAGE_2)
    score = fuzz.partial_ratio(norm_q, norm_s)
    assert FUZZY_THRESHOLD <= score < 100, f"precondition: score={score}"
    c = verify_quote(quote, PAPER)
    assert c.verified and c.match_type == "fuzzy"
    assert c.page == 2


def test_fuzzy_just_below_threshold() -> None:
    # three substitutions in the same quote ≈ 93.2, below the 95 threshold
    quote = "The quick crown fax jumps over the hazy dog"
    norm_q, _ = _normalize_with_map(quote)
    norm_s, _ = _normalize_with_map(PAGE_2)
    score = fuzz.partial_ratio(norm_q, norm_s)
    assert 80 < score < FUZZY_THRESHOLD, f"precondition: score={score}"
    c = verify_quote(quote, PAPER)
    assert not c.verified
    assert c.match_type == "failed"


def test_empty_quote_does_not_crash() -> None:
    c = verify_quote("", PAPER)
    assert not c.verified and c.match_type == "failed"


def test_whitespace_only_quote_does_not_crash() -> None:
    c = verify_quote("  \n\t ", PAPER)
    assert not c.verified and c.match_type == "failed"


def test_empty_paper_does_not_crash() -> None:
    c = verify_quote("anything", make_paper([""]))
    assert not c.verified and c.match_type == "failed"
