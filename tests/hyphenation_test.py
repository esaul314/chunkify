import pytest
from pdf_chunker.text_cleaning import clean_text
import pdf_chunker.pymupdf4llm_integration as p4l


@pytest.fixture
def force_pymupdf4llm(monkeypatch):
    monkeypatch.setenv("PDF_CHUNKER_USE_PYMUPDF4LLM", "true")
    monkeypatch.setattr(p4l, "is_pymupdf4llm_available", lambda: True)
    yield
    monkeypatch.delenv("PDF_CHUNKER_USE_PYMUPDF4LLM", raising=False)


def test_hyphenation_fix_with_pymupdf4llm(force_pymupdf4llm):
    sample = "a con-\n tainer and special-\n ists in man-\n agement"
    cleaned = clean_text(sample)
    assert all(word in cleaned for word in ("container", "specialists", "management"))
    assert all(
        broken not in cleaned
        for broken in (
            "con- tainer",
            "special- ists",
            "man- agement",
        )
    )


@pytest.mark.parametrize(
    "block,expected",
    [
        ({"text": "Storage engi-\n neer", "source": {"page": 1}}, "Storage engineer"),
        ({"text": "_respon-\n_sibility_", "source": {"page": 1}}, "responsibility"),
    ],
)
def test_clean_block_hyphen_fix(block, expected):
    cleaned = p4l._clean_pymupdf4llm_block(block)
    assert cleaned is not None
    assert cleaned["text"] == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "We sell business-critical, off-the-shelf solutions.",
            ("business-critical", "off-the-shelf"),
        ),
        (
            "The release schedule is business-\u00adcritical.",
            ("business-critical",),
        ),
    ],
)
def test_preserve_existing_hyphens(text, expected):
    cleaned = clean_text(text)
    assert all(word in cleaned for word in expected)


def test_bullet_hyphen_continuation():
    text = "* Some text before ambig-\n* uous word"
    cleaned = clean_text(text)
    assert "ambiguous" in cleaned
    assert cleaned.count("*") == 1


def test_join_preserves_double_letters():
    text = "bal-\n loon"
    assert clean_text(text) == "balloon"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("business-\ncritical systems", "business-critical systems"),
        ("business\u00ad\ncritical systems", "businesscritical systems"),
    ],
)
def test_crossline_hyphen_preserved(text, expected):
    assert clean_text(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("provision-\ning", "provisioning"),
        ("through-\nOut", "throughout"),
        ("through\u2010 Out", "throughout"),
    ],
)
def test_crossline_spurious_hyphen_removed(text, expected):
    assert clean_text(text) == expected
