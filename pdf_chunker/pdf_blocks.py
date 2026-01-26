from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import chain
from statistics import median

import fitz  # PyMuPDF

from .heading_detection import TRAILING_PUNCTUATION, _detect_heading_fallback
from .inline_styles import (
    InlineStyleSpan,
    build_index_map,
    build_index_remapper,
    merge_inline_styles,
    normalize_spans,
)
from .language import default_language
from .strategies.bullets import BulletHeuristicStrategy, default_bullet_strategy
from .text_cleaning import (
    HYPHEN_CHARS_ESC,
    bullet_trace_scope,
    clean_text,
    emit_bullet_trace,
    insert_numbered_list_newlines,
    remove_stray_bullet_lines,
)

# -- Data models -------------------------------------------------------------------------


@dataclass
class Block:
    text: str
    source: dict
    type: str = "paragraph"
    language: str | None = None
    bbox: tuple | None = None
    inline_styles: list[InlineStyleSpan] | None = None


@dataclass
class PagePayload:
    number: int
    blocks: list[Block]


def _preview_text(text: str, limit: int = 100) -> str:
    return repr(text[:limit])


def _resolve_strategy(
    strategy: BulletHeuristicStrategy | None,
) -> BulletHeuristicStrategy:
    return strategy or default_bullet_strategy()


# -- Page extraction --------------------------------------------------------------------


def _filter_margin_artifacts(blocks, page_height: float) -> list:
    numeric_pattern = r"^[0-9ivxlcdm]+$"

    def is_numeric_fragment(text: str) -> bool:
        import re

        words = text.split()
        return 0 < len(words) <= 6 and all(re.fullmatch(numeric_pattern, w) for w in words)

    filtered: list = []
    for block in blocks:
        x0, y0, x1, y1, raw_text = block[:5]
        if y0 < page_height * 0.15 or y0 > page_height * 0.85:
            cleaned = clean_text(raw_text).strip()
            if is_numeric_fragment(cleaned):
                if filtered and filtered[-1][1] > page_height * 0.85:
                    filtered.pop()
                continue
        filtered.append(block)
    return filtered


def _spans_indicate_heading(spans: list[dict], text: str) -> bool:
    return any(span.get("flags", 0) & 2 for span in spans) and not text.rstrip().endswith(
        TRAILING_PUNCTUATION
    )


def _structured_block(
    page, block_tuple, page_num, filename, page_font_size: float | None = None
) -> Block | None:
    raw_text = block_tuple[4]
    cleaned = clean_text(raw_text)
    if not cleaned:
        return None

    is_heading = False
    if len(cleaned.split()) < 15:
        try:
            block_dict = page.get_text("dict", clip=block_tuple[:4])["blocks"][0]
            spans = block_dict["lines"][0]["spans"]
            is_heading = _spans_indicate_heading(spans, cleaned)
        except (KeyError, IndexError, TypeError):
            is_heading = _detect_heading_fallback(cleaned)

    inline_styles = _extract_block_inline_styles(
        page, block_tuple, cleaned, page_font_size
    )

    return Block(
        type="heading" if is_heading else "paragraph",
        text=cleaned,
        language=default_language(),
        source={"filename": filename, "page": page_num, "location": None},
        bbox=block_tuple[:4],
        inline_styles=inline_styles,
    )


def _extract_block_inline_styles(
    page, block_tuple, cleaned_text: str, page_font_size: float | None = None
) -> list[InlineStyleSpan] | None:
    if not cleaned_text:
        return None

    try:
        block_dict = page.get_text("dict", clip=block_tuple[:4])
    except Exception:
        return None

    raw_spans = tuple(_iter_text_spans(block_dict))
    if not raw_spans:
        return None

    raw_text = "".join(str(span.get("text", "")) for span, _ in raw_spans)
    if not raw_text:
        return None

    offsets = tuple(_span_offsets(raw_spans))
    styles_per_span = tuple(
        tuple(_collect_styles(span, baseline, page_font_size))
        for span, baseline in raw_spans
    )
    spans = tuple(
        InlineStyleSpan(
            start=start,
            end=end,
            style=style,
            confidence=1.0,
            attrs=attrs,
        )
        for (start, end), styles in zip(offsets, styles_per_span)
        for style, attrs in styles
    )
    if not spans:
        return None

    remapper = build_index_remapper(build_index_map(raw_text, cleaned_text))
    normalized = normalize_spans(spans, len(cleaned_text), remapper)
    return list(normalized) or None


def _iter_text_spans(
    block_dict: Mapping[str, object],
) -> Iterable[tuple[Mapping[str, object], float | None]]:
    blocks = block_dict.get("blocks", [])
    for block in blocks if isinstance(blocks, Sequence) else ():
        if not hasattr(block, "get"):
            continue
        if block.get("type", 0) not in (0, None):
            continue
        lines = block.get("lines", [])
        if not isinstance(lines, Sequence):
            continue
        for line in lines:
            if not hasattr(line, "get"):
                continue
            spans = line.get("spans", [])
            if not isinstance(spans, Sequence):
                continue
            baseline = _line_baseline(spans)
            for span in spans:
                if not hasattr(span, "get"):
                    continue
                text = span.get("text", "")
                if not text:
                    continue
                yield span, baseline


def _span_offsets(
    spans: Sequence[tuple[Mapping[str, object], float | None]],
) -> Iterable[tuple[int, int]]:
    cursor = 0
    for span, _ in spans:
        text = str(span.get("text", ""))
        length = len(text)
        start = cursor
        cursor += length
        yield (start, cursor)


def _collect_styles(
    span: Mapping[str, object],
    line_baseline: float | None,
    page_font_size: float | None = None,
) -> Iterable[tuple[str, Mapping[str, str] | None]]:
    flags = int(span.get("flags", 0))
    font = str(span.get("font", "")).lower()

    if flags & 16 or any(marker in font for marker in ("bold", "black", "heavy")):
        yield ("bold", None)
    if flags & 2 or any(marker in font for marker in ("italic", "oblique", "slanted")):
        yield ("italic", None)
    if flags & 8:
        yield ("underline", None)
    if _is_monospace(font):
        yield ("monospace", None)

    # Detect large font (heading by size)
    span_size = span.get("size") if hasattr(span, "get") else None
    if (
        page_font_size
        and isinstance(span_size, (int, float))
        and span_size > page_font_size * 1.3
    ):
        yield ("large", None)

    baseline_style = _baseline_style(span, line_baseline)
    if baseline_style:
        yield (baseline_style, None)

    attrs = _link_attrs(span)
    if attrs:
        yield ("link", attrs)


def _line_baseline(spans: Sequence[Mapping[str, object]]) -> float | None:
    positions = [
        float(origin[1])
        for span in spans
        if hasattr(span, "get")
        and isinstance((origin := span.get("origin")), Sequence)
        and len(origin) >= 2
    ]
    return median(positions) if positions else None


def _baseline_style(span: Mapping[str, object], line_baseline: float | None) -> str | None:
    if line_baseline is None:
        return None
    origin = span.get("origin") if hasattr(span, "get") else None
    size = span.get("size") if hasattr(span, "get") else None
    if not isinstance(origin, Sequence) or len(origin) < 2:
        return None
    if not isinstance(size, (int, float)) or size <= 0:
        return None
    y = float(origin[1])
    delta = line_baseline - y
    threshold = max(1.0, float(size) * 0.25)
    if delta >= threshold:
        return "superscript"
    if delta <= -threshold:
        return "subscript"
    return None


def _link_attrs(span: Mapping[str, object]) -> Mapping[str, str] | None:
    link = span.get("link") if hasattr(span, "get") else None
    if isinstance(link, str):
        return {"href": link}
    if isinstance(link, Mapping):
        href = link.get("uri") or link.get("url") or link.get("href")
        if href:
            attrs = {"href": str(href)}
            title = link.get("title")
            if title:
                attrs["title"] = str(title)
            return attrs
    for key in ("uri", "url", "href"):
        value = span.get(key) if hasattr(span, "get") else None
        if value:
            return {"href": str(value)}
    return None


def _is_monospace(font: str) -> bool:
    return any(token in font for token in ("mono", "code", "courier", "console"))


def _filter_by_zone_margins(
    blocks: list,
    page_height: float,
    footer_margin: float | None = None,
    header_margin: float | None = None,
) -> list:
    """Filter out blocks in header/footer zones based on margins.

    Args:
        blocks: Raw block tuples from page.get_text("blocks")
        page_height: Height of the page in points
        footer_margin: Points from bottom to exclude
        header_margin: Points from top to exclude

    Returns:
        Filtered list of blocks
    """
    if not footer_margin and not header_margin:
        return blocks

    filtered = []
    for block in blocks:
        # Block tuple: (x0, y0, x1, y1, text, block_no, block_type)
        y0, y1 = block[1], block[3]

        # Check header zone (top of page)
        if header_margin and y0 < header_margin:
            continue

        # Check footer zone (bottom of page)
        if footer_margin and y1 > (page_height - footer_margin):
            continue

        filtered.append(block)

    return filtered


def _compute_page_font_size(page) -> float | None:
    """Compute the most common (baseline) font size for a page.

    Used to detect headings that are distinguished by larger font size
    rather than bold/italic styling.
    """
    try:
        page_dict = page.get_text("dict")
        sizes: list[float] = []
        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size")
                    text = span.get("text", "")
                    # Only count spans with actual text content
                    if isinstance(size, (int, float)) and size > 0 and text.strip():
                        sizes.append(round(size, 1))
        if not sizes:
            return None
        # Return the most common font size (mode)
        size_counts = Counter(sizes)
        return size_counts.most_common(1)[0][0]
    except Exception:
        return None


def _extract_page_blocks(
    page,
    page_num: int,
    filename: str,
    footer_margin: float | None = None,
    header_margin: float | None = None,
) -> list[Block]:
    page_height = page.rect.height
    raw_blocks = page.get_text("blocks")
    # Calculate baseline font size for heading detection
    page_font_size = _compute_page_font_size(page)
    # First apply zone margin filtering (geometric)
    zone_filtered = _filter_by_zone_margins(raw_blocks, page_height, footer_margin, header_margin)
    # Then apply existing heuristic filtering
    filtered = _filter_margin_artifacts(zone_filtered, page_height)
    return [
        b
        for block in filtered
        if (b := _structured_block(page, block, page_num, filename, page_font_size))
        is not None
    ]


def read_pages(
    filepath: str,
    excluded: set[int],
    extractor: Callable[[fitz.Page, int, str], list[Block]] | None = None,
    *,
    footer_margin: float | None = None,
    header_margin: float | None = None,
) -> Iterable[PagePayload]:
    """Yield ``PagePayload`` objects for each non-excluded page.

    Args:
        filepath: Path to PDF file
        excluded: Set of page numbers to exclude
        extractor: Custom block extractor function (deprecated, use margins)
        footer_margin: Points from bottom to exclude as footer zone
        header_margin: Points from top to exclude as header zone
    """
    doc = fitz.open(filepath)
    try:
        for page_num, page in enumerate(doc, start=1):
            if page_num in excluded:
                continue
            if extractor is not None:
                blocks = extractor(page, page_num, os.path.basename(filepath))
            else:
                blocks = _extract_page_blocks(
                    page,
                    page_num,
                    os.path.basename(filepath),
                    footer_margin=footer_margin,
                    header_margin=header_margin,
                )
            yield PagePayload(number=page_num, blocks=blocks)
    finally:
        doc.close()


# -- Block merging ---------------------------------------------------------------------

MIN_WORDS_FOR_CONTINUATION = 6

COMMON_SENTENCE_STARTERS = {
    "The",
    "This",
    "That",
    "A",
    "An",
    "In",
    "On",
    "At",
    "As",
    "By",
    "For",
    "From",
    "If",
    "When",
    "While",
    "After",
    "Before",
    "It",
    "He",
    "She",
    "They",
    "We",
    "I",
}

SENTENCE_CONTINUATION_LOWER = frozenset(
    chain(
        (word.lower() for word in COMMON_SENTENCE_STARTERS if word != "I"),
        (
            "and",
            "but",
            "or",
            "nor",
            "so",
            "yet",
            "also",
            "still",
            "thus",
            "hence",
            "therefore",
            "then",
            "otherwise",
            "instead",
            "meanwhile",
            "besides",
            "consequently",
            "finally",
            "furthermore",
            "likewise",
            "nevertheless",
            "nonetheless",
            "next",
            "overall",
            "similarly",
            "subsequently",
            "ultimately",
            "additionally",
            "of",
            "to",
            "into",
            "onto",
            "upon",
            "within",
            "without",
            "across",
            "through",
            "throughout",
            "toward",
            "towards",
        ),
    )
)

_CAPTION_PREFIXES = ("Figure", "Table", "Exhibit")
_CAPTION_RE = re.compile(rf"^(?:{'|'.join(_CAPTION_PREFIXES)})\s+\d")
_CAPTION_STYLE_TAGS = frozenset({"italic", "bold"})
_CAPTION_STYLE_THRESHOLD = 0.6


def _inline_style_ratio(block: Block, styles: Iterable[str]) -> float:
    spans = tuple(block.inline_styles or ())
    total = len(block.text or "")
    style_set = frozenset(styles)
    return (
        sum(
            max(0, min(total, span.end) - max(0, span.start))
            for span in spans
            if span.style in style_set
        )
        / total
        if spans and total > 0
        else 0.0
    )


def _looks_like_caption(text: str, *, emphasis_ratio: float = 0.0) -> bool:
    stripped = text.strip()
    return (
        bool(stripped and _CAPTION_RE.match(stripped)) or emphasis_ratio >= _CAPTION_STYLE_THRESHOLD
    )


def _word_count(text: str) -> int:
    return len(text.split())


def _has_sufficient_context(text: str) -> bool:
    return _word_count(text) >= MIN_WORDS_FOR_CONTINUATION


def _fragment_tail(text: str) -> str:
    return re.split(r"[.!?]\s*", text)[-1]


def _is_common_sentence_starter(word: str) -> bool:
    return word in COMMON_SENTENCE_STARTERS


def _is_comma_uppercase_continuation(curr_text: str, next_text: str) -> bool:
    return curr_text.endswith(",") and next_text[:1].isupper()


def _is_heading_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return stripped.endswith(":") or _detect_heading_fallback(stripped)


def _is_indented_continuation(curr: Block, nxt: Block) -> bool:
    curr_bbox = curr.bbox
    next_bbox = nxt.bbox
    if not curr_bbox or not next_bbox:
        return False
    curr_x0, _, _, curr_y1 = curr_bbox
    next_x0, next_y0, _, _ = next_bbox
    vertical_gap = next_y0 - curr_y1
    indent_diff = next_x0 - curr_x0
    return indent_diff > 10 and vertical_gap < 8


def _indent_delta(curr: Block, nxt: Block) -> float | None:
    curr_bbox = curr.bbox
    next_bbox = nxt.bbox
    if not curr_bbox or not next_bbox:
        return None
    try:
        return float(next_bbox[0]) - float(curr_bbox[0])
    except (TypeError, ValueError, IndexError):
        return None


def _is_numbered_list_continuation(
    curr: Block,
    nxt: Block,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    if not curr.text or not nxt.text:
        return False
    curr_text = curr.text.strip()
    next_text = nxt.text.strip()
    if not curr_text or not next_text:
        return False
    heuristics = _resolve_strategy(strategy)
    if not heuristics.starts_with_number(curr_text):
        return False
    if heuristics.starts_with_number(next_text):
        return False
    if _is_heading_like(next_text):
        return False
    if nxt.text[:1].isspace():
        return True
    indent = _indent_delta(curr, nxt)
    return indent is not None and indent > 8


def _looks_like_quote_boundary(curr_text: str, next_text: str) -> bool:
    if (
        curr_text.endswith(('."', ".'", '!"', "!'", '?"', "?'"))
        and next_text
        and next_text[0].isupper()
    ):
        return True
    attribution_starters = [
        "said",
        "asked",
        "replied",
        "continued",
        "added",
        "noted",
    ]
    if curr_text.endswith(('"', "'")) and any(
        next_text.lower().startswith(starter) for starter in attribution_starters
    ):
        return True
    return False


def _is_cross_page_continuation(curr: Block, nxt: Block) -> bool:
    curr_text = curr.text.strip()
    next_text = nxt.text.strip()
    curr_page = curr.source.get("page")
    next_page = nxt.source.get("page")
    if _looks_like_caption(
        curr_text, emphasis_ratio=_inline_style_ratio(curr, _CAPTION_STYLE_TAGS)
    ):
        return False
    if not (curr_text and next_text):
        return False
    if curr_page is None or next_page is None or next_page != curr_page + 1:
        return False
    if curr_text.endswith((".", "!", "?")):
        return False
    if _looks_like_quote_boundary(curr_text, next_text):
        return False
    if _detect_heading_fallback(next_text) and not _has_sufficient_context(curr_text):
        return False
    tail_words = _word_count(_fragment_tail(curr_text))
    if tail_words > 12:
        return False
    first_word = next_text.split()[0]
    if first_word[0].islower():
        return True
    if _is_common_sentence_starter(first_word):
        return False
    return True


def _is_cross_page_paragraph_continuation(curr: Block, nxt: Block) -> bool:
    if _looks_like_caption(
        curr.text, emphasis_ratio=_inline_style_ratio(curr, _CAPTION_STYLE_TAGS)
    ):
        return False
    curr_page = curr.source.get("page")
    next_page = nxt.source.get("page")
    if curr_page is None or next_page is None or next_page != curr_page + 1:
        return False
    curr_bbox = curr.bbox
    next_bbox = nxt.bbox
    if not curr_bbox or not next_bbox:
        return False
    curr_x0, _, _, _ = curr_bbox
    next_x0, _, _, _ = next_bbox
    indent_diff = next_x0 - curr_x0
    if indent_diff > 10:
        return False
    next_text = nxt.text.strip()
    if _detect_heading_fallback(next_text):
        return False
    return True


def _is_same_page_continuation(
    curr: Block,
    nxt: Block,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    heuristics = _resolve_strategy(strategy)
    curr_text = curr.text.strip()
    next_text = nxt.text.strip()
    curr_page = curr.source.get("page")
    next_page = nxt.source.get("page")
    if curr_page is None or next_page is None:
        return False
    if curr_page != next_page or not next_text:
        return False
    if _looks_like_caption(
        curr_text,
        emphasis_ratio=_inline_style_ratio(curr, _CAPTION_STYLE_TAGS),
    ):
        return False
    if any(b in curr_text for b in heuristics.bullet_chars):
        return False
    if _is_heading_like(next_text):
        return False
    first_word = next_text.split()[0]
    if curr_text.endswith((".", "!", "?", ":", ";")) and not _is_common_sentence_starter(
        first_word
    ):
        return False
    if _is_comma_uppercase_continuation(curr_text, next_text):
        return True
    if next_text[0].islower() or _is_common_sentence_starter(first_word):
        return True
    trailing = _trailing_alpha_token(curr_text)
    letters = re.sub(r"[^A-Za-z]", "", first_word)
    if (
        trailing
        and trailing.islower()
        and letters
        and letters[0].isupper()
        and letters[1:].islower()
    ):
        return True
    return False


def _is_quote_continuation(curr_text: str, next_text: str) -> bool:
    curr_open_quotes = curr_text.count('"') - curr_text.count('\\"')
    curr_open_single = curr_text.count("'") - curr_text.count("\\'")
    next_closing_quotes = next_text.count('"') - next_text.count('\\"')
    next_closing_single = next_text.count("'") - next_text.count("\\'")
    return (curr_open_quotes % 2 == 1 and next_closing_quotes > 0) or (
        curr_open_single % 2 == 1 and next_closing_single > 0
    )


def _merge_bullet_text(
    reason: str,
    current: str,
    nxt: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> tuple[str, str | None]:
    heuristics = _resolve_strategy(strategy)

    def merge_fragment() -> tuple[str, str | None]:
        fragment, remainder = heuristics.split_bullet_fragment(nxt)
        return f"{current} {fragment}", remainder

    def merge_continuation() -> tuple[str, str | None]:
        return current.rstrip(" " + heuristics.bullet_chars) + " " + nxt, None

    def merge_short_fragment() -> tuple[str, str | None]:
        return f"{current} {nxt}", None

    def merge_list() -> tuple[str, str | None]:
        adjusted = re.sub(
            rf":\s*(?=-|[{heuristics.bullet_chars_esc}])",
            ":\n",
            current,
        )
        return adjusted + "\n" + nxt, None

    handlers = {
        "bullet_fragment": merge_fragment,
        "bullet_continuation": merge_continuation,
        "bullet_short_fragment": merge_short_fragment,
        "bullet_list": merge_list,
    }

    merged, remainder = handlers[reason]()
    emit_bullet_trace(
        "merge_bullet_raw",
        before=current,
        after=merged,
        extra={
            "reason": reason,
            "next_preview": _preview_text(nxt),
            "remainder_preview": _preview_text(remainder) if remainder else None,
        },
    )
    cleaned = remove_stray_bullet_lines(merged)
    emit_bullet_trace(
        "merge_bullet_cleaned",
        before=merged,
        after=cleaned,
        extra={"reason": reason},
    )
    return cleaned, remainder


def _leading_alpha_token(text: str) -> str:
    """Return the first alphabetic token stripped of punctuation."""

    for token in text.split():
        letters = re.sub(r"[^A-Za-z]", "", token)
        if letters:
            return letters
    return ""


_FIRST_ALPHA_RE = re.compile(r"([A-Za-z])")


def _lower_first_alpha(token: str) -> str:
    """Lower-case the first alphabetic character within ``token``."""

    return _FIRST_ALPHA_RE.sub(lambda match: match.group(1).lower(), token, count=1)


def _hyphen_head_word(text: str) -> str:
    """Return the word preceding a terminal hyphen in ``text``."""

    match = re.search(rf"([A-Za-z]+)[{HYPHEN_CHARS_ESC}]$", text)
    return match.group(1) if match else ""


def _trailing_alpha_token(text: str) -> str:
    """Return the last alphabetic token stripped of punctuation."""

    for token in reversed(text.split()):
        letters = re.sub(r"[^A-Za-z]", "", token)
        if letters:
            return letters
    return ""


def _normalize_hyphenated_tail(curr_text: str, next_text: str) -> str:
    """Lower-case the continuation when a hyphenated word crosses lines."""

    head = _hyphen_head_word(curr_text)
    if not next_text or not head:
        return next_text

    prefix, sep, remainder = next_text.partition(" ")
    letters = re.sub(r"[^A-Za-z]", "", prefix)
    if letters and letters[0].isupper() and letters[1:].islower() and head.islower():
        lowered = prefix[0].lower() + prefix[1:]
        return lowered + (sep + remainder if sep else "")
    return next_text


def _normalize_sentence_tail(current_text: str, next_text: str) -> str:
    """Lower-case titlecased sentence continuations following soft breaks."""

    if not next_text:
        return next_text

    trailing = _trailing_alpha_token(current_text)
    prefix, sep, remainder = next_text.partition(" ")
    letters = re.sub(r"[^A-Za-z]", "", prefix)
    normalized_candidate = letters.lower()
    has_single_alpha = len(letters) == 1
    if (
        trailing
        and trailing.islower()
        and letters
        and letters[0].isupper()
        and (letters[1:].islower() or has_single_alpha)
        and normalized_candidate in SENTENCE_CONTINUATION_LOWER
        and not current_text.endswith((".", "!", "?", ":", ";"))
    ):
        lowered = _lower_first_alpha(prefix)
        return lowered + (sep + remainder if sep else "")
    return next_text


def _block_has_list_markers(block: Block) -> bool:
    """Return True when ``block`` carries explicit list annotations."""

    if getattr(block, "type", None) == "list_item":
        return True
    source = block.source if isinstance(block.source, dict) else {}
    nested = (
        source,
        *(
            candidate
            for candidate in (source.get("attrs"), source.get("block_attrs"))
            if isinstance(candidate, dict)
        ),
    )
    return any(candidate.get("list_kind") for candidate in nested if isinstance(candidate, dict))


def _is_footnote_block(block: Block) -> bool:
    source = block.source if isinstance(block.source, dict) else {}
    return bool(source.get("footnote_block"))


def _is_in_footer_zone(
    block: Block,
    page_heights: dict[int, float] | None,
    footer_margin: float | None,
) -> bool:
    """Check if a block is in the footer zone based on Y position."""
    if not footer_margin or not page_heights:
        return False

    page = block.source.get("page") if block.source else None
    if page is None or page not in page_heights:
        return False

    # bbox is (x0, y0, x1, y1)
    if not block.bbox or len(block.bbox) < 4:
        return False

    y1 = block.bbox[3]  # Bottom edge of block
    page_height = page_heights[page]

    # Block is in footer zone if its bottom edge is within footer_margin from page bottom
    return y1 > (page_height - footer_margin)


def _should_merge_blocks(
    curr: Block,
    nxt: Block,
    *,
    strategy: BulletHeuristicStrategy | None = None,
    page_heights: dict[int, float] | None = None,
    footer_margin: float | None = None,
) -> tuple[bool, str]:
    heuristics = _resolve_strategy(strategy)
    curr_text = curr.text.strip()
    next_text = nxt.text.strip()

    if not curr_text or not next_text:
        return False, "empty_text"

    # Don't merge if the next block is a standalone styled heading
    if _is_standalone_styled_heading(nxt):
        return False, "next_is_styled_heading"

    # Don't merge if CURRENT block is in the footer zone (don't append content to footer)
    if _is_in_footer_zone(curr, page_heights, footer_margin):
        return False, "curr_in_footer_zone"

    # Don't merge if next block is in the footer zone
    if _is_in_footer_zone(nxt, page_heights, footer_margin):
        return False, "next_in_footer_zone"

    curr_page = curr.source.get("page")
    next_page = nxt.source.get("page")

    if next_text.startswith("—"):
        return True, "author_attribution"

    if heuristics.is_bullet_continuation(curr_text, next_text):
        return True, "bullet_continuation"

    if heuristics.is_bullet_fragment(curr_text, next_text):
        return True, "bullet_fragment"

    last_line = heuristics.last_non_empty_line(curr_text)
    if (
        heuristics.starts_with_bullet(last_line)
        and not heuristics.starts_with_bullet(next_text)
        and len(next_text.split()) <= 3
    ):
        return True, "bullet_short_fragment"

    colon_intro = curr_text.rstrip().endswith(":") and not heuristics.starts_with_bullet(curr_text)
    if heuristics.is_bullet_list_pair(curr_text, next_text):
        if colon_intro and not _block_has_list_markers(curr):
            return False, "colon_intro_without_list_markers"
        return True, "bullet_list"

    if heuristics.is_numbered_list_pair(curr_text, next_text):
        return True, "numbered_list"

    if heuristics.is_numbered_continuation(curr_text, next_text) and not _is_heading_like(
        next_text
    ):
        return True, "numbered_continuation"
    if _is_numbered_list_continuation(curr, nxt, strategy=heuristics):
        return True, "numbered_continuation"

    if re.fullmatch(r"\d+[.)]", curr_text) and not heuristics.starts_with_number(next_text):
        return True, "numbered_standalone"

    if re.search(r"\n\d+[.)]\s*$", curr_text) and not heuristics.starts_with_number(next_text):
        return True, "numbered_suffix"

    curr_has_quote = '"' in curr_text or "'" in curr_text
    next_has_quote = '"' in next_text or "'" in next_text
    if curr_has_quote or next_has_quote:
        if _is_heading_like(next_text):
            return False, "no_merge"
        if _is_quote_continuation(curr_text, next_text):
            return True, "quote_continuation"

    if _is_indented_continuation(curr, nxt) and not _detect_heading_fallback(next_text):
        return True, "indented_continuation"

    hyphen_pattern = rf"[{HYPHEN_CHARS_ESC}]$"
    double_hyphen_pattern = rf"[{HYPHEN_CHARS_ESC}]{{2,}}$"
    tail_token = _leading_alpha_token(next_text)
    head_word = _hyphen_head_word(curr_text)
    tail_is_titlecase = tail_token and tail_token[0].isupper() and tail_token[1:].islower()
    tail_is_lower = bool(next_text and next_text[0].islower())
    if (
        re.search(hyphen_pattern, curr_text)
        and not re.search(double_hyphen_pattern, curr_text)
        and next_text
        and (tail_is_lower or (tail_is_titlecase and head_word.islower()))
    ):
        return True, "hyphenated_continuation"
    elif (
        _is_same_page_continuation(curr, nxt, strategy=heuristics)
        or _is_cross_page_continuation(curr, nxt)
        or _is_cross_page_paragraph_continuation(curr, nxt)
    ):
        return True, "sentence_continuation"

    return False, "no_merge"


_HEADING_STYLE_TAGS = frozenset(
    {"bold", "italic", "small_caps", "caps", "uppercase", "large"}
)


def _is_standalone_styled_heading(block: Block) -> bool:
    """Return True if block is a standalone heading based on inline styles.

    Detects short text (≤12 words) that is fully covered by a heading-style
    span (bold, italic, large, etc.) - such blocks should not be merged into
    adjacent paragraphs.
    """
    styles = block.inline_styles
    if not styles:
        return False

    text = block.text.strip()
    if not text:
        return False

    words = text.split()
    if len(words) > 12:
        return False

    length = len(text)

    for span in styles:
        if span.start <= 0 and span.end >= length:
            # Span covers entire text
            style = span.style.lower() if span.style else ""
            if style in _HEADING_STYLE_TAGS:
                return True

    return False


def within_page_span(pages: Iterable[int | None], limit: int = 1) -> bool:
    """Return True if pages stay within ``limit`` span."""
    nums = [p for p in pages if p is not None]
    return not nums or max(nums) - min(nums) <= limit


def merge_continuation_blocks(
    blocks: Iterable[Block],
    *,
    strategy: BulletHeuristicStrategy | None = None,
    page_heights: dict[int, float] | None = None,
    footer_margin: float | None = None,
) -> Iterable[Block]:
    """Merge adjacent blocks that form continuations.

    Args:
        blocks: Iterable of Block objects to merge
        strategy: Bullet heuristic strategy
        page_heights: Dict mapping page numbers to page heights (in points)
        footer_margin: Points from bottom to treat as footer zone
    """
    heuristics = _resolve_strategy(strategy)
    items = list(blocks)
    if not items:
        return []

    merged: list[Block] = []
    i = 0
    while i < len(items):
        current = items[i]
        if _is_footnote_block(current):
            if merged:
                # Merge footnote inline_styles too
                prev = merged[-1]
                prev_text = prev.text.rstrip()
                fn_text = current.text.strip()
                new_text = f"{prev_text}\n{fn_text}"
                new_styles = merge_inline_styles(
                    prev.inline_styles,
                    current.inline_styles,
                    len(prev_text),
                    1,  # newline separator
                )
                merged[-1] = replace(prev, text=new_text, inline_styles=new_styles)
            else:
                merged.append(current)
            i += 1
            continue
        current_text = current.text.strip()
        current_styles = current.inline_styles
        pages = [current.source.get("page")]
        j = i + 1
        deferred_footnotes: list[Block] = []
        while j < len(items):
            nxt = items[j]
            if _is_footnote_block(nxt):
                deferred_footnotes.append(nxt)
                j += 1
                continue
            next_text = nxt.text.strip()
            next_page = nxt.source.get("page")
            candidate_pages = [*pages, next_page]
            if any(
                b is not None and a is not None and b - a > 1
                for a, b in zip(candidate_pages, candidate_pages[1:])
            ) or not within_page_span(candidate_pages):
                break
            curr_for_merge = replace(current, source={**current.source, "page": pages[-1]})
            should_merge, reason = _should_merge_blocks(
                curr_for_merge,
                nxt,
                strategy=heuristics,
                page_heights=page_heights,
                footer_margin=footer_margin,
            )
            if should_merge:
                # Determine separator length based on merge reason
                separator_len = 0
                if reason == "hyphenated_continuation":
                    normalized_tail = _normalize_hyphenated_tail(current_text, next_text)
                    merged_text = (
                        re.sub(rf"[{HYPHEN_CHARS_ESC}]$", "", current_text) + normalized_tail
                    )
                    # No separator for hyphenated continuation
                    separator_len = 0
                elif reason == "sentence_continuation":
                    normalized_sentence = _normalize_sentence_tail(current_text, next_text)
                    merged_text = current_text + " " + normalized_sentence
                    separator_len = 1  # space
                elif reason.startswith("bullet_"):
                    with bullet_trace_scope(
                        current.source,
                        stage="merge_continuation_blocks",
                        reason=reason,
                        current_preview=_preview_text(current_text),
                        next_preview=_preview_text(next_text),
                        page=current.source.get("page"),
                        next_page=next_page,
                    ):
                        merged_text, remainder = _merge_bullet_text(
                            reason,
                            current_text,
                            next_text,
                            strategy=heuristics,
                        )
                    # For bullet merges, separator varies
                    separator_len = len(merged_text) - len(current_text) - len(next_text)
                    if separator_len < 0:
                        separator_len = 0
                    new_styles = merge_inline_styles(
                        current_styles,
                        nxt.inline_styles,
                        len(current_text),
                        separator_len,
                    )
                    current = replace(current, text=merged_text, inline_styles=new_styles)
                    current_text = merged_text
                    current_styles = new_styles
                    if remainder:
                        items[j] = replace(nxt, text=remainder.lstrip())
                        pages.append(next_page)
                        break
                    j += 1
                    pages.append(next_page)
                    continue
                elif reason == "numbered_list":
                    merged_text = current_text + "\n" + next_text
                    separator_len = 1  # newline
                elif reason == "numbered_continuation":
                    merged_text = current_text + " " + next_text
                    separator_len = 1  # space
                elif reason == "numbered_standalone":
                    processed = insert_numbered_list_newlines(next_text)
                    merged_text = current_text + " " + processed
                    separator_len = 1  # space
                elif reason == "numbered_suffix":
                    marker_match = re.search(r"\n(\d+[.)])\s*$", current_text)
                    base = re.sub(r"\n\d+[.)]\s*$", "", current_text)
                    processed = insert_numbered_list_newlines(next_text)
                    marker = marker_match.group(1) if marker_match else ""
                    prefix = f"{marker} " if marker else ""
                    merged_text = f"{base}\n{prefix}{processed}".strip()
                    # Complex case - separator calculation approximate
                    separator_len = len(merged_text) - len(current_text) - len(next_text)
                    if separator_len < 0:
                        separator_len = 0
                elif reason == "indented_continuation":
                    merged_text = current_text + "\n" + next_text
                    separator_len = 1  # newline
                elif reason == "author_attribution":
                    merged_text = current_text + " " + next_text
                    separator_len = 1  # space
                else:
                    merged_text = current_text + " " + next_text
                    separator_len = 1  # space

                # Merge inline styles from both blocks
                new_styles = merge_inline_styles(
                    current_styles,
                    nxt.inline_styles,
                    len(current_text),
                    separator_len,
                )
                current = replace(current, text=merged_text, inline_styles=new_styles)
                current_text = merged_text
                current_styles = new_styles
                j += 1
                pages.append(next_page)
            else:
                break
        if within_page_span(pages) and len(pages) > 1:
            page_nums = [p for p in pages if p is not None]
            if page_nums:
                current = replace(
                    current,
                    source={
                        **current.source,
                        "page_range": (min(page_nums), max(page_nums)),
                    },
                )
        if deferred_footnotes:
            footnote_texts = [fn.text.strip() for fn in deferred_footnotes if fn.text.strip()]
            if footnote_texts:
                fn_combined = "\n".join(footnote_texts)
                prev_text = current.text.rstrip()
                merged_text = prev_text + "\n" + fn_combined
                # Merge footnote styles (collect all)
                fn_styles: list[InlineStyleSpan] = []
                offset = len(prev_text) + 1  # +1 for the first newline
                for fn in deferred_footnotes:
                    if fn.inline_styles:
                        for span in fn.inline_styles:
                            fn_styles.append(
                                replace(span, start=span.start + offset, end=span.end + offset)
                            )
                    offset += len(fn.text.strip()) + 1  # +1 for newline between
                combined_styles: list[InlineStyleSpan] | None = None
                if current_styles or fn_styles:
                    combined_styles = list(current_styles or []) + fn_styles
                current = replace(current, text=merged_text, inline_styles=combined_styles)
        merged.append(current)
        i = j
    return merged
