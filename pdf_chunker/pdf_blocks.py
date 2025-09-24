from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import chain
from statistics import median
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Tuple
import os
import re

import fitz  # PyMuPDF

from .inline_styles import (
    InlineStyleSpan,
    build_index_map,
    build_index_remapper,
    normalize_spans,
)
from .text_cleaning import (
    clean_text,
    HYPHEN_CHARS_ESC,
    remove_stray_bullet_lines,
    insert_numbered_list_newlines,
)
from .heading_detection import _detect_heading_fallback, TRAILING_PUNCTUATION
from .language import default_language
from .list_detection import (
    BULLET_CHARS,
    BULLET_CHARS_ESC,
    is_bullet_continuation,
    is_bullet_fragment,
    is_bullet_list_pair,
    is_numbered_continuation,
    is_numbered_list_pair,
    split_bullet_fragment,
    starts_with_bullet,
    starts_with_number,
    _last_non_empty_line,
)

# -- Data models -------------------------------------------------------------------------


@dataclass
class Block:
    text: str
    source: dict
    type: str = "paragraph"
    language: Optional[str] = None
    bbox: Optional[tuple] = None
    inline_styles: Optional[list[InlineStyleSpan]] = None


@dataclass
class PagePayload:
    number: int
    blocks: List[Block]


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


def _structured_block(page, block_tuple, page_num, filename) -> Block | None:
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

    inline_styles = _extract_block_inline_styles(page, block_tuple, cleaned)

    return Block(
        type="heading" if is_heading else "paragraph",
        text=cleaned,
        language=default_language(),
        source={"filename": filename, "page": page_num, "location": None},
        bbox=block_tuple[:4],
        inline_styles=inline_styles,
    )


def _extract_block_inline_styles(page, block_tuple, cleaned_text: str) -> Optional[list[InlineStyleSpan]]:
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
    styles_per_span = tuple(tuple(_collect_styles(span, baseline)) for span, baseline in raw_spans)
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
    block_dict: Mapping[str, object]
) -> Iterable[tuple[Mapping[str, object], Optional[float]]]:
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


def _span_offsets(spans: Sequence[tuple[Mapping[str, object], Optional[float]]]) -> Iterable[tuple[int, int]]:
    cursor = 0
    for span, _ in spans:
        text = str(span.get("text", ""))
        length = len(text)
        start = cursor
        cursor += length
        yield (start, cursor)


def _collect_styles(
    span: Mapping[str, object], line_baseline: Optional[float]
) -> Iterable[tuple[str, Optional[Mapping[str, str]]]]:
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

    baseline_style = _baseline_style(span, line_baseline)
    if baseline_style:
        yield (baseline_style, None)

    attrs = _link_attrs(span)
    if attrs:
        yield ("link", attrs)


def _line_baseline(spans: Sequence[Mapping[str, object]]) -> Optional[float]:
    positions = [
        float(origin[1])
        for span in spans
        if hasattr(span, "get")
        and isinstance((origin := span.get("origin")), Sequence)
        and len(origin) >= 2
    ]
    return median(positions) if positions else None


def _baseline_style(span: Mapping[str, object], line_baseline: Optional[float]) -> str | None:
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


def _link_attrs(span: Mapping[str, object]) -> Optional[Mapping[str, str]]:
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
def _extract_page_blocks(page, page_num: int, filename: str) -> list[Block]:
    page_height = page.rect.height
    raw_blocks = page.get_text("blocks")
    filtered = _filter_margin_artifacts(raw_blocks, page_height)
    return [
        b
        for block in filtered
        if (b := _structured_block(page, block, page_num, filename)) is not None
    ]


def read_pages(
    filepath: str,
    excluded: set[int],
    extractor: Callable[[fitz.Page, int, str], list[Block]] = _extract_page_blocks,
) -> Iterable[PagePayload]:
    """Yield ``PagePayload`` objects for each non-excluded page."""

    doc = fitz.open(filepath)
    try:
        for page_num, page in enumerate(doc, start=1):
            if page_num in excluded:
                continue
            blocks = extractor(page, page_num, os.path.basename(filepath))
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


def _looks_like_caption(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped and _CAPTION_RE.match(stripped))


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


def _is_cross_page_continuation(
    curr_text: str,
    next_text: str,
    curr_page: Optional[int],
    next_page: Optional[int],
) -> bool:
    if _looks_like_caption(curr_text):
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
    curr_text: str, next_text: str, curr_page: Optional[int], next_page: Optional[int]
) -> bool:
    if curr_page is None or next_page is None:
        return False
    if curr_page != next_page or not next_text:
        return False
    if _looks_like_caption(curr_text):
        return False
    if any(b in curr_text for b in BULLET_CHARS):
        return False
    if _is_heading_like(next_text):
        return False
    first_word = next_text.split()[0]
    if curr_text.endswith((".", "!", "?", ":", ";")) and not _is_common_sentence_starter(first_word):
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
    import re

    curr_open_quotes = curr_text.count('"') - curr_text.count('\\"')
    curr_open_single = curr_text.count("'") - curr_text.count("\\'")
    next_closing_quotes = next_text.count('"') - next_text.count('\\"')
    next_closing_single = next_text.count("'") - next_text.count("\\'")
    return (curr_open_quotes % 2 == 1 and next_closing_quotes > 0) or (
        curr_open_single % 2 == 1 and next_closing_single > 0
    )


def _merge_bullet_text(reason: str, current: str, nxt: str) -> Tuple[str, Optional[str]]:
    def merge_fragment() -> Tuple[str, Optional[str]]:
        fragment, remainder = split_bullet_fragment(nxt)
        return f"{current} {fragment}", remainder

    def merge_continuation() -> Tuple[str, Optional[str]]:
        return current.rstrip(" " + BULLET_CHARS) + " " + nxt, None

    def merge_short_fragment() -> Tuple[str, Optional[str]]:
        return f"{current} {nxt}", None

    def merge_list() -> Tuple[str, Optional[str]]:
        adjusted = re.sub(rf":\s*(?=-|[{BULLET_CHARS_ESC}])", ":\n", current)
        return adjusted + "\n" + nxt, None

    handlers = {
        "bullet_fragment": merge_fragment,
        "bullet_continuation": merge_continuation,
        "bullet_short_fragment": merge_short_fragment,
        "bullet_list": merge_list,
    }

    merged, remainder = handlers[reason]()
    return remove_stray_bullet_lines(merged), remainder


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
    if (
        letters
        and letters[0].isupper()
        and letters[1:].islower()
        and head.islower()
    ):
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
        *(candidate for candidate in (source.get("attrs"), source.get("block_attrs")) if isinstance(candidate, dict)),
    )
    return any(candidate.get("list_kind") for candidate in nested if isinstance(candidate, dict))


def _should_merge_blocks(curr: Block, nxt: Block) -> Tuple[bool, str]:
    curr_text = curr.text.strip()
    next_text = nxt.text.strip()

    if not curr_text or not next_text:
        return False, "empty_text"

    curr_page = curr.source.get("page")
    next_page = nxt.source.get("page")

    if next_text.startswith("—"):
        return True, "author_attribution"

    if is_bullet_continuation(curr_text, next_text):
        return True, "bullet_continuation"

    if is_bullet_fragment(curr_text, next_text):
        return True, "bullet_fragment"

    last_line = _last_non_empty_line(curr_text)
    if (
        starts_with_bullet(last_line)
        and not starts_with_bullet(next_text)
        and len(next_text.split()) <= 3
    ):
        return True, "bullet_short_fragment"

    colon_intro = curr_text.rstrip().endswith(":") and not starts_with_bullet(curr_text)
    if is_bullet_list_pair(curr_text, next_text):
        if colon_intro and not _block_has_list_markers(curr):
            return False, "colon_intro_without_list_markers"
        return True, "bullet_list"

    if is_numbered_list_pair(curr_text, next_text):
        return True, "numbered_list"

    if is_numbered_continuation(curr_text, next_text) and not _is_heading_like(next_text):
        return True, "numbered_continuation"

    if re.fullmatch(r"\d+[.)]", curr_text) and not starts_with_number(next_text):
        return True, "numbered_standalone"

    if re.search(r"\n\d+[.)]\s*$", curr_text) and not starts_with_number(next_text):
        return True, "numbered_suffix"

    curr_has_quote = '"' in curr_text or "'" in curr_text
    next_has_quote = '"' in next_text or "'" in next_text
    if curr_has_quote or next_has_quote:
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
    elif _is_same_page_continuation(curr_text, next_text, curr_page, next_page):
        return True, "sentence_continuation"
    elif _is_cross_page_continuation(curr_text, next_text, curr_page, next_page):
        return True, "sentence_continuation"
    elif _is_cross_page_paragraph_continuation(curr, nxt):
        return True, "sentence_continuation"

    return False, "no_merge"


def within_page_span(pages: Iterable[Optional[int]], limit: int = 1) -> bool:
    """Return True if pages stay within ``limit`` span."""
    nums = [p for p in pages if p is not None]
    return not nums or max(nums) - min(nums) <= limit


def merge_continuation_blocks(blocks: Iterable[Block]) -> Iterable[Block]:
    items = list(blocks)
    if not items:
        return []

    merged: list[Block] = []
    i = 0
    while i < len(items):
        current = items[i]
        current_text = current.text.strip()
        pages = [current.source.get("page")]
        j = i + 1
        merged_any = False
        while j < len(items):
            nxt = items[j]
            next_text = nxt.text.strip()
            next_page = nxt.source.get("page")
            candidate_pages = [*pages, next_page]
            if any(
                b is not None and a is not None and b - a > 1
                for a, b in zip(candidate_pages, candidate_pages[1:])
            ) or not within_page_span(candidate_pages):
                break
            curr_for_merge = replace(current, source={**current.source, "page": pages[-1]})
            should_merge, reason = _should_merge_blocks(curr_for_merge, nxt)
            if should_merge:
                if reason == "hyphenated_continuation":
                    normalized_tail = _normalize_hyphenated_tail(current_text, next_text)
                    merged_text = re.sub(rf"[{HYPHEN_CHARS_ESC}]$", "", current_text) + normalized_tail
                elif reason == "sentence_continuation":
                    normalized_sentence = _normalize_sentence_tail(current_text, next_text)
                    merged_text = current_text + " " + normalized_sentence
                elif reason.startswith("bullet_"):
                    merged_text, remainder = _merge_bullet_text(reason, current_text, next_text)
                    current = replace(current, text=merged_text)
                    current_text = merged_text
                    merged_any = True
                    if remainder:
                        items[j] = replace(nxt, text=remainder.lstrip())
                        pages.append(next_page)
                        break
                    j += 1
                    pages.append(next_page)
                    continue
                elif reason == "numbered_list":
                    merged_text = current_text + "\n" + next_text
                elif reason == "numbered_continuation":
                    merged_text = current_text + " " + next_text
                elif reason == "numbered_standalone":
                    processed = insert_numbered_list_newlines(next_text)
                    merged_text = current_text + " " + processed
                elif reason == "numbered_suffix":
                    marker_match = re.search(r"\n(\d+[.)])\s*$", current_text)
                    base = re.sub(r"\n\d+[.)]\s*$", "", current_text)
                    processed = insert_numbered_list_newlines(next_text)
                    marker = marker_match.group(1) if marker_match else ""
                    prefix = f"{marker} " if marker else ""
                    merged_text = f"{base}\n{prefix}{processed}".strip()
                elif reason == "indented_continuation":
                    merged_text = current_text + "\n" + next_text
                elif reason == "author_attribution":
                    merged_text = current_text + " " + next_text
                else:
                    merged_text = current_text + " " + next_text

                current = replace(current, text=merged_text)
                current_text = merged_text
                merged_any = True
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
        merged.append(current)
        i = j if merged_any else i + 1
    return merged
