"""In-memory contracts for geometric selection, scoring, and presentation."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pdf_chunker import geometry
from pdf_chunker.geometry import (
    DocumentZones,
    TextBlock,
    ZoneCandidate,
    _analyze_all_zone_candidates,
    _analyze_zone_consistency,
    _extract_blocks_from_page,
    _get_bottom_blocks,
    _get_top_blocks,
    detect_document_zones,
    detect_footer_zone,
    detect_header_zone,
    iter_content_blocks,
)


def _block(y0: float, y1: float) -> TextBlock:
    return TextBlock("Body.", 10.0, y0, 30.0, y1, page=2)


def test_positioned_block_dimensions_and_distances() -> None:
    block = _block(20.0, 35.0)

    assert (block.width, block.height) == (20.0, 15.0)
    assert block.distance_from_bottom(100.0) == 65.0
    assert block.distance_from_top(100.0) == 20.0


@pytest.mark.parametrize("zone", ("header", "footer"))
@pytest.mark.parametrize("distance, excluded", ((9.0, True), (10.0, False), (11.0, False)))
def test_zone_margins_are_strict(zone: str, distance: float, excluded: bool) -> None:
    zones = DocumentZones(**{f"{zone}_margin": 10.0})
    block = _block(distance, 100.0 - distance)

    assert zones.should_exclude(block, 100.0) is excluded


@pytest.mark.parametrize("header, footer", ((None, None), (10.0, None), (None, 10.0), (10, 10)))
def test_exclusion_is_the_union_of_header_and_footer(
    header: float | None, footer: float | None
) -> None:
    zones = DocumentZones(header_margin=header, footer_margin=footer)

    assert zones.should_exclude(_block(5.0, 50.0), 100.0) is (header is not None)
    assert zones.should_exclude(_block(50.0, 95.0), 100.0) is (footer is not None)
    assert not zones.should_exclude(_block(20.0, 80.0), 100.0)


@pytest.mark.parametrize(
    "settings",
    ({}, {"footer_margin": 0.0}, {"header_margin": 10.0, "left_margin": 5.0, "right_margin": 4.0}),
)
def test_zone_serialization_preserves_margins_but_not_detection_telemetry(
    settings: dict[str, float],
) -> None:
    zones = DocumentZones(**settings, detected_footer_y=80.0, confidence=0.9)

    assert zones.to_dict() == settings
    assert DocumentZones.from_dict(zones.to_dict()).to_dict() == settings
    assert zones.detected_footer_y == 80.0
    assert zones.confidence == 0.9


def test_side_margins_are_serialized_but_do_not_filter_blocks() -> None:
    zones = DocumentZones(left_margin=100.0, right_margin=100.0)

    assert not zones.should_exclude(_block(0.0, 100.0), 100.0)


@pytest.mark.parametrize("select", (_get_top_blocks, _get_bottom_blocks))
@pytest.mark.parametrize("margin", (0.0, 0.1, 0.2))
def test_partially_applied_zone_selection_preserves_order_and_inputs(
    select: Callable[..., list[TextBlock]], margin: float
) -> None:
    blocks = (_block(90, 95), _block(0, 5), _block(10, 90), _block(5, 99))
    before = deepcopy(blocks)
    stage = partial(select, page_height=100.0, margin_percent=margin)
    expected = [
        block
        for block in blocks
        if (
            block.y0 < 100.0 * margin
            if select is _get_top_blocks
            else block.y1 > 100.0 * (1 - margin)
        )
    ]

    assert stage(blocks) == expected
    assert blocks == before


@pytest.mark.parametrize(
    "positions, expected",
    (
        ([], (None, 0.0)),
        ([20.1, 20.4, 21.1], (20.0, 2 / 3)),
        ([20.5, 21.5], (20.0, 0.5)),
        ([30.0, 20.0], (30.0, 0.5)),
        ([-1.5, -2.5], (-2.0, 1.0)),
    ),
)
def test_consistency_uses_rounded_bins_and_first_seen_ties(
    positions: list[float], expected: tuple[float | None, float]
) -> None:
    before = positions.copy()

    assert _analyze_zone_consistency(positions) == expected
    assert _analyze_zone_consistency(positions, tolerance=100.0) == expected
    assert positions == before


@pytest.mark.parametrize("min_count", (0, 1, 2, 3))
def test_candidates_keep_frequency_order_ties_and_sample_lists(min_count: int) -> None:
    positions = [20.1, 30.2, 20.4, 30.4, 40.0]
    texts = {20: ["Twenty"], 30: ["Thirty"], 99: ["Unrelated"]}
    before = deepcopy((positions, texts))
    expected = [
        (position, count / 5, texts.get(int(position), []))
        for position, count in ((20.0, 2), (30.0, 2), (40.0, 1))
        if count >= min_count
    ]

    result = _analyze_all_zone_candidates(positions, texts, min_count=min_count)

    assert result == expected
    assert (positions, texts) == before
    if result:
        assert result[0][2] is texts[20]  # Existing samples are borrowed, not copied.
    assert _analyze_all_zone_candidates([], texts, min_count=min_count) == []


@given(
    positions=st.lists(st.integers(-40, 40).map(lambda value: value / 2), max_size=20),
    min_count=st.integers(0, 5),
)
def test_candidate_scoring_matches_stable_frequency_ranking(
    positions: list[float], min_count: int
) -> None:
    rounded = [round(position) for position in positions]
    first_seen = tuple(dict.fromkeys(rounded))
    ranked = sorted(first_seen, key=rounded.count, reverse=True)
    texts = {position: [str(position)] for position in first_seen}
    before = deepcopy((positions, texts))
    stage = partial(_analyze_all_zone_candidates, texts_by_position=texts)
    expected = [
        (float(position), rounded.count(position) / len(rounded), texts[position])
        for position in ranked
        if rounded.count(position) >= min_count
    ]

    result = stage(positions, min_count=min_count)

    assert result == expected
    assert stage(positions, min_count=min_count + 1) == [
        candidate for candidate in result if rounded.count(int(candidate[0])) >= min_count + 1
    ]
    assert (positions, texts) == before


@pytest.mark.parametrize("zone, edge", (("header", "top"), ("footer", "bottom")))
@pytest.mark.parametrize(
    "samples, previews",
    (
        ([], []),
        (["short\ncaption"], ["short caption"]),
        (["x" * 70], ["x" * 70]),
        (["x" * 71], ["x" * 70 + "..."]),
        (["  \n ", "one", "two", "ignored"], ["", "one", "two"]),
        (["\n" * 71], ["..."]),
    ),
)
def test_candidate_prompt_is_exact_bounded_and_input_preserving(
    zone: str, edge: str, samples: list[str], previews: list[str]
) -> None:
    candidate = ZoneCandidate(zone, 25.16, 80.24, 2 / 3, samples, 2, 3)
    before = deepcopy(candidate)
    expected = "\n".join(
        (
            f"[4] {zone.title()} Zone Candidate",
            f"    Position: 80.2pt from top (25.2pt margin from {edge})",
            "    Confidence: 67% (2/3 pages)",
            "    Sample content:",
            *(f"      • {preview}" for preview in previews),
        )
    )

    assert candidate.format_for_prompt(4) == expected
    assert candidate.format_for_prompt(4) == expected
    assert candidate == before


@dataclass(frozen=True)
class _Rect:
    height: float = 100.0


@dataclass(frozen=True)
class _Page:
    blocks: tuple[dict[str, Any], ...]
    rect: _Rect = _Rect()

    def get_text(self, mode: str) -> dict[str, Any]:
        assert mode == "dict"
        return {"blocks": self.blocks}


def _page(header: float = 8.0, footer: float = 95.0) -> _Page:
    return _Page(
        tuple(
            {
                "bbox": (0, y0, 20, y1),
                "lines": [{"spans": [{"text": text, "size": 10.0}]}],
            }
            for y0, y1, text in ((2.0, header, "Header"), (40, 60, "Body"), (92, footer, "Footer"))
        )
    )


@pytest.mark.parametrize("detect, margin", ((detect_footer_zone, 7.0), (detect_header_zone, 14.0)))
def test_detection_samples_after_one_based_exclusions_without_mutation(
    detect: Callable[..., tuple[float | None, float]], margin: float
) -> None:
    pages = (_page(7, 95), _page(9, 98), _page(6, 94))
    exclusions = {1}
    before = deepcopy((pages, exclusions))

    assert detect(pages, sample_pages=1, exclude_pages=exclusions) == (margin, 1.0)
    assert detect(pages, exclude_pages={1, 2, 3}) == (None, 0.0)
    assert (pages, exclusions) == before


@pytest.mark.parametrize("detect", (detect_header_zone, detect_footer_zone))
def test_detection_returns_confidence_even_when_below_threshold(
    detect: Callable[..., tuple[float | None, float]],
) -> None:
    pages = (_page(8, 95), _page(8, 95), _page(9, 96))

    margin, confidence = detect(pages, sample_pages=0, min_confidence=2 / 3)
    assert margin is not None
    assert confidence == 2 / 3
    assert detect(pages, sample_pages=0, min_confidence=0.7) == (None, 2 / 3)


def test_document_zones_and_content_iterator_preserve_page_block_order() -> None:
    pages = (_page(), _page())
    before = deepcopy(pages)
    zones = detect_document_zones(pages)

    assert zones.to_dict() == {"header_margin": 13.0, "footer_margin": 10.0}
    assert zones.confidence == 1.0
    assert [(block.page, block.text) for block in iter_content_blocks(pages, zones)] == [
        (0, "Body"),
        (1, "Body"),
    ]
    assert [block.text for block in iter_content_blocks(pages)] == ["Header", "Body", "Footer"] * 2
    assert pages == before


def test_extraction_preserves_completed_blocks_when_a_later_block_is_malformed() -> None:
    page = _Page((_page().blocks[0], {"lines": [{}]}, _page().blocks[1]))
    before = deepcopy(page)

    # Empty text is skipped before bbox access; nonempty malformed blocks stop extraction.
    assert [block.text for block in _extract_blocks_from_page(page, 3)] == ["Header", "Body"]
    malformed = {"lines": [{"spans": [{"text": "Broken"}]}]}
    result = _extract_blocks_from_page(_Page((page.blocks[0], malformed, page.blocks[-1])), 3)
    assert [(block.text, block.page) for block in result] == [("Header", 3)]
    assert page == before


def test_unavailable_backend_keeps_legacy_noop_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(geometry, "fitz", None)
    invalid_doc = object()  # Must not be inspected when the backend is unavailable.

    assert detect_footer_zone(invalid_doc) == (None, 0.0)
    assert detect_header_zone(invalid_doc) == (None, 0.0)
    assert detect_document_zones(invalid_doc) == DocumentZones()
    assert list(iter_content_blocks(invalid_doc)) == []
    assert geometry.discover_zones_interactive(invalid_doc) == DocumentZones()
