"""Geometric zone detection for headers and footers.

This module provides position-based detection of page elements like headers
and footers. Unlike text-pattern matching, geometric detection uses the
consistent positioning of these elements across pages.

Key insight: In professionally formatted PDFs, headers and footers appear at
nearly identical Y coordinates on every page. This positional consistency is
a much stronger signal than text patterns.

Usage:
    from pdf_chunker.geometry import detect_footer_zone, DocumentZones
    
    # Automatic detection
    zones = detect_document_zones(doc)
    
    # Manual specification
    zones = DocumentZones(footer_margin=40)  # exclude bottom 40 points
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Try to import fitz, but allow module to load without it for testing
try:
    import fitz
except ImportError:
    fitz = None  # type: ignore


@dataclass(frozen=True)
class TextBlock:
    """A positioned text block from a PDF page."""
    
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int
    font_size: float = 0.0
    
    @property
    def height(self) -> float:
        """Block height in points."""
        return self.y1 - self.y0
    
    @property
    def width(self) -> float:
        """Block width in points."""
        return self.x1 - self.x0
    
    def distance_from_bottom(self, page_height: float) -> float:
        """Distance from block bottom to page bottom."""
        return page_height - self.y1
    
    def distance_from_top(self, page_height: float) -> float:
        """Distance from block top to page top."""
        return self.y0


@dataclass
class DocumentZones:
    """Configuration for zones to exclude during extraction.
    
    All measurements are in points (72 points = 1 inch).
    
    Attributes:
        header_margin: Points from top to exclude (None = no header exclusion)
        footer_margin: Points from bottom to exclude (None = no footer exclusion)
        left_margin: Points from left to exclude (None = no left exclusion)
        right_margin: Points from right to exclude (None = no right exclusion)
        detected_footer_y: The Y coordinate where footers were detected
        detected_header_y: The Y coordinate where headers were detected
        confidence: Detection confidence (0.0 to 1.0)
    """
    
    header_margin: float | None = None
    footer_margin: float | None = None
    left_margin: float | None = None
    right_margin: float | None = None
    detected_footer_y: float | None = field(default=None, repr=False)
    detected_header_y: float | None = field(default=None, repr=False)
    confidence: float = field(default=0.0, repr=False)
    
    def is_in_footer(self, block: TextBlock, page_height: float) -> bool:
        """Check if block is within the footer zone."""
        if self.footer_margin is None:
            return False
        return block.distance_from_bottom(page_height) < self.footer_margin
    
    def is_in_header(self, block: TextBlock, page_height: float) -> bool:
        """Check if block is within the header zone."""
        if self.header_margin is None:
            return False
        return block.y0 < self.header_margin
    
    def should_exclude(self, block: TextBlock, page_height: float) -> bool:
        """Check if block should be excluded based on zones."""
        return self.is_in_footer(block, page_height) or self.is_in_header(block, page_height)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            k: v for k, v in {
                "header_margin": self.header_margin,
                "footer_margin": self.footer_margin,
                "left_margin": self.left_margin,
                "right_margin": self.right_margin,
            }.items() if v is not None
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentZones:
        """Create from dictionary."""
        return cls(
            header_margin=data.get("header_margin"),
            footer_margin=data.get("footer_margin"),
            left_margin=data.get("left_margin"),
            right_margin=data.get("right_margin"),
        )


def _extract_blocks_from_page(page: Any, page_num: int) -> list[TextBlock]:
    """Extract text blocks with positions from a PyMuPDF page."""
    blocks = []
    try:
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if "lines" not in block:
                continue
            
            # Combine text from all spans
            text_parts = []
            font_sizes = []
            for line in block["lines"]:
                for span in line.get("spans", []):
                    text_parts.append(span.get("text", ""))
                    font_sizes.append(span.get("size", 0))
            
            text = " ".join(text_parts).strip()
            if not text:
                continue
            
            bbox = block["bbox"]
            avg_font_size = statistics.mean(font_sizes) if font_sizes else 0
            
            blocks.append(TextBlock(
                text=text,
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
                page=page_num,
                font_size=avg_font_size,
            ))
    except Exception:
        pass  # Gracefully handle extraction failures
    
    return blocks


def _get_bottom_blocks(
    blocks: Sequence[TextBlock],
    page_height: float,
    margin_percent: float = 0.10,
) -> list[TextBlock]:
    """Get blocks in the bottom portion of the page."""
    threshold = page_height * (1 - margin_percent)
    return [b for b in blocks if b.y1 > threshold]


def _get_top_blocks(
    blocks: Sequence[TextBlock],
    page_height: float,
    margin_percent: float = 0.10,
) -> list[TextBlock]:
    """Get blocks in the top portion of the page."""
    threshold = page_height * margin_percent
    return [b for b in blocks if b.y0 < threshold]


def _analyze_zone_consistency(
    y_positions: Sequence[float],
    tolerance: float = 5.0,
) -> tuple[float | None, float]:
    """Analyze Y positions for consistency.
    
    Returns:
        Tuple of (most_common_y, confidence) where confidence is 0.0-1.0
    """
    if not y_positions:
        return None, 0.0
    
    # Round positions to nearest point for grouping
    rounded = [round(y) for y in y_positions]
    counter = Counter(rounded)
    
    if not counter:
        return None, 0.0
    
    most_common_y, count = counter.most_common(1)[0]
    confidence = count / len(y_positions)
    
    return float(most_common_y), confidence


def _analyze_all_zone_candidates(
    y_positions: Sequence[float],
    texts_by_position: dict[int, list[str]],
    min_count: int = 2,
) -> list[tuple[float, float, list[str]]]:
    """Analyze Y positions and return ALL candidates sorted by frequency.
    
    Returns:
        List of (y_position, confidence, sample_texts) tuples, sorted by confidence descending.
    """
    if not y_positions:
        return []
    
    # Round positions to nearest point for grouping
    rounded = [round(y) for y in y_positions]
    counter = Counter(rounded)
    
    if not counter:
        return []
    
    total = len(y_positions)
    candidates = []
    
    for y_pos, count in counter.most_common():
        if count >= min_count:
            confidence = count / total
            texts = texts_by_position.get(y_pos, [])
            candidates.append((float(y_pos), confidence, texts))
    
    return candidates


def detect_footer_zone(
    doc: Any,
    sample_pages: int = 10,
    min_confidence: float = 0.6,
    exclude_pages: set[int] | None = None,
) -> tuple[float | None, float]:
    """Detect consistent footer zone across document pages.
    
    Analyzes the bottom portion of pages to find consistently positioned
    text blocks that are likely footers.
    
    Args:
        doc: PyMuPDF document object
        sample_pages: Number of pages to sample (0 = all pages)
        min_confidence: Minimum confidence threshold
        exclude_pages: Set of 1-based page numbers to exclude (e.g., cover, TOC)
    
    Returns:
        Tuple of (footer_margin, confidence) where footer_margin is the
        distance from page bottom to exclude, or (None, 0.0) if no
        consistent footer zone found.
    """
    if fitz is None:
        return None, 0.0
    
    exclude_pages = exclude_pages or set()
    num_pages = len(doc)
    
    # Build list of pages to check, respecting exclusions
    pages_to_check = [
        i for i in range(num_pages)
        if (i + 1) not in exclude_pages  # i is 0-indexed, exclude_pages is 1-indexed
    ]
    
    # Limit sample size
    if sample_pages > 0 and len(pages_to_check) > sample_pages:
        pages_to_check = pages_to_check[:sample_pages]
    
    footer_y_positions: list[float] = []
    page_heights: list[float] = []
    
    for i in pages_to_check:
        page = doc[i]
        page_height = page.rect.height
        page_heights.append(page_height)
        
        blocks = _extract_blocks_from_page(page, i + 1)  # Pass 1-indexed page
        bottom_blocks = _get_bottom_blocks(blocks, page_height)
        
        if bottom_blocks:
            # Get the Y position of the bottommost block
            bottommost = max(bottom_blocks, key=lambda b: b.y1)
            footer_y_positions.append(bottommost.y1)
    
    if not footer_y_positions:
        return None, 0.0
    
    # Analyze consistency
    common_y, confidence = _analyze_zone_consistency(footer_y_positions)
    
    if common_y is None or confidence < min_confidence:
        return None, confidence
    
    # Calculate margin from bottom
    avg_page_height = statistics.mean(page_heights)
    footer_margin = avg_page_height - common_y + 5  # Add small buffer
    
    return footer_margin, confidence


def detect_header_zone(
    doc: Any,
    sample_pages: int = 10,
    min_confidence: float = 0.6,
    exclude_pages: set[int] | None = None,
) -> tuple[float | None, float]:
    """Detect consistent header zone across document pages.
    
    Similar to detect_footer_zone but for the top of pages.
    
    Args:
        exclude_pages: Set of 1-based page numbers to exclude (e.g., cover, TOC)
    
    Returns:
        Tuple of (header_margin, confidence) where header_margin is the
        distance from page top to exclude.
    """
    if fitz is None:
        return None, 0.0
    
    exclude_pages = exclude_pages or set()
    num_pages = len(doc)
    
    # Build list of pages to check, respecting exclusions
    pages_to_check = [
        i for i in range(num_pages)
        if (i + 1) not in exclude_pages
    ]
    
    if sample_pages > 0 and len(pages_to_check) > sample_pages:
        pages_to_check = pages_to_check[:sample_pages]
    
    header_y_positions: list[float] = []
    
    for i in pages_to_check:
        page = doc[i]
        page_height = page.rect.height
        
        blocks = _extract_blocks_from_page(page, i + 1)
        top_blocks = _get_top_blocks(blocks, page_height)
        
        if top_blocks:
            # Get the Y position of the topmost block's bottom
            topmost = min(top_blocks, key=lambda b: b.y0)
            header_y_positions.append(topmost.y1)
    
    if not header_y_positions:
        return None, 0.0
    
    # Analyze consistency
    common_y, confidence = _analyze_zone_consistency(header_y_positions)
    
    if common_y is None or confidence < min_confidence:
        return None, confidence
    
    # The header margin is from top to the bottom of the header block
    header_margin = common_y + 5  # Add small buffer
    
    return header_margin, confidence


def detect_document_zones(
    doc: Any,
    sample_pages: int = 10,
    min_confidence: float = 0.6,
    exclude_pages: set[int] | None = None,
) -> DocumentZones:
    """Detect both header and footer zones in a document.
    
    Args:
        doc: PyMuPDF document object
        sample_pages: Number of pages to sample
        min_confidence: Minimum confidence for detection
        exclude_pages: Set of 1-based page numbers to exclude (e.g., cover, TOC)
    
    Returns:
        DocumentZones with detected margins
    """
    footer_margin, footer_conf = detect_footer_zone(
        doc, sample_pages, min_confidence, exclude_pages
    )
    header_margin, header_conf = detect_header_zone(
        doc, sample_pages, min_confidence, exclude_pages
    )
    
    # Use the higher confidence as overall confidence
    overall_conf = max(footer_conf, header_conf) if (footer_margin or header_margin) else 0.0
    
    return DocumentZones(
        header_margin=header_margin,
        footer_margin=footer_margin,
        confidence=overall_conf,
    )


def iter_content_blocks(
    doc: Any,
    zones: DocumentZones | None = None,
) -> Iterator[TextBlock]:
    """Iterate over content blocks, excluding header/footer zones.
    
    Args:
        doc: PyMuPDF document object
        zones: Zone exclusion configuration (None = no exclusion)
    
    Yields:
        TextBlock objects for content (non-header/footer) blocks
    """
    if fitz is None:
        return
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        
        blocks = _extract_blocks_from_page(page, page_num)
        
        for block in blocks:
            if zones is not None and zones.should_exclude(block, page_height):
                continue
            yield block


# ---------------------------------------------------------------------------
# Interactive zone discovery
# ---------------------------------------------------------------------------


@dataclass
class ZoneCandidate:
    """A candidate zone for interactive confirmation."""
    
    zone_type: str  # 'header' or 'footer'
    margin: float
    y_position: float  # The detected Y coordinate
    confidence: float
    sample_texts: list[str]
    pages_detected: int
    total_pages: int
    
    def format_for_prompt(self, index: int = 0) -> str:
        """Format candidate for user prompt."""
        lines = [
            f"[{index}] {self.zone_type.title()} Zone Candidate",
            f"    Position: {self.y_position:.1f}pt from top ({self.margin:.1f}pt margin from {'top' if self.zone_type == 'header' else 'bottom'})",
            f"    Confidence: {self.confidence:.0%} ({self.pages_detected}/{self.total_pages} pages)",
            "    Sample content:",
        ]
        for i, text in enumerate(self.sample_texts[:3], 1):
            preview = text[:70] + "..." if len(text) > 70 else text
            # Escape newlines for display
            preview = preview.replace('\n', ' ').strip()
            lines.append(f"      • {preview}")
        return "\n".join(lines)


def _collect_zone_data(
    doc: Any,
    pages_to_check: list[int],
    zone_type: str,
    margin_percent: float = 0.15,
) -> tuple[list[float], list[float], dict[int, list[str]]]:
    """Collect Y positions and texts from specified pages for zone detection.
    
    Args:
        doc: PyMuPDF document
        pages_to_check: List of 0-indexed page numbers to analyze
        zone_type: 'header' or 'footer'
        margin_percent: Percentage of page to consider for zone
    
    Returns:
        Tuple of (y_positions, page_heights, texts_by_rounded_y)
    """
    y_positions: list[float] = []
    page_heights: list[float] = []
    texts_by_y: dict[int, list[str]] = {}
    
    for i in pages_to_check:
        page = doc[i]
        page_height = page.rect.height
        page_heights.append(page_height)
        
        blocks = _extract_blocks_from_page(page, i + 1)
        
        if zone_type == 'footer':
            zone_blocks = _get_bottom_blocks(blocks, page_height, margin_percent)
            if zone_blocks:
                # Get the bottommost block
                target = max(zone_blocks, key=lambda b: b.y1)
                y_pos = target.y1
                y_positions.append(y_pos)
                rounded = round(y_pos)
                texts_by_y.setdefault(rounded, []).append(target.text)
        else:  # header
            zone_blocks = _get_top_blocks(blocks, page_height, margin_percent)
            if zone_blocks:
                # Get the topmost block's bottom edge
                target = min(zone_blocks, key=lambda b: b.y0)
                y_pos = target.y1
                y_positions.append(y_pos)
                rounded = round(y_pos)
                texts_by_y.setdefault(rounded, []).append(target.text)
    
    return y_positions, page_heights, texts_by_y


def discover_zones_interactive(
    doc: Any,
    callback: Any = None,
    exclude_pages: set[int] | None = None,
    sample_pages: int = 20,
) -> DocumentZones:
    """Interactively discover and confirm header/footer zones.
    
    This function analyzes the document, identifies ALL potential footer/header
    candidates (not just the most common), presents them to the user, and lets
    the user choose which zone to exclude.
    
    Args:
        doc: PyMuPDF document object
        callback: Optional callback for confirmation (default: CLI prompt)
        exclude_pages: Set of 1-based page numbers to exclude (e.g., cover, TOC)
        sample_pages: Number of pages to sample (more = better detection)
    
    Returns:
        Confirmed DocumentZones configuration
    """
    if fitz is None:
        return DocumentZones()
    
    exclude_pages = exclude_pages or set()
    num_pages = len(doc)
    
    # Build list of pages to check, respecting exclusions
    pages_to_check = [
        i for i in range(num_pages)
        if (i + 1) not in exclude_pages
    ]
    
    # Limit sample size but try to get good coverage
    if len(pages_to_check) > sample_pages:
        pages_to_check = pages_to_check[:sample_pages]
    
    if not pages_to_check:
        print("No pages to analyze (all pages excluded)")
        return DocumentZones()
    
    print(f"\nAnalyzing {len(pages_to_check)} pages for zone detection...")
    if exclude_pages:
        print(f"(Excluding pages: {sorted(exclude_pages)})")
    
    # Collect footer data
    footer_y, page_heights, footer_texts_by_y = _collect_zone_data(
        doc, pages_to_check, 'footer'
    )
    
    result = DocumentZones()
    avg_height = statistics.mean(page_heights) if page_heights else 0
    
    # Get ALL footer candidates, not just the best one
    footer_candidates = _analyze_all_zone_candidates(
        footer_y, footer_texts_by_y, min_count=2
    )
    
    if footer_candidates:
        print(f"\nFound {len(footer_candidates)} potential footer zone(s):\n")
        
        zone_options: list[ZoneCandidate] = []
        for i, (y_pos, conf, texts) in enumerate(footer_candidates):
            margin = avg_height - y_pos + 5
            candidate = ZoneCandidate(
                zone_type="footer",
                margin=margin,
                y_position=y_pos,
                confidence=conf,
                sample_texts=texts,
                pages_detected=int(conf * len(pages_to_check)),
                total_pages=len(pages_to_check),
            )
            zone_options.append(candidate)
            print(candidate.format_for_prompt(i + 1))
            print()
        
        # Ask user to select
        if callback is not None:
            selection = callback(zone_options)
        else:
            print("Enter the number of the zone to exclude, or 0 to skip footer exclusion.")
            try:
                response = input("Your choice [1]: ").strip()
                if response == "" or response == "1":
                    selection = 0  # First candidate (index 0)
                elif response == "0":
                    selection = -1  # Skip
                else:
                    selection = int(response) - 1
            except (ValueError, EOFError):
                selection = 0  # Default to first
        
        if 0 <= selection < len(zone_options):
            chosen = zone_options[selection]
            print(f"\n✓ Footer zone confirmed: {chosen.margin:.1f}pt margin from bottom")
            result = DocumentZones(
                footer_margin=chosen.margin,
                detected_footer_y=chosen.y_position,
                confidence=chosen.confidence,
            )
        else:
            print("\n✗ Footer zone detection skipped")
    else:
        print("\nNo consistent footer zone detected across sampled pages.")
    
    # TODO: Add header detection with same interactive approach
    
    return result
