# Footer Detection Enhancement Plan

## Status: IMPLEMENTED ✅

The geometric footer detection approach has been implemented. See usage examples below.

## Implementation Summary

### New CLI Options
```bash
# Auto-detect footer/header zones using geometry
pdf_chunker convert input.pdf --auto-detect-zones --out output.jsonl

# Manually specify footer margin (in points from page bottom)
pdf_chunker convert input.pdf --footer-margin 40 --out output.jsonl

# Specify header margin (in points from page top)
pdf_chunker convert input.pdf --header-margin 50 --out output.jsonl

# Combine with interactive mode for text-pattern fallback
pdf_chunker convert input.pdf --auto-detect-zones --interactive --out output.jsonl
```

### Key Files Modified/Added
- `pdf_chunker/geometry.py` (new) - Geometric zone detection
- `pdf_chunker/pdf_blocks.py` - Zone filtering at extraction time
- `pdf_chunker/pdf_parsing.py` - Zone margin parameter threading
- `pdf_chunker/adapters/io_pdf.py` - Zone margin support in adapter
- `pdf_chunker/core_new.py` - Zone config propagation
- `pdf_chunker/cli.py` - New CLI options

---

## Original Problems Identified

### 1. Text-Based Heuristic Issues
- **Context bleeding**: Footer candidates sometimes include preceding text (e.g., "Assume the\nCollective Wisdom from the Experts 13")
- **Text concatenation bugs**: Sometimes footer text gets merged with preceding content (e.g., "somecollective Wisdom from the Experts 31" where 'C' became lowercase)
- **False positives**: Heuristics can match legitimate content that happens to look like a footer

### 2. Root Cause Analysis
The text-based approach has fundamental limitations:
- It works on already-extracted text, which may have lost positional information
- Text cleaning passes may inadvertently merge or alter footer text
- Pattern matching is inherently imprecise without positional context

## Solution: Geometric Footer Detection

### Key Insight
PDF footers are typically at **consistent positions** on every page. Analysis of `97things-manager-short.pdf` shows:
- All footers at y=612.9 on 648pt pages
- Consistent 35.1pt margin from bottom
- This positional consistency is a much stronger signal than text patterns

### Two-Phase Architecture

#### Phase 1: Footer Zone Discovery (Interactive)
1. **Analyze page geometry** across all pages
2. **Identify candidate footer zones** based on:
   - Text blocks in bottom N% of page (configurable, default 10%)
   - Consistency of content type across pages (e.g., "Title + PageNumber" pattern)
   - Font size/style consistency (footers often use smaller or different fonts)
3. **Interactive confirmation**:
   - Present discovered zones to user
   - User confirms which zones contain headers/footers
   - System learns the precise Y coordinates to exclude

#### Phase 2: Extraction with Zone Exclusion
1. **Apply learned zones** during PDF extraction
2. **Skip blocks** within header/footer zones entirely
3. **No text-pattern matching needed** - purely positional

### Implementation Components

#### 1. `pdf_chunker/geometry.py` (new module)
```python
@dataclass
class PageZone:
    """A rectangular region on a page."""
    x0: float
    y0: float
    x1: float
    y1: float
    zone_type: str  # 'header', 'footer', 'margin'

@dataclass
class DocumentZones:
    """Zones to exclude during extraction."""
    header_margin: float | None = None  # points from top
    footer_margin: float | None = None  # points from bottom
    left_margin: float | None = None
    right_margin: float | None = None
    
def detect_footer_zone(doc: fitz.Document) -> float | None:
    """Analyze document to find consistent footer position."""
    ...

def detect_header_zone(doc: fitz.Document) -> float | None:
    """Analyze document to find consistent header position."""
    ...
```

#### 2. Interactive Zone Discovery CLI
```bash
# New command to discover and confirm zones
pdf_chunker discover-zones input.pdf --interactive

# Output: zone configuration that can be reused
# zones.yaml:
# header_margin: 50
# footer_margin: 40
```

#### 3. Integration with Extraction
```bash
# Use discovered zones during conversion
pdf_chunker convert input.pdf --zones zones.yaml --out output.jsonl

# Or inline specification
pdf_chunker convert input.pdf --footer-margin 40 --out output.jsonl
```

### Fallback Strategy

When geometric detection fails or is ambiguous:
1. **Fall back to text-based heuristics**
2. **Use interactive mode** to confirm uncertain cases
3. **Learn from user decisions** to improve future detection

### Benefits

1. **Higher accuracy**: Positional detection is more reliable than text patterns
2. **Simpler logic**: No complex regex patterns needed
3. **Faster processing**: Zone exclusion happens at extraction time
4. **Consistent results**: Same zones apply to all pages uniformly
5. **User control**: Interactive discovery gives users visibility and control

## Migration Path

### Phase 1: Add Geometric Detection (MVP)
1. Implement `geometry.py` module
2. Add `--footer-margin` / `--header-margin` CLI options
3. Integrate with `pdf_parse` pass to exclude zones during extraction

### Phase 2: Interactive Discovery
1. Add `discover-zones` command
2. Implement zone analysis algorithms
3. Add YAML zone configuration support

### Phase 3: Hybrid Approach
1. Use geometric detection as primary
2. Fall back to text heuristics for edge cases
3. Interactive confirmation for ambiguous cases

## Relation to Existing Features

### Current `--footer-pattern` Flag
- **Keep it**: Still useful for documents where geometric detection fails
- **Lower priority**: Check after geometric exclusion
- **Complementary**: Can catch footers that slipped through zone exclusion

### Current `--interactive` Flag
- **Elevate importance**: Make it the primary discovery mechanism
- **Two modes**:
  1. Zone discovery mode (new)
  2. Pattern confirmation mode (current)

## Technical Notes

### PyMuPDF Capabilities
- `page.get_text('dict')` returns blocks with `bbox` coordinates
- `page.rect` gives page dimensions
- Can filter blocks by position before text extraction

### Coordinate System
- PyMuPDF uses points (72 points = 1 inch)
- Origin (0,0) is top-left
- Y increases downward

### Typical Footer Margins
- Books: 30-50 points
- Academic papers: 40-60 points
- Technical docs: 25-40 points

## Success Criteria

1. **Zero false positives** with geometric detection on well-formatted PDFs
2. **< 10% false positive rate** in interactive mode for edge cases
3. **Backward compatible** with existing `--footer-pattern` usage
4. **Discoverable** through `--interactive` or `discover-zones` command
