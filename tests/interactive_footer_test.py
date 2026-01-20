"""Tests for interactive footer confirmation module."""

from pdf_chunker.interactive import (
    FooterConfig,
    FooterDecisionCache,
    classify_footer,
    make_batch_footer_prompt,
)


class TestFooterConfig:
    """Tests for FooterConfig dataclass."""

    def test_from_dict_empty(self):
        config = FooterConfig.from_dict({})
        assert config.known_patterns == ()
        assert config.never_patterns == ()
        assert config.callback is None

    def test_from_dict_with_patterns(self):
        config = FooterConfig.from_dict(
            {
                "known_footer_patterns": [r"Book Title.*\d+", r"Page \d+"],
                "never_footer_patterns": [r"^Chapter \d+"],
            }
        )
        assert len(config.known_patterns) == 2
        assert len(config.never_patterns) == 1
        # Patterns should match case-insensitively
        assert config.known_patterns[0].search("book title 123")
        assert config.never_patterns[0].search("Chapter 5")

    def test_from_dict_filters_non_strings(self):
        config = FooterConfig.from_dict(
            {
                "known_footer_patterns": ["valid", None, 123, "also valid"],
            }
        )
        assert len(config.known_patterns) == 2


class TestFooterDecisionCache:
    """Tests for FooterDecisionCache."""

    def test_cache_miss_returns_none(self):
        cache = FooterDecisionCache()
        assert cache.get("some text") is None

    def test_cache_stores_and_retrieves(self):
        cache = FooterDecisionCache()
        cache.set("Collective Wisdom 42", True)
        assert cache.get("Collective Wisdom 42") is True
        cache.set("Chapter 1", False)
        assert cache.get("Chapter 1") is False

    def test_normalizes_page_numbers(self):
        cache = FooterDecisionCache()
        cache.set("Book Title 159", True)
        # Should match regardless of trailing page number
        assert cache.get("Book Title 160") is True
        assert cache.get("Book Title 1") is True

    def test_normalizes_whitespace(self):
        cache = FooterDecisionCache()
        cache.set("  Book   Title   42  ", True)
        assert cache.get("Book Title 99") is True


class TestClassifyFooter:
    """Tests for classify_footer function."""

    def test_known_pattern_match(self):
        config = FooterConfig.from_dict(
            {
                "known_footer_patterns": [r"Collective Wisdom.*\d+"],
            }
        )
        is_footer, reason = classify_footer(
            "Collective Wisdom from the Experts 159",
            page=159,
            config=config,
        )
        assert is_footer is True
        assert "known footer pattern" in reason

    def test_never_pattern_match(self):
        config = FooterConfig.from_dict(
            {
                "never_footer_patterns": [r"^Chapter \d+"],
            }
        )
        is_footer, reason = classify_footer(
            "Chapter 5: The Beginning",
            page=5,
            config=config,
        )
        assert is_footer is False
        assert "never-footer pattern" in reason

    def test_cached_decision(self):
        config = FooterConfig.from_dict({})
        cache = FooterDecisionCache()
        cache.set("My Footer Text 42", True)

        is_footer, reason = classify_footer(
            "My Footer Text 99",
            page=99,
            config=config,
            cache=cache,
        )
        assert is_footer is True
        assert "cached" in reason

    def test_high_confidence_heuristic(self):
        config = FooterConfig(confidence_threshold=0.7)
        is_footer, reason = classify_footer(
            "Some text",
            page=1,
            config=config,
            heuristic_confidence=0.9,
        )
        assert is_footer is True
        assert "heuristic confidence" in reason

    def test_low_confidence_fallback(self):
        config = FooterConfig(confidence_threshold=0.7)
        is_footer, reason = classify_footer(
            "Some text",
            page=1,
            config=config,
            heuristic_confidence=0.3,
        )
        assert is_footer is False
        assert "fallback" in reason

    def test_callback_invoked_for_ambiguous(self):
        decisions = []

        def callback(text, page, ctx):
            decisions.append((text, page))
            return True

        config = FooterConfig(
            callback=callback,
            confidence_threshold=0.9,
        )
        is_footer, reason = classify_footer(
            "Ambiguous text",
            page=42,
            config=config,
            heuristic_confidence=0.5,  # Below threshold
        )
        assert is_footer is True
        assert "user confirmation" in reason
        assert decisions == [("Ambiguous text", 42)]

    def test_callback_result_cached(self):
        call_count = [0]

        def callback(text, page, ctx):
            call_count[0] += 1
            return True

        config = FooterConfig(
            callback=callback,
            confidence_threshold=0.9,
            cache_decisions=True,
        )
        cache = FooterDecisionCache()

        # First call should invoke callback
        classify_footer("Footer 1", page=1, config=config, cache=cache, heuristic_confidence=0.5)
        assert call_count[0] == 1

        # Second call with similar text should use cache
        classify_footer("Footer 2", page=2, config=config, cache=cache, heuristic_confidence=0.5)
        assert call_count[0] == 1  # Not incremented


class TestBatchFooterPrompt:
    """Tests for make_batch_footer_prompt helper."""

    def test_exact_match(self):
        callback = make_batch_footer_prompt(
            {
                "Book Title 159": True,
                "Chapter 1": False,
            }
        )
        assert callback("Book Title 159", 159, {}) is True
        assert callback("Chapter 1", 1, {}) is False

    def test_substring_match(self):
        callback = make_batch_footer_prompt(
            {
                "Wisdom": True,
            }
        )
        assert callback("Collective Wisdom from Experts", 1, {}) is True

    def test_fallback_to_heuristic(self):
        callback = make_batch_footer_prompt({})
        assert callback("Unknown text", 1, {"heuristic_confidence": 0.8}) is True
        assert callback("Unknown text", 1, {"heuristic_confidence": 0.3}) is False


class TestDetectPageArtifactsPassIntegration:
    """Integration tests for footer patterns in detect_page_artifacts pass."""

    def test_pass_strips_known_footer_pattern(self):
        from pdf_chunker.framework import Artifact
        from pdf_chunker.passes.detect_page_artifacts import _DetectPageArtifactsPass

        pass_obj = _DetectPageArtifactsPass(
            known_footer_patterns=(r"Collective Wisdom.*\d+",),
        )

        doc = {
            "type": "page_blocks",
            "pages": [
                {
                    "page": 159,
                    "blocks": [
                        {"text": "Regular paragraph content."},
                        {"text": "Collective Wisdom from the Experts 159"},
                    ],
                }
            ],
        }
        result = pass_obj(Artifact(payload=doc))
        blocks = result.payload["pages"][0]["blocks"]
        # Footer block should be removed
        assert len(blocks) == 1
        assert blocks[0]["text"] == "Regular paragraph content."

    def test_pass_preserves_non_footer_blocks(self):
        from pdf_chunker.framework import Artifact
        from pdf_chunker.passes.detect_page_artifacts import _DetectPageArtifactsPass

        pass_obj = _DetectPageArtifactsPass(
            known_footer_patterns=(r"Footer Pattern \d+",),
        )

        doc = {
            "type": "page_blocks",
            "pages": [
                {
                    "page": 1,
                    "blocks": [
                        {"text": "Chapter 1: Introduction"},
                        {"text": "This is body text."},
                    ],
                }
            ],
        }
        result = pass_obj(Artifact(payload=doc))
        blocks = result.payload["pages"][0]["blocks"]
        assert len(blocks) == 2


class TestTextCleanPassFooterPatterns:
    """Integration tests for footer patterns in text_clean pass."""

    def test_text_clean_strips_footer_via_pattern(self):
        from pdf_chunker.framework import Artifact
        from pdf_chunker.passes.text_clean import _TextCleanPass

        pass_obj = _TextCleanPass(
            footer_patterns=(r"Collective Wisdom.*\d+",),
        )

        doc = {
            "type": "page_blocks",
            "pages": [
                {
                    "page": 159,
                    "blocks": [
                        {"text": "Regular paragraph content."},
                        {"text": "Collective Wisdom from the Experts 159"},
                    ],
                }
            ],
        }
        result = pass_obj(Artifact(payload=doc))
        blocks = result.payload["pages"][0]["blocks"]
        # Footer block should be removed
        assert len(blocks) == 1
        assert "Regular paragraph" in blocks[0]["text"]

    def test_text_clean_uses_runtime_options(self):
        from pdf_chunker.framework import Artifact
        from pdf_chunker.passes.text_clean import _TextCleanPass

        pass_obj = _TextCleanPass()  # No patterns at construction

        doc = {
            "type": "page_blocks",
            "pages": [
                {
                    "page": 1,
                    "blocks": [
                        {"text": "Body text here."},
                        {"text": "Book Title 42"},
                    ],
                }
            ],
        }
        # Provide patterns via runtime options
        meta = {
            "options": {
                "text_clean": {
                    "footer_patterns": (r"Book Title \d+",),
                }
            }
        }
        result = pass_obj(Artifact(payload=doc, meta=meta))
        blocks = result.payload["pages"][0]["blocks"]
        assert len(blocks) == 1
        assert "Body text" in blocks[0]["text"]

    def test_text_clean_reports_pattern_metrics(self):
        from pdf_chunker.framework import Artifact
        from pdf_chunker.passes.text_clean import _TextCleanPass

        pass_obj = _TextCleanPass(
            footer_patterns=(r"Footer \d+",),
        )

        doc = {
            "type": "page_blocks",
            "pages": [{"page": 1, "blocks": [{"text": "Content"}]}],
        }
        result = pass_obj(Artifact(payload=doc))
        metrics = result.meta["metrics"]["text_clean"]
        assert metrics["footer_patterns_applied"] == 1
