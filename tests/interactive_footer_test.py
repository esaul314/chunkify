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


class TestInlineFooterStripping:
    """Tests for inline footer stripping (footers merged mid-text)."""

    def test_build_inline_footer_pattern(self):
        from pdf_chunker.interactive import build_inline_footer_pattern

        pattern = build_inline_footer_pattern(r"Scale Communication")
        assert pattern.search("\n\nScale Communication 202 continuation")
        assert not pattern.search("Scale Communication without page")
        assert not pattern.search("not matching\nSingle newline 1")

    def test_compile_footer_patterns_inline_mode(self):
        from pdf_chunker.interactive import compile_footer_patterns

        patterns = compile_footer_patterns(
            ("Scale Communication", "Chapter Title"),
            inline=True,
            midtext=True,
        )
        # 4 patterns: 2 per input (one \n\n prefix, one mid-text)
        assert len(patterns) == 4
        # Both should match inline structure
        text = "previous text\n\nScale Communication 202 next text"
        assert any(p.search(text) for p in patterns)
        # Also test mid-text match
        midtext = "some previous text. Scale Communication 202 next text"
        assert any(p.search(midtext) for p in patterns)

    def test_compile_footer_patterns_non_inline_mode(self):
        from pdf_chunker.interactive import compile_footer_patterns

        patterns = compile_footer_patterns(
            ("Scale Communication",),
            inline=False,
        )
        assert len(patterns) == 1
        # Should match directly, not require \n\n prefix
        assert patterns[0].search("Scale Communication")

    def test_strip_inline_footers_basic(self):
        from pdf_chunker.interactive import compile_footer_patterns, strip_inline_footers

        # Pattern with .* to match full title before page number
        patterns = compile_footer_patterns(("Scale Communication.*",), inline=True)
        text = (
            "There's conflicting scientific literature on the subject."
            "\n\nScale Communication Through Writing 202 Aside from that, "
            "we can all likely agree on a few basic elements."
        )
        cleaned, stripped = strip_inline_footers(text, patterns)
        # Footer should be removed but paragraph break preserved
        assert "Scale Communication" not in cleaned
        assert "conflicting scientific" in cleaned
        assert "Aside from that" in cleaned
        assert len(stripped) == 1
        assert "Scale Communication" in stripped[0]

    def test_strip_inline_footers_preserves_paragraph_break(self):
        from pdf_chunker.interactive import compile_footer_patterns, strip_inline_footers

        patterns = compile_footer_patterns(("Chapter",), inline=True)
        text = "First paragraph.\n\nChapter 5 Second paragraph."
        cleaned, _ = strip_inline_footers(text, patterns)
        # Should preserve double newline
        assert "\n\n" in cleaned
        assert cleaned == "First paragraph.\n\n Second paragraph."

    def test_strip_inline_footers_with_callback(self):
        from pdf_chunker.interactive import (
            compile_footer_patterns,
            make_batch_footer_prompt,
            strip_inline_footers,
        )

        patterns = compile_footer_patterns(("Header",), inline=True)
        # Create callback that rejects "Header 1" as footer
        decisions = {"Header 1": False}
        callback = make_batch_footer_prompt(decisions)

        text = "Text before.\n\nHeader 1 Text after."
        cleaned, stripped = strip_inline_footers(text, patterns, callback=callback, page=1)
        # Should NOT strip because callback rejected
        assert "Header 1" in cleaned
        assert len(stripped) == 0

    def test_strip_inline_footers_multiple_occurrences(self):
        from pdf_chunker.interactive import compile_footer_patterns, strip_inline_footers

        patterns = compile_footer_patterns(("Footer",), inline=True)
        text = "Para 1.\n\nFooter 10 Para 2.\n\nFooter 20 Para 3."
        cleaned, stripped = strip_inline_footers(text, patterns)
        assert "Footer 10" not in cleaned
        assert "Footer 20" not in cleaned
        assert len(stripped) == 2


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

    def test_text_clean_strips_inline_footer(self):
        """Test that inline footers (merged into text) are stripped."""
        from pdf_chunker.framework import Artifact
        from pdf_chunker.passes.text_clean import _TextCleanPass

        pass_obj = _TextCleanPass(
            footer_patterns=(r"Scale Communication.*",),
        )

        doc = {
            "type": "page_blocks",
            "pages": [
                {
                    "page": 202,
                    "blocks": [
                        {
                            "text": (
                                "There's conflicting scientific literature on the subject."
                                "\n\nScale Communication Through Writing 202 "
                                "Aside from that, we can all likely agree."
                            )
                        },
                    ],
                }
            ],
        }
        result = pass_obj(Artifact(payload=doc))
        blocks = result.payload["pages"][0]["blocks"]
        assert len(blocks) == 1
        text = blocks[0]["text"]
        assert "Scale Communication" not in text
        assert "conflicting scientific" in text
        assert "Aside from that" in text

    def test_text_clean_reports_inline_footer_metrics(self):
        """Test that inline footers stripped count is reported in metrics."""
        from pdf_chunker.framework import Artifact
        from pdf_chunker.passes.text_clean import _TextCleanPass

        pass_obj = _TextCleanPass(
            footer_patterns=(r"Chapter",),
        )

        doc = {
            "type": "page_blocks",
            "pages": [
                {
                    "page": 5,
                    "blocks": [
                        {"text": "Intro text.\n\nChapter 5 Body text."},
                    ],
                }
            ],
        }
        result = pass_obj(Artifact(payload=doc))
        metrics = result.meta["metrics"]["text_clean"]
        assert metrics.get("inline_footers_stripped", 0) == 1


class TestHeuristicFooterDetection:
    """Tests for heuristic footer detection (no patterns provided)."""

    def test_detect_inline_footer_candidates(self):
        from pdf_chunker.interactive import detect_inline_footer_candidates

        text = (
            "First paragraph content here."
            "\n\nScale Communication Through Writing 202 "
            "Next paragraph continues here."
        )
        candidates = detect_inline_footer_candidates(text)
        assert len(candidates) == 1
        assert "Scale Communication" in candidates[0][0]

    def test_detect_multiple_candidates(self):
        from pdf_chunker.interactive import detect_inline_footer_candidates

        text = "Para 1.\n\nChapter One 10 Para 2.\n\nThe Art of Leadership 25 Para 3."
        candidates = detect_inline_footer_candidates(text)
        assert len(candidates) == 2

    def test_strip_inline_footers_interactive_basic(self):
        from pdf_chunker.interactive import (
            FooterDecisionCache,
            make_batch_footer_prompt,
            strip_inline_footers_interactive,
        )

        # Create a callback that accepts all as footers
        callback = make_batch_footer_prompt({"Scale Communication": True})
        cache = FooterDecisionCache()

        text = (
            "Previous content here."
            "\n\nScale Communication Through Writing 202 "
            "Continuation text follows."
        )
        cleaned, stripped = strip_inline_footers_interactive(
            text, callback=callback, cache=cache, page=202
        )
        assert "Scale Communication" not in cleaned
        assert len(stripped) == 1

    def test_strip_inline_footers_interactive_reject(self):
        from pdf_chunker.interactive import (
            FooterDecisionCache,
            make_batch_footer_prompt,
            strip_inline_footers_interactive,
        )

        # Create a callback that rejects footers
        callback = make_batch_footer_prompt({"Chapter": False})
        cache = FooterDecisionCache()

        text = "Text before.\n\nChapter One 10 Text after."
        cleaned, stripped = strip_inline_footers_interactive(
            text, callback=callback, cache=cache, page=10
        )
        # Should NOT strip because callback rejected
        assert "Chapter One" in cleaned
        assert len(stripped) == 0

    def test_text_clean_pass_interactive_heuristic_mode(self):
        """Test that interactive mode without patterns uses heuristic detection."""
        from pdf_chunker.framework import Artifact
        from pdf_chunker.interactive import make_batch_footer_prompt
        from pdf_chunker.passes.text_clean import _TextCleanPass

        # Create pass with interactive enabled but no patterns
        pass_obj = _TextCleanPass(interactive_footers=True)

        # Override the callback for testing (normally uses stdin)
        pass_obj._footer_callback = make_batch_footer_prompt({"Scale Communication": True})

        doc = {
            "type": "page_blocks",
            "pages": [
                {
                    "page": 202,
                    "blocks": [
                        {
                            "text": (
                                "Previous content here."
                                "\n\nScale Communication Through Writing 202 "
                                "Continuation text follows."
                            )
                        },
                    ],
                }
            ],
        }
        result = pass_obj(Artifact(payload=doc))
        blocks = result.payload["pages"][0]["blocks"]
        text = blocks[0]["text"]
        assert "Scale Communication" not in text
        assert "Continuation text" in text
        metrics = result.meta["metrics"]["text_clean"]
        assert metrics.get("interactive_heuristic") is True


class TestStandaloneFooterDetection:
    """Tests for standalone footer block detection (entire block is footer)."""

    def test_is_standalone_footer_candidate_positive(self):
        from pdf_chunker.interactive import is_standalone_footer_candidate

        result = is_standalone_footer_candidate("Scale Communication Through Writing 1")
        assert result is not None
        assert result[0] == "Scale Communication Through Writing"
        assert result[1] == "1"

    def test_is_standalone_footer_candidate_with_page_number(self):
        from pdf_chunker.interactive import is_standalone_footer_candidate

        result = is_standalone_footer_candidate("On Accountability 160")
        assert result is not None
        assert result[0] == "On Accountability"
        assert result[1] == "160"

    def test_is_standalone_footer_candidate_negative_no_number(self):
        from pdf_chunker.interactive import is_standalone_footer_candidate

        result = is_standalone_footer_candidate("Just a normal sentence.")
        assert result is None

    def test_is_standalone_footer_candidate_negative_too_many_words(self):
        from pdf_chunker.interactive import is_standalone_footer_candidate

        # More than 6 words after the initial cap word
        result = is_standalone_footer_candidate("This Has Way Too Many Words For A Footer Title 1")
        assert result is None

    def test_text_clean_pass_standalone_footer_interactive(self):
        """Test that interactive mode detects standalone footer blocks."""
        from pdf_chunker.framework import Artifact
        from pdf_chunker.interactive import make_batch_footer_prompt
        from pdf_chunker.passes.text_clean import _TextCleanPass

        # Create pass with interactive enabled but no patterns
        pass_obj = _TextCleanPass(interactive_footers=True)

        # Override the callback for testing - confirm it's a footer
        pass_obj._footer_callback = make_batch_footer_prompt(
            {"Scale Communication Through Writing": True}
        )

        doc = {
            "type": "page_blocks",
            "pages": [
                {
                    "page": 1,
                    "blocks": [
                        {"text": "Body text before."},
                        {"text": "Scale Communication Through Writing 1"},  # Standalone footer
                        {"text": "Body text after."},
                    ],
                }
            ],
        }
        result = pass_obj(Artifact(payload=doc))
        blocks = result.payload["pages"][0]["blocks"]
        # Footer block should be empty (removed)
        texts = [b["text"] for b in blocks]
        assert "Body text before." in texts
        assert "Body text after." in texts
        # The footer should be stripped (empty string)
        assert "" in texts or len(texts) == 2  # Either empty or removed

    def test_text_clean_pass_standalone_footer_rejected(self):
        """Test that rejected standalone footer blocks are preserved."""
        from pdf_chunker.framework import Artifact
        from pdf_chunker.interactive import make_batch_footer_prompt
        from pdf_chunker.passes.text_clean import _TextCleanPass

        pass_obj = _TextCleanPass(interactive_footers=True)

        # Override the callback - reject the footer
        pass_obj._footer_callback = make_batch_footer_prompt(
            {"Scale Communication Through Writing": False}
        )

        doc = {
            "type": "page_blocks",
            "pages": [
                {
                    "page": 1,
                    "blocks": [
                        {"text": "Body text before."},
                        {"text": "Scale Communication Through Writing 1"},
                        {"text": "Body text after."},
                    ],
                }
            ],
        }
        result = pass_obj(Artifact(payload=doc))
        blocks = result.payload["pages"][0]["blocks"]
        # Footer block should be preserved
        texts = [b["text"] for b in blocks]
        # The footer text should still be present (though cleaned)
        assert any("Scale Communication" in t for t in texts)


class TestPatternsPlusInteractiveMode:
    """Test that patterns + interactive mode work together correctly."""

    def test_patterns_auto_strip_without_prompting(self):
        """Explicit patterns should strip footers without prompting."""
        from pdf_chunker.framework import Artifact
        from pdf_chunker.passes.text_clean import _TextCleanPass

        # Pattern-only mode (no interactive)
        pass_obj = _TextCleanPass(
            footer_patterns=("Collective Wisdom.*",),
            interactive_footers=False,
        )

        doc = {
            "type": "page_blocks",
            "pages": [
                {
                    "page": 1,
                    "blocks": [
                        {"text": "Body text.\n\nCollective Wisdom from the Experts 153 More content."},
                    ],
                }
            ],
        }
        result = pass_obj(Artifact(payload=doc))
        text = result.payload["pages"][0]["blocks"][0]["text"]
        # Pattern should auto-strip without prompting
        assert "Collective Wisdom" not in text
        assert "Body text" in text

    def test_interactive_catches_additional_footers_with_patterns(self):
        """Interactive mode should catch footers not matched by patterns."""
        from pdf_chunker.framework import Artifact
        from pdf_chunker.interactive import make_batch_footer_prompt
        from pdf_chunker.passes.text_clean import _TextCleanPass

        pass_obj = _TextCleanPass(
            footer_patterns=("Known Footer.*",),
            interactive_footers=True,
        )

        # Override callback to auto-accept everything
        pass_obj._footer_callback = make_batch_footer_prompt(
            {"Heuristic Footer": True, "Known Footer": True}
        )

        doc = {
            "type": "page_blocks",
            "pages": [
                {
                    "page": 1,
                    "blocks": [
                        {"text": "Body.\n\nKnown Footer Pattern 100 followed by text."},
                        {"text": "More body. Heuristic Footer 200 continuation."},
                    ],
                }
            ],
        }
        result = pass_obj(Artifact(payload=doc))
        blocks = result.payload["pages"][0]["blocks"]
        texts = [b["text"] for b in blocks]
        full_text = " ".join(texts)

        # Known Footer should be stripped by pattern (no prompting)
        assert "Known Footer" not in full_text
        # Heuristic Footer should be stripped by interactive callback
        assert "Heuristic Footer" not in full_text
        # Body text should be preserved
        assert "Body" in full_text
