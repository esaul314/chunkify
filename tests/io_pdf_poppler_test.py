"""Native Poppler reader contracts; optional source acceptance needs explicit pins."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from tempfile import TemporaryDirectory
from unittest.mock import patch

import structlog
from structlog.testing import capture_logs

from pdf_chunker import fallbacks
from pdf_chunker.adapters import io_pdf

PROSE = "A coherent paragraph contains ordinary words and preserves the complete source body."


class PopplerReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch.dict(os.environ))
        for key in ("PDF_CHUNKER_OCR_LANGUAGE", "PDF_CHUNKER_OCR_CPUS"):
            os.environ.pop(key, None)
        os.environ["PDF_CHUNKER_PDF_READER"] = "pdftotext"
        self.path = "library/Original Source.pdf"
        self.body = f"{PROSE}\f\f{PROSE} Final page retained.\f"
        self.info = "Title: Synthetic fixture\nPages:          3\nEncrypted: no\n"
        self.opened = self.enterContext(
            patch.object(io_pdf.fitz, "open", create=True, side_effect=AssertionError("MuPDF call"))
        )
        for name in (
            "_all_blocks",
            "_primary_blocks",
            "_page_numbers",
            "_ensure_all_pages",
            "_recover_ocr_pages",
            "_filter_by_zones",
            "ProcessPoolExecutor",
        ):
            self.enterContext(patch.object(io_pdf, name, side_effect=AssertionError(name)))
        self.enterContext(
            patch.object(
                fallbacks, "_extract_with_pdfminer", side_effect=AssertionError("pdfminer")
            )
        )
        self.runner = self.enterContext(patch.object(io_pdf, "run", side_effect=self._run_tool))
        self.timeout_before = io_pdf._PDFTOTEXT_TIMEOUT
        self.addCleanup(setattr, io_pdf, "_PDFTOTEXT_TIMEOUT", self.timeout_before)

    def _run_tool(self, cmd, **kwargs):
        if cmd[0] == "pdfinfo":
            return CompletedProcess(cmd, 0, self.info, "")
        if cmd[0] == "pdftotext":
            return CompletedProcess(cmd, 0, self.body, "")
        raise AssertionError(cmd)

    def test_full_body_slots_provenance_real_command_arguments_and_timeouts(self) -> None:
        result = io_pdf.read(self.path, timeout=17)
        self.assertEqual(result["source_path"], self.path)
        self.assertEqual(result["type"], "page_blocks")
        self.assertEqual([page["page"] for page in result["pages"]], [1, 2, 3])
        self.assertEqual(result["pages"][1]["blocks"], [])
        self.assertEqual(result["pages"][0]["blocks"][0]["text"], PROSE)
        self.assertEqual(result["pages"][2]["blocks"][0]["text"], PROSE + " Final page retained.")
        for page in result["pages"]:
            for block in page["blocks"]:
                self.assertEqual(
                    block["source"],
                    {
                        "filename": "Original Source.pdf",
                        "path": self.path,
                        "page": page["page"],
                        "method": "pdftotext",
                    },
                )
        info_call, text_call = self.runner.call_args_list
        command_path = str(Path(self.path).absolute())
        self.assertEqual(info_call.args, (["pdfinfo", command_path],))
        self.assertEqual(info_call.kwargs["env"]["LC_ALL"], "C")
        self.assertEqual(info_call.kwargs["env"]["LANG"], "C")
        self.assertEqual(info_call.kwargs["encoding"], "utf-8")
        self.assertEqual(info_call.kwargs["timeout"], 17)
        self.assertTrue(info_call.kwargs["capture_output"])
        self.assertTrue(info_call.kwargs["text"])
        self.assertEqual(text_call.args, (["pdftotext", command_path, "-"],))
        self.assertEqual(text_call.kwargs, {"capture_output": True, "text": True, "timeout": 17})
        quality = result["extraction_quality"]
        self.assertEqual(quality["selected_engine"], "pdftotext")
        self.assertEqual(quality["reason"], "explicit_native_reader")
        self.assertEqual(quality["text_order"], "reading")
        self.assertEqual(quality["missing_pages"], [2])
        self.opened.assert_not_called()

    def test_large_selected_body_is_not_truncated(self) -> None:
        self.body = f"{PROSE * 10000}\f\f{PROSE} End sentinel.\f"
        result = io_pdf.read(self.path)
        self.assertEqual(result["pages"][0]["blocks"][0]["text"], PROSE * 10000)
        self.assertTrue(result["pages"][2]["blocks"][0]["text"].endswith("End sentinel."))

    def test_reading_order_keeps_quality_gate_and_legacy_layout_default(self) -> None:
        padding = " " * 60
        layout = (padding.join(PROSE.split(" ")) + "\n") * 20
        reading = (PROSE + "\n") * 20
        self.assertLess(fallbacks._assess_text_quality(layout)["quality_score"], 0.7)
        self.assertGreaterEqual(fallbacks._assess_text_quality(reading)["quality_score"], 0.7)
        self.assertEqual("".join(layout.split()), "".join(reading.split()))

        def rendered(cmd, **kwargs):
            if cmd[0] == "pdfinfo":
                return CompletedProcess(cmd, 0, self.info, "")
            body = layout if "-layout" in cmd else reading
            return CompletedProcess(cmd, 0, body + "\f\f" + PROSE + "\f", "")

        self.runner.side_effect = rendered
        result = io_pdf.read(self.path)
        self.assertEqual([page["page"] for page in result["pages"]], [1, 2, 3])
        self.assertIn(PROSE, result["pages"][0]["blocks"][0]["text"])
        self.assertEqual(fallbacks._extract_with_pdftotext(self.path), [])
        self.assertEqual(self.runner.call_args.args[0], ["pdftotext", "-layout", self.path, "-"])

    def test_exclusions_keep_original_page_numbers_and_all_excluded_is_empty(self) -> None:
        for exclusions in ("1-2", [1, 2], (1, 2)):
            with self.subTest(exclusions=exclusions):
                result = io_pdf.read(self.path, exclude_pages=exclusions)
                self.assertEqual([p["page"] for p in result["pages"]], [3])
                self.assertEqual(result["pages"][0]["blocks"][0]["source"]["page"], 3)
        self.runner.reset_mock()
        self.assertEqual(io_pdf.read(self.path, exclude_pages="1-3")["pages"], [])
        self.assertEqual(self.runner.call_count, 1)  # pdfinfo only
        # Historical adapter semantics ignore exclusion numbers outside the PDF.
        result = io_pdf.read(self.path, exclude_pages=[99])
        self.assertEqual([p["page"] for p in result["pages"]], [1, 2, 3])

    def test_excluding_only_quality_body_fails_honestly(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "native text"):
            io_pdf.read(self.path, exclude_pages="1,3")

    def test_partial_extra_missing_delimiters_and_garbage_are_rejected(self) -> None:
        for body in (
            "",
            "\f\f\f",
            f"{PROSE}\f",
            f"{PROSE}\f\f",
            f"{PROSE}\f\f\f\f",
            f"{PROSE}\f\f{PROSE}",
            "�" * 1000 + "\f\f\f",
        ):
            with self.subTest(length=len(body)):
                self.body = body
                with self.assertRaisesRegex(RuntimeError, "native text"):
                    io_pdf.read(self.path)

    def test_pdfinfo_malformed_duplicate_zero_negative_or_overflow_counts(self) -> None:
        for info in (
            "",
            "Pages: zero\n",
            "Pages: 0\n",
            "Pages: -3\n",
            "Pages: 3.0\n",
            "Pages: ３\n",
            "Pages: 3\nPages: 3\n",
            "Pages: 3\nPages: bad\n",
            "Pages: 99999999999999999\n",
            "Pages: 2147483648\n",
        ):
            with self.subTest(info=info):
                self.info = info
                with self.assertRaisesRegex(RuntimeError, "page count"):
                    io_pdf.read(self.path)

    def test_error_exit_missing_binary_and_timeout_never_return_a_document(self) -> None:
        for tool in ("pdfinfo", "pdftotext"):
            for error in (FileNotFoundError(tool), TimeoutExpired(tool, 17), "exit"):
                with self.subTest(tool=tool, error=type(error).__name__):

                    def fail(cmd, *, _tool=tool, _error=error, **kwargs):
                        if cmd[0] == _tool:
                            if _error == "exit":
                                return CompletedProcess(
                                    cmd,
                                    1,
                                    self.info if _tool == "pdfinfo" else self.body,
                                    "private diagnostics",
                                )
                            raise _error
                        return self._run_tool(cmd, **kwargs)

                    self.runner.side_effect = fail
                    with self.assertRaises(RuntimeError) as raised:
                        io_pdf.read(self.path, timeout=17)
                    self.assertNotIn("private diagnostics", str(raised.exception))

    def test_unknown_empty_and_whitespace_reader_values_refuse_before_subprocess(self) -> None:
        for value in ("", " ", "mupdf", "pymupdf", "auto", "PDFTOTEXT", "pdftotext "):
            with self.subTest(value=value):
                os.environ["PDF_CHUNKER_PDF_READER"] = value
                with self.assertRaisesRegex(ValueError, "PDF_CHUNKER_PDF_READER"):
                    io_pdf.read(self.path)
        self.runner.assert_not_called()

    def test_ocr_enhancement_interaction_and_margins_refuse_before_subprocess(self) -> None:
        for key, value in (
            ("PDF_CHUNKER_OCR_LANGUAGE", "eng"),
            ("PDF_CHUNKER_OCR_CPUS", "28,29,30"),
        ):
            with patch.dict(os.environ, {key: value}), self.assertRaisesRegex(ValueError, "OCR"):
                io_pdf.read(self.path)
        for options in (
            {"use_pymupdf4llm": True},
            {"interactive": True},
            {"footer_margin": 0},
            {"header_margin": 40},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                io_pdf.read(self.path, **options)
        self.runner.assert_not_called()

    def test_invalid_timeout_format_and_page_range_refuse(self) -> None:
        for timeout in (0, -1, 1.5, True, None):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                io_pdf.read(self.path, timeout=timeout)
        for path in ("book.epub", "book.txt", "book.png"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "reclassification"):
                io_pdf.read(path)
        with self.assertRaises(ValueError):
            io_pdf.read(self.path, exclude_pages="3-1")
        self.runner.assert_not_called()

    def test_invalid_blocks_cannot_disappear_or_be_reclassified(self) -> None:
        base = {
            "type": "paragraph",
            "text": PROSE,
            "source": {"filename": "Original Source.pdf", "page": 1, "method": "pdftotext"},
        }
        for source in (
            {"page": 0},
            {"page": 4},
            {"page": True},
            {"page": "1"},
            {"page": 2},
            {"page_range": (1, 4)},
            {"method": "pdfminer"},
            {"filename": "repaired.pdf"},
        ):
            with self.subTest(source=source):
                block = {**base, "source": {**base["source"], **source}}
                with (
                    patch.object(fallbacks, "_extract_with_pdftotext", return_value=[block]),
                    self.assertRaises(RuntimeError),
                ):
                    io_pdf.read(self.path, exclude_pages="2")

    def test_source_blocks_are_copied_and_leading_dash_path_is_safe(self) -> None:
        block = {
            "type": "paragraph",
            "text": PROSE,
            "source": {"filename": "-original.pdf", "page": 1, "method": "pdftotext"},
        }
        before = deepcopy(block)
        with patch.object(fallbacks, "_extract_with_pdftotext", return_value=[block]):
            result = io_pdf.read("-original.pdf")
        self.assertEqual(block, before)
        self.assertEqual(result["source_path"], "-original.pdf")
        self.assertEqual(result["pages"][0]["blocks"][0]["source"]["path"], "-original.pdf")
        self.assertTrue(Path(self.runner.call_args.args[0][1]).is_absolute())

    def test_default_dispatch_remains_unchanged(self) -> None:
        os.environ.pop("PDF_CHUNKER_PDF_READER")
        with (
            patch.object(io_pdf, "_all_blocks", return_value=[]) as primary,
            patch.object(io_pdf, "_ensure_all_pages", return_value=[]) as ensure,
        ):
            self.assertEqual(
                io_pdf.read(self.path),
                {
                    "type": "page_blocks",
                    "source_path": self.path,
                    "pages": [],
                    "extraction_quality": {},
                },
            )
        primary.assert_called_once()
        ensure.assert_called_once()
        self.runner.assert_not_called()

    def test_legacy_fallback_without_count_keeps_old_semantics(self) -> None:
        self.body = PROSE  # Legacy fallback did not require page terminators.
        self.assertTrue(fallbacks._extract_with_pdftotext(self.path))
        self.assertEqual(fallbacks._extract_with_pdftotext(self.path, expected_page_count=3), [])

    def test_real_subprocess_runner_preserves_argv_and_enforces_timeout(self) -> None:
        with patch.object(io_pdf, "run", subprocess.run):
            result = io_pdf.run_pdftotext(
                [
                    sys.executable,
                    "-c",
                    "import sys; print(repr(sys.argv[1:]))",
                    "source with spaces.pdf",
                    "-",
                ],
                timeout=5,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout.strip(), "['source with spaces.pdf', '-']")
            with self.assertRaises(TimeoutExpired):
                io_pdf.run_pdftotext(
                    [sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.05
                )


class OptionalRealPopplerTest(unittest.TestCase):
    def test_explicit_hash_pinned_source_adapter_only(self) -> None:
        keys = (
            "PDF_CHUNKER_POPPLER_TEST_PATH",
            "PDF_CHUNKER_POPPLER_TEST_SHA256",
            "PDF_CHUNKER_POPPLER_TEST_PAGES",
        )
        values = [os.getenv(key) for key in keys]
        if all(value is None for value in values):
            self.skipTest("optional real source requires explicit path, SHA256 and page count")
        if not all(values):
            self.fail("all three PDF_CHUNKER_POPPLER_TEST_* pins are required")
        raw_path, expected_hash, expected_pages = values
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash or ""):
            self.fail("expected SHA256 must be 64 lowercase hexadecimal characters")
        if not re.fullmatch(r"[1-9][0-9]*", expected_pages or ""):
            self.fail("expected page count must be a positive integer")
        path = Path(raw_path)

        def raw_hash():
            with path.open("rb") as source:
                return hashlib.file_digest(source, "sha256").hexdigest()

        if raw_hash() != expected_hash:
            self.fail("raw source SHA256 does not match pin")
        with patch.dict(os.environ):
            for key in ("PDF_CHUNKER_OCR_LANGUAGE", "PDF_CHUNKER_OCR_CPUS"):
                os.environ.pop(key, None)
            os.environ["PDF_CHUNKER_PDF_READER"] = "pdftotext"
            # Existing cleanup uses stdlib handlers that may cache stderr.
            # Silence only this test process and restore its prior state.
            disabled_before = logging.root.manager.disable
            try:
                logging.disable(logging.CRITICAL)
                with (
                    patch.object(io_pdf.fitz, "open", side_effect=AssertionError("MuPDF call")),
                    capture_logs(),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                ):
                    result = io_pdf.read(raw_path, timeout=60)
            finally:
                logging.disable(disabled_before)
        # Use fail messages that never include private body text.
        if result["source_path"] != raw_path or result["type"] != "page_blocks":
            self.fail("source path/type mismatch")
        if [p["page"] for p in result["pages"]] != list(range(1, int(expected_pages) + 1)):
            self.fail("page slots do not match expected complete page count")
        characters = blocks = populated = 0
        for page in result["pages"]:
            populated += bool(page["blocks"])
            for block in page["blocks"]:
                expected_source = {
                    "filename": path.name,
                    "path": raw_path,
                    "page": page["page"],
                    "method": "pdftotext",
                }
                if block["source"] != expected_source:
                    self.fail("original source provenance mismatch")
                characters += len(block["text"])
                blocks += 1
        if not characters or result["extraction_quality"]["selected_engine"] != "pdftotext":
            self.fail("no native body or wrong reader")
        if raw_hash() != expected_hash:
            self.fail("raw source SHA256 changed during extraction")
        print(
            json.dumps(
                {
                    "sha256": expected_hash,
                    "pages": len(result["pages"]),
                    "populated_pages": populated,
                    "blocks": blocks,
                    "characters": characters,
                },
                sort_keys=True,
            )
        )


class GeneratedPopplerIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("pdfinfo") and shutil.which("pdftotext"),
        "installed Poppler pdfinfo and pdftotext are required",
    )
    def test_generated_pdf_with_real_poppler_and_mupdf_sentinel(self) -> None:
        # Build a tiny three-page PDF using only bytes; no MuPDF fixture creation.
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 5 0 R 7 0 R] /Count 3 >>",
        ]
        for number in (3, 5, 7):
            objects.append(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 9 0 R >> >> /Contents {number + 1} 0 R >>".encode()
            )
            text = PROSE if number != 5 else ""
            stream = f"BT /F1 10 Tf 40 700 Td ({text}) Tj ET".encode()
            objects.append(
                f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
            )
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        payload = b"%PDF-1.4\n"
        offsets = [0]
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(payload))
            payload += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
        xref = len(payload)
        payload += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
        payload += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
        payload += (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n" f"startxref\n{xref}\n%%EOF\n"
        ).encode()
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "Generated native.pdf"
            path.write_bytes(payload)
            with patch.dict(os.environ, {"PDF_CHUNKER_PDF_READER": "pdftotext"}):
                os.environ.pop("PDF_CHUNKER_OCR_LANGUAGE", None)
                os.environ.pop("PDF_CHUNKER_OCR_CPUS", None)
                with patch.object(io_pdf.fitz, "open", side_effect=AssertionError("MuPDF call")):
                    result = io_pdf.read(str(path), timeout=10)
                    selected = io_pdf.read(str(path), exclude_pages="1-2", timeout=10)
            self.assertEqual([p["page"] for p in result["pages"]], [1, 2, 3])
            self.assertEqual(result["pages"][1]["blocks"], [])
            self.assertEqual([p["page"] for p in selected["pages"]], [3])
            for page in (result["pages"][0], result["pages"][2], selected["pages"][0]):
                self.assertEqual(page["blocks"][0]["text"].strip(), PROSE)
                self.assertEqual(
                    page["blocks"][0]["source"],
                    {
                        "filename": path.name,
                        "path": str(path),
                        "page": page["page"],
                        "method": "pdftotext",
                    },
                )
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).digest(), hashlib.sha256(payload).digest()
            )


class OptionalPinContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch.dict(os.environ))
        for key in tuple(os.environ):
            if key.startswith("PDF_CHUNKER_POPPLER_TEST_"):
                os.environ.pop(key)
        self.case = OptionalRealPopplerTest("test_explicit_hash_pinned_source_adapter_only")

    def test_partial_and_invalid_pins_fail_before_source_access(self) -> None:
        cases = [
            {"PDF_CHUNKER_POPPLER_TEST_PATH": "/not-read.pdf"},
            {
                "PDF_CHUNKER_POPPLER_TEST_PATH": "/not-read.pdf",
                "PDF_CHUNKER_POPPLER_TEST_SHA256": "invalid",
                "PDF_CHUNKER_POPPLER_TEST_PAGES": "313",
            },
            {
                "PDF_CHUNKER_POPPLER_TEST_PATH": "/not-read.pdf",
                "PDF_CHUNKER_POPPLER_TEST_SHA256": "a" * 64,
                "PDF_CHUNKER_POPPLER_TEST_PAGES": "0",
            },
        ]
        for pins in cases:
            with patch.dict(os.environ, pins), patch.object(Path, "open") as opened:
                with self.assertRaises(AssertionError):
                    self.case.test_explicit_hash_pinned_source_adapter_only()
                opened.assert_not_called()

    def test_generic_counts_hash_gates_and_stdout_privacy(self) -> None:
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "Synthetic source.pdf"
            raw = b"synthetic hash fixture; not a sourcebook"
            path.write_bytes(raw)
            expected_hash = hashlib.sha256(raw).hexdigest()
            for count in (313, 434):
                pins = {
                    "PDF_CHUNKER_POPPLER_TEST_PATH": str(path),
                    "PDF_CHUNKER_POPPLER_TEST_SHA256": expected_hash,
                    "PDF_CHUNKER_POPPLER_TEST_PAGES": str(count),
                }
                pages = [{"page": page, "blocks": []} for page in range(1, count + 1)]
                pages[-1]["blocks"] = [
                    {
                        "text": "private text sentinel",
                        "source": {
                            "filename": path.name,
                            "path": str(path),
                            "page": count,
                            "method": "pdftotext",
                        },
                    }
                ]
                document = {
                    "type": "page_blocks",
                    "source_path": str(path),
                    "pages": pages,
                    "extraction_quality": {"selected_engine": "pdftotext"},
                }
                output = io.StringIO()
                errors = io.StringIO()
                stdlogger = logging.getLogger("pdf_chunker.page_artifacts")
                disabled_before = logging.root.manager.disable

                def noisy_read(*args, _document=document, _stdlogger=stdlogger, **kwargs):
                    _stdlogger.warning("private text sentinel")
                    structlog.get_logger().warning("private text sentinel")
                    print("private text sentinel")
                    print("private text sentinel", file=sys.stderr)
                    return _document

                with (
                    patch.dict(os.environ, pins),
                    patch.object(io_pdf, "read", side_effect=noisy_read),
                    patch.object(stdlogger, "handlers", [logging.StreamHandler(errors)]),
                    patch.object(stdlogger, "propagate", False),
                    redirect_stdout(output),
                    redirect_stderr(errors),
                ):
                    self.case.test_explicit_hash_pinned_source_adapter_only()
                counts = json.loads(output.getvalue())
                self.assertEqual(counts["pages"], count)
                self.assertEqual(counts["sha256"], expected_hash)
                self.assertNotIn("private text sentinel", output.getvalue())
                self.assertNotIn("private text sentinel", errors.getvalue())
                self.assertEqual(logging.root.manager.disable, disabled_before)
                self.assertNotIn(str(path), output.getvalue())
                with patch.dict(os.environ, {**pins, "PDF_CHUNKER_POPPLER_TEST_SHA256": "0" * 64}):
                    with patch.object(io_pdf, "read") as read, self.assertRaises(AssertionError):
                        self.case.test_explicit_hash_pinned_source_adapter_only()
                    read.assert_not_called()

                def changed_source(*args, _document=document, **kwargs):
                    path.write_bytes(b"changed synthetic bytes")
                    return _document

                with (
                    patch.dict(os.environ, pins),
                    patch.object(io_pdf, "read", changed_source),
                    self.assertRaisesRegex(AssertionError, "changed during"),
                ):
                    self.case.test_explicit_hash_pinned_source_adapter_only()
                path.write_bytes(raw)
