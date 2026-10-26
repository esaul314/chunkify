"""Bounded optional process OCR: ordering, errors, provenance and real parity."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pdf_chunker.adapters import io_pdf
from tests.local_source_controls import private_value


def _identity(_):
    return (
        os.getpid(),
        os.getpgrp(),
        sorted(os.sched_getaffinity(0)),
        os.getenv("OMP_THREAD_LIMIT"),
    )


@pytest.mark.parametrize("value", ["0", "1,1", "1,2,3,4", "-1,2", "1, 2", "a,b", "１,２"])
def test_invalid_cpu_pool_is_explicit_failure(value, monkeypatch):
    monkeypatch.setenv("PDF_CHUNKER_OCR_CPUS", value)
    with pytest.raises(ValueError, match="PDF_CHUNKER_OCR_CPUS"):
        io_pdf._ocr_cpus()


@pytest.mark.parametrize("value", ["", " "])
def test_default_serial_cpu_setting(value, monkeypatch):
    monkeypatch.setenv("PDF_CHUNKER_OCR_CPUS", value)
    assert io_pdf._ocr_cpus() == ()


def test_parallel_mode_is_opt_in_and_requires_affinity_support(monkeypatch):
    monkeypatch.delenv("PDF_CHUNKER_OCR_CPUS", raising=False)
    assert io_pdf._ocr_cpus() == ()
    monkeypatch.setenv("PDF_CHUNKER_OCR_CPUS", "28,29,30")
    assert io_pdf._ocr_cpus() == (28, 29, 30)
    monkeypatch.delattr(os, "sched_setaffinity")
    with pytest.raises(ValueError, match="affinity"):
        io_pdf._ocr_cpus()


def test_pool_bounds_tasks_and_preserves_page_order():
    pool = MagicMock()
    pool.map.side_effect = lambda function, tasks: [[{"page": task[1]}] for task in tasks]
    with patch.object(io_pdf, "ProcessPoolExecutor") as factory:
        factory.return_value.__enter__.return_value = pool
        result = list(
            io_pdf._parallel_ocr_pages("book.pdf", [2, 4, 5, 8, 9], "eng", 40, 50, (28, 29, 30))
        )
    assert result == [(page, [{"page": page}]) for page in [2, 4, 5, 8, 9]]
    assert [len(call.args[1]) for call in pool.map.call_args_list] == [3, 2]
    args = factory.call_args.kwargs
    assert args["max_workers"] == 3
    assert args["mp_context"].get_start_method() == "spawn"
    assert args["initializer"] is io_pdf._ocr_worker_init
    assert args["initargs"] == ((28, 29, 30),)


def test_worker_uses_original_page_and_source_context():
    page = SimpleNamespace(rect=SimpleNamespace(height=800), get_textpage_ocr=lambda **kwargs: "tp")
    page.get_text = lambda *args, **kwargs: [
        (50, 50, 400, 760, " Real text. ", 7, 0),
        (50, 10, 400, 20, "Header", 8, 0),
    ]
    doc = MagicMock()
    doc.__getitem__.return_value = page
    with patch.object(io_pdf.fitz, "open") as opened:
        opened.return_value.__enter__.return_value = doc
        rows = io_pdf._ocr_page_task(("Original.pdf", 5, "eng", 40, 50))
    doc.__getitem__.assert_called_once_with(4)
    assert len(rows) == 1
    assert rows[0]["text"] == "Real text."
    assert rows[0]["source"] == {
        "filename": "Original.pdf",
        "path": "Original.pdf",
        "page": 5,
        "index": 7,
        "location": None,
        "method": "pymupdf_ocr",
    }


def test_failed_worker_reports_source_and_page_and_does_not_return_partial_output():
    with (
        patch.object(io_pdf.fitz, "open", side_effect=RuntimeError("engine failure")),
        pytest.raises(RuntimeError, match="Original.pdf.*page 7.*engine failure"),
    ):
        io_pdf._ocr_page_task(("Original.pdf", 7, "eng", None, None))


def test_parallel_error_never_returns_partial_document(monkeypatch):
    monkeypatch.setenv("PDF_CHUNKER_OCR_CPUS", "28,29,30")
    quality = {}

    def partial(*args):
        yield (1, [{"text": "first page"}])
        raise RuntimeError("OCR failed for book.pdf, page 2")

    with (
        patch.object(io_pdf, "_page_numbers", return_value=range(1, 4)),
        patch.object(io_pdf, "_parallel_ocr_pages", partial),
        pytest.raises(RuntimeError, match="page 2"),
    ):
        io_pdf._recover_ocr_pages(
            "book.pdf",
            [],
            set(),
            "eng",
            footer_margin=None,
            header_margin=None,
            quality_report=quality,
        )
    assert "ocr" not in quality


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux affinity required")
def test_spawn_workers_keep_process_group_and_explicit_cpu_envelope():
    cpus = tuple(sorted(os.sched_getaffinity(0))[:3])
    if len(cpus) < 2:
        pytest.skip("two allowed test CPUs required")
    with ProcessPoolExecutor(
        max_workers=len(cpus),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=io_pdf._ocr_worker_init,
        initargs=(cpus,),
    ) as pool:
        records = list(pool.map(_identity, range(len(cpus))))
    assert all(
        (
            pid != os.getpid()
            and group == os.getpgrp()
            and (affinity == list(cpus))
            and (omp == "1")
            for pid, group, affinity, omp in records
        )
    )


def test_optional_actual_book_parallel_payload_equivalence(monkeypatch):
    path = Path(private_value("control_007", ""))
    if not path.is_file():
        pytest.skip("optional local corpus fixture unavailable")
    expected = private_value("control_008", "")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    selected = {5, 67, 132}
    with io_pdf.fitz.open(path) as document:
        excluded = [page for page in range(1, len(document) + 1) if page not in selected]
    monkeypatch.setenv("PDF_CHUNKER_OCR_LANGUAGE", "eng")
    monkeypatch.delenv("PDF_CHUNKER_OCR_CPUS", raising=False)
    start = time.monotonic()
    serial = io_pdf.read(str(path), exclude_pages=excluded)
    serial_seconds = time.monotonic() - start
    monkeypatch.setenv("PDF_CHUNKER_OCR_CPUS", "28,29,30")
    start = time.monotonic()
    parallel = io_pdf.read(str(path), exclude_pages=excluded)
    parallel_seconds = time.monotonic() - start
    assert parallel == serial
    assert serial["extraction_quality"]["ocr"]["recovered_characters"] > 8000
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    print(
        json.dumps(
            {
                "serial_seconds": serial_seconds,
                "parallel_seconds": parallel_seconds,
                "speedup": serial_seconds / parallel_seconds,
                "exact_payload_equal": True,
                "source_sha256": expected,
                "selected_pages": sorted(selected),
            }
        )
    )
