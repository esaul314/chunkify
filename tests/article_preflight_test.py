"""Complete UTF-8 row admission over existing, synthetic base article output."""

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from functools import partial

import pytest

from pdf_chunker.adapters import article_conversion as adapter
from pdf_chunker.adapters.article_conversion import (
    execute_bound_conversion,
    prepare_bound_conversion,
)
from pdf_chunker.adapters.emit_jsonl import _serialize
from pdf_chunker.article_output import preflight_article_output
from tests.verified_markdown_ingress_test import SOURCE, bind_article


@pytest.fixture(params=[".md", ".epub"])
def case(tmp_path, monkeypatch, request):
    monkeypatch.setattr(adapter, "collect_article_runtime", lambda: {"fixture": "stable"})
    text = SOURCE + "\nUnicode separators stay in one JSON row: \u0085\u2028\u2029.\n"
    bound = bind_article(tmp_path / "article", suffix=request.param, text=text)
    spec = bound.config()
    spec.options["emit_jsonl"]["metadata_key"] = "article_meta"
    prepared = prepare_bound_conversion(bound.binding, spec, corpus="technology")
    output = execute_bound_conversion(bound.binding, prepared=prepared, corpus="technology")
    return prepared.descriptor, output, text


def stage(descriptor, output, **settings):
    return partial(
        preflight_article_output,
        descriptor=descriptor,
        expected_sha256=output.sha256,
        **settings,
    )


def test_complete_wire_measurement_is_pure_detached_and_partially_applicable(case, monkeypatch):
    descriptor, output, text = case
    rows = output.rows()
    # The preflight must operate entirely on the already retained bytes.
    monkeypatch.setattr(adapter, "convert", lambda *args: pytest.fail("conversion attempted"))
    monkeypatch.setattr("pathlib.Path.open", lambda *args, **kwargs: pytest.fail("I/O attempted"))
    check = stage(descriptor, output, max_row_bytes=100000, enrichment_headroom_bytes=128)
    report = check(output.jsonl)
    sizes = tuple(len(json.dumps(row, ensure_ascii=False).encode("utf-8")) for row in rows)

    assert report.output == output
    assert report.row_sizes_bytes == sizes
    assert report.largest_row_bytes == max(sizes)
    assert report.projected_output_bytes == output.size_bytes + 128 * output.row_count
    assert sum(sizes) + len(sizes) == output.size_bytes  # One LF per JSON record.
    assert report.fits and report.oversized_rows == ()
    assert check(output.jsonl) == report
    assert "".join(row["text"] for row in report.output.rows()) == text
    report.output.rows()[0]["article_meta"].clear()
    assert output.rows() == rows
    with pytest.raises(FrozenInstanceError):
        report.max_row_bytes = 1


@pytest.mark.parametrize("headroom", [0, 1, 256])
def test_exact_projected_boundary_fits_and_one_byte_less_requires_review(case, headroom):
    descriptor, output, _ = case
    sizes = tuple(len(line) for line in output.jsonl.split(b"\n")[:-1])
    limit = max(sizes) + headroom
    check = stage(descriptor, output, enrichment_headroom_bytes=headroom)
    admitted = check(output.jsonl, max_row_bytes=limit)
    review = check(output.jsonl, max_row_bytes=limit - 1)

    assert admitted.fits and admitted.oversized_rows == ()
    assert not review.fits
    assert review.oversized_rows == tuple(
        i for i, size in enumerate(sizes, 1) if size == max(sizes)
    )
    assert review.output == admitted.output == output
    assert review.projected_output_bytes == admitted.projected_output_bytes


def test_character_limit_does_not_admit_utf8_bytes_and_all_overflow_rows_are_reported(case):
    descriptor, output, _ = case
    lines = output.jsonl.decode("utf-8").split("\n")[:-1]
    character_limit = max(map(len, lines))
    report = stage(descriptor, output, max_row_bytes=character_limit)(output.jsonl)

    assert max(map(len, lines)) <= character_limit < report.largest_row_bytes
    assert not report.fits
    tiny = stage(descriptor, output, max_row_bytes=1)(output.jsonl)
    assert tiny.oversized_rows == tuple(range(1, output.row_count + 1))
    assert tiny.output.jsonl == output.jsonl


@pytest.mark.parametrize(
    "text",
    [
        "```python\nvalue = '" + "界🧭" * 1000 + "'\n```\n",
        "| Item | Value |\n| --- | --- |\n| large | " + "界🧭" * 1000 + " |\n",
        "[Long source](https://example.org/" + "é" * 2000 + ")\n",
    ],
)
def test_oversized_structural_context_returns_review_with_exact_bytes(tmp_path, monkeypatch, text):
    monkeypatch.setattr(adapter, "collect_article_runtime", lambda: {"fixture": "stable"})
    bound = bind_article(tmp_path / "article", text=text)
    prepared = prepare_bound_conversion(bound.binding, bound.config(), corpus="technology")
    output = execute_bound_conversion(bound.binding, prepared=prepared, corpus="technology")
    report = stage(descriptor=prepared.descriptor, output=output, max_row_bytes=2000)(output.jsonl)

    assert not report.fits and report.oversized_rows
    assert report.output.jsonl == output.jsonl
    assert "".join(row["text"] for row in report.output.rows()) == text
    assert not list(tmp_path.rglob("*.jsonl"))


def test_reserve_changes_only_admission_and_never_base_identity(case):
    descriptor, output, _ = case
    limit = max(len(line) for line in output.jsonl.split(b"\n")[:-1])
    check = stage(descriptor, output, max_row_bytes=limit)
    base = check(output.jsonl)
    reserved = check(output.jsonl, enrichment_headroom_bytes=1)

    assert base.fits and not reserved.fits
    assert base.output == reserved.output == output
    assert base.row_sizes_bytes == reserved.row_sizes_bytes
    assert reserved.projected_output_bytes - base.projected_output_bytes == output.row_count


def test_large_native_metadata_counts_without_splitting_or_dropping_provenance(case):
    descriptor, output, text = case
    native = json.loads(descriptor.metadata_json)
    native["extra_provenance"] = "界🧭" * 2000
    descriptor = replace(descriptor, metadata_json=json.dumps(native))
    rows = output.rows()
    for row in rows:
        row["article_meta"]["extra_provenance"] = native["extra_provenance"]
    data = "".join(line + "\n" for line in _serialize(rows)).encode("utf-8")
    report = preflight_article_output(
        data,
        descriptor=descriptor,
        expected_sha256=hashlib.sha256(data).hexdigest(),
        max_row_bytes=10000,
        enrichment_headroom_bytes=512,
    )

    assert not report.fits
    assert report.oversized_rows == tuple(range(1, len(rows) + 1))
    assert report.output.rows() == rows
    assert "".join(row["text"] for row in rows) == text
    assert report.output.size_bytes == len(data)


@pytest.mark.parametrize(
    "settings",
    [
        {"max_row_bytes": 0},
        {"max_row_bytes": -1},
        {"max_row_bytes": True},
        {"max_row_bytes": 1.0},
        {"max_row_bytes": "1000"},
        {"enrichment_headroom_bytes": -1},
        {"enrichment_headroom_bytes": True},
        {"enrichment_headroom_bytes": 1.5},
        {"enrichment_headroom_bytes": None},
    ],
)
def test_ambiguous_or_invalid_budgets_refuse(case, settings):
    descriptor, output, _ = case
    with pytest.raises(ValueError, match="bytes"):
        stage(descriptor, output, **({"max_row_bytes": 10000} | settings))(output.jsonl)


def test_preflight_never_reports_admission_for_unvalidated_output(case):
    descriptor, output, _ = case
    with pytest.raises(ValueError, match="SHA-256"):
        stage(descriptor, output, max_row_bytes=100000)(output.jsonl + b"broken")
    with pytest.raises(ValueError):
        preflight_article_output(
            b"{}\n",
            descriptor=descriptor,
            expected_sha256=hashlib.sha256(b"{}\n").hexdigest(),
            max_row_bytes=100000,
        )
