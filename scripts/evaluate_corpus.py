#!/usr/bin/env python3
"""Run representative documents and write a category-based quality report.

Example:

    python scripts/evaluate_corpus.py \
        --manifest docs/corpus_manifest.yaml \
        --output /tmp/pdf-chunker-quality.json
"""

# The script adds the repository root to sys.path for direct execution.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdf_chunker.config import PipelineSpec, load_spec
from pdf_chunker.core_new import _input_artifact, run_convert
from pdf_chunker.evaluation import (
    evaluate_rows,
    extraction_summary,
    parser_page_count,
    quality_passed,
)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load and validate a corpus manifest."""

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping) or data.get("version") != 1:
        raise ValueError("corpus manifest must contain version: 1")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("corpus manifest must contain a non-empty cases list")
    result: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"corpus case {index} must be a mapping")
        required = {"name", "path", "kind"}
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"corpus case {index} is missing: {', '.join(missing)}")
        if case["kind"] not in {"pdf", "epub", "txt"}:
            raise ValueError(f"unsupported corpus kind: {case['kind']!r}")
        result.append(dict(case))
    return result


def _materialize_case(case: Mapping[str, Any], root: Path, temp_root: Path) -> tuple[Path, bytes]:
    """Resolve a manifest source, materializing base64 fixtures when requested."""

    source = root / str(case["path"])
    encoding = case.get("encoding", "plain")
    if encoding == "plain":
        data = source.read_bytes()
        return source, data
    if encoding != "base64":
        raise ValueError(f"unsupported corpus encoding: {encoding!r}")
    encoded = "".join(source.read_text(encoding="ascii").split())
    data = base64.b64decode(encoded, validate=True)
    filename = str(case.get("filename") or source.stem)
    target = temp_root / filename
    target.write_bytes(data)
    return target, data


def _case_spec(case: Mapping[str, Any], root: Path, temp_root: Path) -> PipelineSpec:
    """Load a case spec and redirect generated files into the temp directory."""

    spec_ref = case.get("spec", "pipeline.yaml")
    spec_path = root / str(spec_ref)
    spec = load_spec(spec_path, overrides=case.get("options"))
    options = {name: dict(value) for name, value in spec.options.items()}
    options["run_report"] = {"output_path": str(temp_root / "run_report.json")}
    options["emit_jsonl"] = {
        **options.get("emit_jsonl", {}),
        "output_path": str(temp_root / "output.jsonl"),
    }
    return spec.model_copy(update={"options": options})


def _rows(payload: Any) -> list[Mapping[str, Any]]:
    """Return mapping rows from the canonical runner payload."""

    if not isinstance(payload, list):
        raise ValueError("runner did not return emitted rows")
    if any(not isinstance(row, Mapping) for row in payload):
        raise ValueError("runner returned a non-mapping row")
    return list(payload)


def _evaluate_case(case: Mapping[str, Any], root: Path, temp_root: Path) -> dict[str, Any]:
    """Run and score one manifest case, retaining failures in the report."""

    temp_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "name": case["name"],
        "kind": case["kind"],
        "path": str(case["path"]),
        "reviewed_for": list(case.get("reviewed_for", [])),
        "spec": str(case.get("spec", "pipeline.yaml")),
    }
    try:
        source_path, source_bytes = _materialize_case(case, root, temp_root)
        result["sha256"] = hashlib.sha256(source_bytes).hexdigest()
        spec = _case_spec(case, root, temp_root)
        artifact, _ = run_convert(_input_artifact(str(source_path), spec), spec)
        report_path = Path(spec.options["run_report"]["output_path"])
        run_report = json.loads(report_path.read_text(encoding="utf-8"))
        run_metrics = run_report.get("metrics", {})
        metrics = evaluate_rows(
            _rows(artifact.payload),
            page_count=parser_page_count(run_metrics),
            run_metrics=run_metrics,
            sentinels=case.get("sentinels", []),
            ordered_fragments=case.get("ordered_fragments", []),
        )
        expected_pages = case.get("expected_page_count")
        actual_pages = metrics["page_coverage"]["page_count"]
        metrics["preservation"]["page_count"] = {
            "expected": expected_pages,
            "actual": actual_pages,
            "passed": expected_pages is None or expected_pages == actual_pages,
        }
        result.update(
            {
                "pipeline": list(spec.pipeline),
                "extraction": extraction_summary(run_metrics, kind=str(case["kind"])),
                "metrics": metrics,
                "warnings": list(run_report.get("warnings", [])),
                "passed": quality_passed(metrics),
            }
        )
    except Exception as exc:  # noqa: BLE001 - report each case and continue
        result.update({"passed": False, "error": f"{exc.__class__.__name__}: {exc}"})
    return result


def evaluate_manifest(manifest_path: Path, root: Path | None = None) -> dict[str, Any]:
    """Evaluate all manifest cases and return a stable JSON-compatible report."""

    resolved_root = (root or ROOT).resolve()
    resolved_manifest = manifest_path.resolve()
    cases = load_manifest(resolved_manifest)
    with tempfile.TemporaryDirectory(prefix="pdf-chunker-evaluation-") as temp_dir:
        temp_root = Path(temp_dir)
        results = [
            _evaluate_case(case, resolved_root, temp_root / str(index))
            for index, case in enumerate(cases)
        ]
    passed = sum(bool(case.get("passed")) for case in results)
    fallback_cases = sum(
        bool(case.get("extraction", {}).get("fallback_used"))
        for case in results
        if isinstance(case.get("extraction"), Mapping)
    )
    try:
        manifest_reference = str(resolved_manifest.relative_to(resolved_root))
    except ValueError:
        manifest_reference = str(resolved_manifest)
    return {
        "schema_version": 1,
        "manifest": manifest_reference,
        "summary": {
            "cases": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "fallback_rate": round(fallback_cases / len(results), 4) if results else 0.0,
        },
        "cases": results,
    }


def main(argv: list[str] | None = None) -> int:
    """Run corpus evaluation and return a process status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/corpus_manifest.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"corpus evaluation: {report['summary']['passed']}/"
        f"{report['summary']['cases']} cases passed"
    )
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
