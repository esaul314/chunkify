"""Allowlist for parity tests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Tuple

Key = Tuple[str, Tuple[str, ...], str]
Rule = Dict[str, Any]

EXCEPTIONS: Dict[Key, Rule] = {
    (
        "test_e2e_parity_flags",
        ("--chunk-size", "200", "--overlap", "10"),
        "tiny.pdf",
    ): {"ignore": ["meta", "metadata"]},
    (
        "test_e2e_parity_flags",
        ("--no-metadata",),
        "tiny_a.pdf",
    ): {"ignore": []},
    (
        "test_no_metadata_rows_contain_only_text",
        ("--no-metadata",),
        "tiny_b.pdf",
    ): {"ignore": []},
    (
        "test_new_matches_legacy",
        (),
        "tiny.pdf",
    ): {"ignore": ["metadata"]},
    (
        "test_new_matches_legacy",
        (),
        "tiny_a.pdf",
    ): {"ignore": ["metadata"]},
    (
        "test_new_matches_legacy",
        (),
        "tiny_b.pdf",
    ): {"ignore": ["metadata"]},
}


def key(test: str, flags: Iterable[str], fixture: str) -> Key:
    return (test, tuple(flags), fixture)


def get(test: str, flags: Iterable[str], fixture: str) -> Rule | None:
    return EXCEPTIONS.get(key(test, flags, fixture))


def apply(rows: List[Dict[str, Any]], rule: Rule | None) -> List[Dict[str, Any]]:
    if not rule:
        return rows
    ignore = set(rule.get("ignore", []))
    cleaned = [{k: v for k, v in row.items() if k not in ignore} for row in rows]
    return rule.get("expect", cleaned)
