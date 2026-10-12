from __future__ import annotations

import os
import pathlib
import re
import warnings
from collections.abc import Iterable, Mapping
from functools import reduce
from importlib import import_module
from typing import Any, cast

from pydantic import BaseModel, Field

yaml = cast(Any, import_module("yaml"))


_KNOWN_OPTIONS: dict[str, frozenset[str]] = {
    "pdf_parse": frozenset(
        {
            "engine",
            "retries",
            "exclude_pages",
            "metadata_path",
            "interactive",
            "footer_margin",
            "header_margin",
        }
    ),
    "epub_parse": frozenset({"exclude_pages", "exclude_spine", "metadata_path", "interactive"}),
    "text_parse": frozenset({"exclude_pages", "metadata_path", "interactive"}),
    "text_clean": frozenset(
        {
            "normalize_quotes",
            "repair_ligatures",
            "footer_patterns",
            "interactive_footers",
            "footer_callback",
        }
    ),
    "heading_detect": frozenset(),
    "list_detect": frozenset({"strategy"}),
    "merge_footers": frozenset(),
    "detect_doc_end": frozenset(),
    "detect_page_artifacts": frozenset(
        {"known_footer_patterns", "never_footer_patterns", "interactive", "footer_callback"}
    ),
    "split_semantic": frozenset(
        {
            "chunk_size",
            "overlap",
            "min_chunk_size",
            "target_tokens",
            "hard_page_boundaries",
            "generate_metadata",
            "bullet_strategy",
            "interactive_lists",
            "list_continuation_callback",
        }
    ),
    "ai_enrich": frozenset(
        {
            "enabled",
            "model",
            "api_base",
            "api_key",
            "tags_file",
            "tags_dir",
            "tag_configs",
            "client",
            "fallback_model",
            "fallback_api_base",
            "fallback_api_key",
            "fallback_min_chars",
            "fallback_max_tokens",
            "max_tokens",
            "request_budget",
        }
    ),
    "emit_jsonl": frozenset({"output_path", "drop_meta"}),
    "run_report": frozenset({"output_path"}),
}

_INTEGER_LIMITS: dict[tuple[str, str], tuple[int, int | None]] = {
    ("pdf_parse", "retries"): (0, None),
    ("split_semantic", "chunk_size"): (1, None),
    ("split_semantic", "overlap"): (0, None),
    ("split_semantic", "min_chunk_size"): (1, None),
    ("split_semantic", "target_tokens"): (1, None),
    ("emit_jsonl", "min_words"): (1, None),
    ("emit_jsonl", "max_chars"): (1, None),
    ("emit_jsonl", "target_chunk_chars"): (1, None),
    ("emit_jsonl", "very_short_words"): (1, None),
    ("emit_jsonl", "max_merge_chars"): (1, None),
    ("emit_jsonl", "min_row_words"): (1, None),
    ("emit_jsonl", "critical_short_words"): (1, None),
    ("ai_enrich", "fallback_min_chars"): (0, None),
    ("ai_enrich", "fallback_max_tokens"): (1, None),
    ("ai_enrich", "max_tokens"): (1, None),
}

_EMIT_CONFIG_OPTIONS = frozenset(
    {
        "min_words",
        "metadata_key",
        "max_chars",
        "target_chunk_chars",
        "very_short_words",
        "max_merge_chars",
        "min_row_words",
        "critical_short_words",
        "dedup_debug",
    }
)
_KNOWN_OPTIONS["emit_jsonl"] = _KNOWN_OPTIONS["emit_jsonl"] | _EMIT_CONFIG_OPTIONS


class PipelineSpec(BaseModel):
    """Declarative pipeline specification."""

    pipeline: list[str] = Field(default_factory=list)
    options: dict[str, dict[str, Any]] = Field(default_factory=dict)


def _read_yaml(path: str | os.PathLike | None) -> dict[str, Any]:
    """Return a mapping from YAML, allowing an explicit no-spec ``None`` path."""
    if not path:
        return {}
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"pipeline spec does not exist: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError("pipeline.yaml must contain a top-level mapping")
    return data


def _env_overrides() -> dict[str, dict[str, Any]]:
    """
    Map STEP__key=value → options[step][key]=value (step/key lower-cased).
    Values are YAML-coerced (so 'true', '42' etc. become bool/int).
    """
    out: dict[str, dict[str, Any]] = {}
    pattern = re.compile(r"^[A-Za-z0-9_]+__[A-Za-z0-9_]+$")
    for k, v in os.environ.items():
        if "__" not in k or not pattern.match(k):
            continue
        step, key = k.lower().split("__", 1)
        try:
            val = yaml.safe_load(v)
        except Exception:
            val = v
        out.setdefault(step, {})[key] = val
    return out


def _merge_options(
    base: dict[str, dict[str, Any]], override: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Shallow-merge per-step options with comprehension; override wins."""
    sources = set(base).union(override)
    return {s: {**base.get(s, {}), **override.get(s, {})} for s in sources}


def _warn_unknown_options(pipeline: Iterable[str], opts: Mapping[str, Any]) -> None:
    """Warn when ``opts`` contains non-pipeline steps that are enabled."""
    parser_steps = {"pdf_parse", "epub_parse", "text_parse"}
    declared = set(pipeline)
    parser_family_enabled = bool(declared & parser_steps)

    unknown = [
        step
        for step, cfg in opts.items()
        if step
        and step not in pipeline
        and not (parser_family_enabled and step in parser_steps)
        and not (isinstance(cfg, Mapping) and cfg.get("enabled") is False)
    ]
    if unknown:
        warnings.warn(
            f"Unknown pipeline options: {', '.join(sorted(unknown))}",
            stacklevel=2,
        )


def _validate_known_options(opts: Mapping[str, Any]) -> None:
    """Reject misspelled options and invalid numeric ranges before execution."""
    for step, config in opts.items():
        if step not in _KNOWN_OPTIONS or not isinstance(config, Mapping):
            continue
        unknown = sorted(set(config) - _KNOWN_OPTIONS[step])
        if unknown:
            raise ValueError(f"unknown options for {step}: {', '.join(unknown)}")

        for name, (minimum, maximum) in _INTEGER_LIMITS.items():
            if name[0] != step or name[1] not in config or config[name[1]] is None:
                continue
            value = config[name[1]]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{step}.{name[1]} must be an integer")
            if value < minimum or (maximum is not None and value > maximum):
                bound = (
                    f"at least {minimum}" if maximum is None else f"between {minimum} and {maximum}"
                )
                raise ValueError(f"{step}.{name[1]} must be {bound}")

        for name in ("footer_margin", "header_margin"):
            if name in config and config[name] is not None:
                value = config[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{step}.{name} must be a number")
                if value < 0:
                    raise ValueError(f"{step}.{name} must not be negative")

        if step == "split_semantic":
            chunk_size = config.get("chunk_size")
            overlap = config.get("overlap")
            if isinstance(chunk_size, int) and isinstance(overlap, int) and overlap >= chunk_size:
                raise ValueError("split_semantic.overlap must be smaller than chunk_size")


def validate_options(opts: Mapping[str, Any]) -> None:
    """Validate pass option names and ranges for programmatic callers."""
    _validate_known_options(opts)


def load_spec(
    path: str | os.PathLike | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> PipelineSpec:
    """Load YAML + env/CLI overrides into a validated PipelineSpec."""
    data = _read_yaml(path)
    opts = data.get("options", {})
    sources: Iterable[dict[str, dict[str, Any]]] = (
        d for d in (opts, _env_overrides(), overrides) if d
    )
    acc: dict[str, dict[str, Any]] = {}
    merged = reduce(_merge_options, sources, acc)
    pipeline = data.get("pipeline", [])
    _warn_unknown_options(pipeline, merged)
    _validate_known_options(merged)
    if merged:
        data = {**data, "options": merged}
    return PipelineSpec.model_validate(data)
