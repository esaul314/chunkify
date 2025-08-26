from __future__ import annotations

import os
import pathlib
import warnings
from functools import reduce
from typing import Any, Dict, Iterable, Mapping, List

import yaml
from pydantic import BaseModel, Field


class PipelineSpec(BaseModel):
    """Declarative pipeline specification."""

    pipeline: List[str] = Field(default_factory=list)
    options: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


def _read_yaml(path: str | os.PathLike | None) -> Dict[str, Any]:
    """Return a dict from YAML or {} if path is None/missing/empty."""
    if not path:
        return {}
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError("pipeline.yaml must contain a top-level mapping")
    return data


def _env_overrides() -> Dict[str, Dict[str, Any]]:
    """
    Map STEP__key=value → options[step][key]=value (step/key lower-cased).
    Values are YAML-coerced (so 'true', '42' etc. become bool/int).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in os.environ.items():
        if "__" not in k:
            continue
        step, key = k.lower().split("__", 1)
        try:
            val = yaml.safe_load(v)
        except Exception:
            val = v
        out.setdefault(step, {})[key] = val
    return out


def _merge_options(
    base: Dict[str, Dict[str, Any]], override: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Shallow-merge per-step options with comprehension; override wins."""
    return {s: {**base.get(s, {}), **override.get(s, {})} for s in set(base) | set(override)}


def _warn_unknown_options(pipeline: Iterable[str], opts: Mapping[str, Any]) -> None:
    """Emit a warning when options contain steps absent from the pipeline."""

    unknown = [step for step in opts if step not in pipeline]
    if unknown:
        warnings.warn(
            f"Unknown pipeline options: {', '.join(sorted(unknown))}",
            stacklevel=2,
        )


def load_spec(
    path: str | os.PathLike | None = "pipeline.yaml",
    overrides: Dict[str, Dict[str, Any]] | None = None,
) -> PipelineSpec:
    """Load YAML + env/CLI overrides into a validated PipelineSpec."""
    data = _read_yaml(path)
    opts = data.get("options", {})
    merged = reduce(
        _merge_options,
        filter(None, [opts, _env_overrides(), overrides]),
        {},
    )
    pipeline = data.get("pipeline", [])
    _warn_unknown_options(pipeline, merged)
    if merged:
        data = {**data, "options": merged}
    return PipelineSpec.model_validate(data)
