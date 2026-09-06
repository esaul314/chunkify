from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar, runtime_checkable

PayloadT_co = TypeVar("PayloadT_co", covariant=True)
Meta = dict[str, object]


def _copy_metadata_value(value: object) -> object:
    """Copy containers while preserving opaque collaborators such as clients."""
    if isinstance(value, Mapping):
        return {key: _copy_metadata_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_metadata_value(item) for item in value)
    if isinstance(value, set):
        return {_copy_metadata_value(item) for item in value}
    return value


@dataclass(frozen=True)
class Artifact(Generic[PayloadT_co]):
    """Immutable carrier of data + metadata between passes."""

    payload: PayloadT_co
    meta: Meta | None = None

    def __post_init__(self) -> None:
        """Detach nested metadata from callers before it enters the pipeline."""
        if self.meta is not None:
            object.__setattr__(self, "meta", _copy_metadata_value(self.meta))


@runtime_checkable
class Pass(Protocol):
    name: str

    @property
    def input_type(self) -> type: ...

    @property
    def output_type(self) -> type: ...

    def __call__(self, a: Artifact[object]) -> Artifact[object]:
        """Execute the pass."""
        ...


_REGISTRY: Mapping[str, Pass] = MappingProxyType({})


def register(p: Pass) -> Pass:
    """Register a pass by name, rejecting accidental replacement."""
    global _REGISTRY
    if not p.name:
        raise ValueError("pass name must not be empty")
    existing = _REGISTRY.get(p.name)
    if existing is not None and existing is not p:
        raise ValueError(f"pass name collision: {p.name!r} is already registered")
    _REGISTRY = MappingProxyType({**dict(_REGISTRY), p.name: p})
    return p


def run_step(name: str, a: Artifact[object]) -> Artifact[object]:
    """Run a single registered step with an actionable unknown-step error."""
    try:
        step = _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown pipeline step {name!r}; available: {available}") from exc
    return step(a)


def run_pipeline(steps: list[str], a: Artifact[object]) -> Artifact[object]:
    """Apply registered steps in order."""
    return reduce(lambda acc, s: run_step(s, acc), steps, a)


def registry() -> dict[str, Pass]:
    """Shallow copy of the registry for inspection/testing."""
    return dict(_REGISTRY)
