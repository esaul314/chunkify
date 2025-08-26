from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Protocol, Type, runtime_checkable


@dataclass(frozen=True)
class Artifact:
    """Immutable carrier of data + metadata between passes."""

    payload: Any
    meta: Dict[str, Any] | None = None


@runtime_checkable
class Pass(Protocol):
    name: str
    input_type: Type
    output_type: Type

    def __call__(self, a: Artifact) -> Artifact:
        """Execute the pass."""
        ...


_REGISTRY: Mapping[str, Pass] = MappingProxyType({})


def register(p: Pass) -> Pass:
    """Register a pass by name; idempotent for same object."""
    global _REGISTRY
    _REGISTRY = MappingProxyType({**dict(_REGISTRY), p.name: p})
    return p


def run_step(name: str, a: Artifact) -> Artifact:
    """Run a single registered step."""
    return _REGISTRY[name](a)


def run_pipeline(steps: List[str], a: Artifact) -> Artifact:
    """Apply registered steps in order."""
    return reduce(lambda acc, s: run_step(s, acc), steps, a)


def registry() -> Dict[str, Pass]:
    """Shallow copy of the registry for inspection/testing."""
    return dict(_REGISTRY)
