"""Pure, copy-on-write updates to per-pass artifact metrics."""

from collections.abc import Mapping


def with_metrics(
    meta: Mapping[str, object] | None,
    *,
    pass_name: str,
    updates: Mapping[str, object],
) -> dict[str, object]:
    """Merge metrics without mutating the input or replacing unrelated entries.

    Return new containers along the updated path; untouched values remain shared
    until the receiving ``Artifact`` detaches nested metadata. Updates take
    precedence over existing metrics for this pass. Keyword-only settings allow
    partial application to create a reusable, data-first metadata transform.
    Malformed metric containers raise ``TypeError`` rather than losing metrics.
    """
    base = meta or {}
    metrics = base.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise TypeError("artifact metrics must be a mapping")
    existing = metrics.get(pass_name, {})
    if not isinstance(existing, Mapping):
        raise TypeError(f"metrics for pass {pass_name!r} must be a mapping")
    return {
        **base,
        "metrics": {**metrics, pass_name: {**existing, **updates}},
    }
