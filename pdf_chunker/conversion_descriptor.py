"""Pure, explicit article conversion identity; no cache or proof authority.

Callers supply the complete resolved configuration and pinned implementation,
normalization, chunk-schema and serialization versions. No defaults, environment,
files or model configuration are discovered here. Snapshots detach mutable input;
mapping order is irrelevant, while sequence order and Unicode remain exact.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal


def _json_value(value: object) -> object:
    """Detach JSON containers, rejecting coercions and opaque collaborators."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("descriptor JSON requires string mapping keys")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("descriptor requires JSON values, not opaque collaborators")


def _snapshot(value: object) -> str:
    snapshot = json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    snapshot.encode("utf-8")  # Refuse unpaired surrogates rather than hash invalid UTF-8.
    return snapshot


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _label(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"nonempty {name} required")


def _contract_snapshot(contract: Mapping[str, object]) -> str:
    versions = ("implementation", "normalization", "chunk_schema", "serialization")
    if set(contract) != {*versions, "configuration"}:
        raise ValueError("contract requires four versions and resolved configuration")
    for name in versions:
        _label(contract[name], f"contract {name}")
    if not isinstance(contract["configuration"], Mapping):
        raise ValueError("contract configuration must be a resolved mapping")
    return _snapshot(contract)


def _read_mapping_snapshot(data: str) -> dict[str, Any]:
    """Read exact canonical JSON, refusing duplicates, coercions and invalid Unicode."""
    if not isinstance(data, str):
        raise ValueError("canonical JSON string required")
    value = json.loads(data)
    if not isinstance(value, dict) or _snapshot(value) != data:
        raise ValueError("canonical JSON object required")
    return value


def _require_sha256(value: object) -> None:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("lowercase SHA-256 required")


@dataclass(frozen=True)
class NoteAssociation:
    """One logical reference and its content revision, routing and full provenance."""

    reference: str
    content_sha256: str
    corpus: str
    provenance_json: str

    @property
    def key(self) -> str:
        """Fingerprint this association independently of other notes/physical files."""
        return _digest(_snapshot({"schema": 1, **asdict(self)}))


@dataclass(frozen=True)
class ConversionDescriptor:
    """Immutable input identity, separate from output validation or chunk IDs.

    JSON snapshots retain all supplied metadata/configuration. The complete key
    includes physical provenance because today's rows and IDs still depend on it.
    An equal key alone never authorizes skipping evidence or cached-output checks.
    """

    source_path: str
    source_format: str
    source_sha256: str
    content_sha256: str
    contract_json: str
    metadata_json: str
    associations: tuple[NoteAssociation, ...]

    def to_json(self) -> str:
        """Return compact sorted UTF-8-compatible descriptor JSON (schema 1)."""
        return _snapshot({"schema": 1, **asdict(self)})

    @property
    def key(self) -> str:
        """Fingerprint every input represented by this versioned descriptor."""
        return _digest(self.to_json())


def _read_association(value: object, *, content_sha256: str) -> NoteAssociation:
    if not isinstance(value, dict) or set(value) != {f.name for f in fields(NoteAssociation)}:
        raise ValueError("complete note association required")
    for name in ("reference", "corpus"):
        _label(value[name], f"association {name}")
    if value["content_sha256"] != content_sha256:
        raise ValueError("association content differs from descriptor")
    _read_mapping_snapshot(value["provenance_json"])
    return NoteAssociation(**value)


def read_conversion_descriptor(data: str) -> ConversionDescriptor:
    """Restore schema-1 canonical snapshots, without granting proof or completion.

    Exact round trips reject duplicate/extra keys, alternate serialization and
    malformed nested snapshots. Associations must be nonempty, unique and sorted,
    sharing the descriptor's content revision and a single corpus. No files or
    effective runtime settings are discovered; fresh preparation remains necessary.
    """
    value = _read_mapping_snapshot(data)
    if (
        set(value) != {"schema", *(f.name for f in fields(ConversionDescriptor))}
        or type(value["schema"]) is not int
        or value["schema"] != 1
    ):
        raise ValueError("complete conversion descriptor schema 1 required")
    for name in ("source_path", "source_format"):
        _label(value[name], name)
    for name in ("source_sha256", "content_sha256"):
        _require_sha256(value[name])
    _contract_snapshot(_read_mapping_snapshot(value["contract_json"]))
    _read_mapping_snapshot(value["metadata_json"])
    items = value["associations"]
    if not isinstance(items, list) or not items:
        raise ValueError("nonempty descriptor associations required")
    associations = tuple(
        _read_association(item, content_sha256=value["content_sha256"]) for item in items
    )
    references = [item.reference for item in associations]
    if references != sorted(set(references)) or len({item.corpus for item in associations}) != 1:
        raise ValueError("unique sorted associations with one corpus required")
    return ConversionDescriptor(
        **{key: item for key, item in value.items() if key not in {"schema", "associations"}},
        associations=associations,
    )


def describe_conversion(
    text: str,
    *,
    source_path: str,
    source_format: str,
    source_sha256: str,
    metadata: Mapping[str, object],
    references: Mapping[str, Mapping[str, object]],
    corpus: str,
    contract: Mapping[str, object],
) -> ConversionDescriptor:
    """Snapshot supplied article data, data-first and usable with ``partial``.

    This pure builder does not verify source bytes, approval, or completeness of
    caller-resolved settings. Use the verified adapter for bound Joplin articles.
    References are sorted by logical identity; each retains its whole provenance.
    Their metadata is deliberately conservative: sequence order is significant.
    """
    for name, value in (
        ("source path", source_path),
        ("source format", source_format),
        ("corpus", corpus),
    ):
        _label(value, name)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("nonempty article text required")
    if not isinstance(source_sha256, str) or re.fullmatch("[0-9a-f]{64}", source_sha256) is None:
        raise ValueError("lowercase source SHA-256 required")
    if not isinstance(metadata, Mapping) or not isinstance(references, Mapping) or not references:
        raise ValueError("metadata and nonempty reference mappings required")
    for reference, provenance in references.items():
        _label(reference, "logical reference")
        if not isinstance(provenance, Mapping):
            raise ValueError("reference provenance must be a mapping")
    content_sha256 = _digest(text)
    associations = tuple(
        NoteAssociation(reference, content_sha256, corpus, _snapshot(provenance))
        for reference, provenance in sorted(references.items())
    )
    return ConversionDescriptor(
        source_path,
        source_format,
        source_sha256,
        content_sha256,
        _contract_snapshot(contract),
        _snapshot(metadata),
        associations,
    )


@dataclass(frozen=True)
class ConversionChange:
    """Comparison only: no cache admission, reference removal or write actions."""

    status: Literal["new", "changed", "unchanged"]
    changed_fields: tuple[str, ...]


def compare_conversion(
    current: ConversionDescriptor, *, previous: ConversionDescriptor | None = None
) -> ConversionChange:
    """Compare paired source snapshots in stable field order without side effects.

    The coordinator owns pairing a source across runs. Absence from a descriptor
    does not authorize deleting any reference, cache entry or downstream point.
    """
    changed = tuple(
        field.name
        for field in fields(current)
        if previous is None or getattr(current, field.name) != getattr(previous, field.name)
    )
    status: Literal["new", "changed", "unchanged"] = (
        "new" if previous is None else "changed" if changed else "unchanged"
    )
    return ConversionChange(status, changed)
