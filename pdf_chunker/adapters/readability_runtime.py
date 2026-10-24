"""Inspect readability resources used by the bounded base article profile.

Only this adapter reads resource files or observes process caches. Fingerprints
use detached canonical JSON; normal cache warming is deliberately identity-neutral.
No dictionary, cache entry, environment setting or file is repaired or replaced.
"""

from __future__ import annotations

import hashlib
from copy import copy
from importlib import import_module
from pathlib import Path

import pyphen

from pdf_chunker import utils
from pdf_chunker.conversion_descriptor import _snapshot


def _state_hash(state: object) -> str:
    """Hash explicit in-memory values without depending on mapping order."""
    return hashlib.sha256(_snapshot(state).encode("utf-8")).hexdigest()


def collect_readability_runtime() -> dict[str, object]:
    """Pin raw dictionaries and effective loaded tables; reject stale word caches.

    CMU is optional and its loader may have fallen back to ``None``. Record that
    effective state independently of raw resource availability. Pyphen positions
    are checked against the same loaded patterns in an isolated cache, so earlier
    documents cannot change identity merely by warming the shared cache. Position
    alternatives do not affect readability, which only counts the filtered points.

    The first call may load the runner's lazy dictionaries. Subsequent inspection
    does not clear caches or require loaded tables to equal today's resource bytes:
    both are pinned independently, reflecting what the runner will actually use.
    This snapshot does not guard concurrent mutation or attest imported code.
    """
    language = utils._READABILITY_LANG
    pronunciations = utils._get_cmudict(language)
    hyphenator = utils._get_pyphen(language)
    if type(hyphenator) is not pyphen.Pyphen or type(hyphenator.hd) is not pyphen.HyphDict:
        raise ValueError("canonical readability hyphenator required")

    # Reuse Pyphen's own computation, with the loaded patterns but no shared
    # memoized positions. Constructing Pyphen(cache=False) replaces its global
    # dictionary cache, so copy only the existing HyphDict and detach its cache.
    isolated = copy(hyphenator.hd)
    isolated.cache = {}
    if any(points != isolated.positions(word) for word, points in hyphenator.hd.cache.items()):
        raise ValueError("readability hyphenation cache differs from loaded patterns")

    cmu_resource_hash = None
    if utils._load_cmudict is not None:
        with import_module("cmudict").dict_stream() as stream:
            cmu_resource_hash = hashlib.sha256(stream.read()).hexdigest()
    hyphen_path = Path(pyphen.LANGUAGES[pyphen.language_fallback(language)])
    return {
        "schema": 1,
        "language": language,
        "cmudict": {
            "available": pronunciations is not None,
            "resource_sha256": cmu_resource_hash,
            "loaded_sha256": _state_hash(pronunciations),
        },
        "pyphen": {
            "resource_sha256": hashlib.sha256(hyphen_path.read_bytes()).hexdigest(),
            "loaded_sha256": _state_hash(
                {
                    "patterns": hyphenator.hd.patterns,
                    "max_length": hyphenator.hd.maxlen,
                    "left": hyphenator.left,
                    "right": hyphenator.right,
                }
            ),
        },
    }
