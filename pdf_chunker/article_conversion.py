"""Pure settings resolution for the bounded verified-article base pipeline.

Defaults arrive explicitly from the runner boundary. No environment, evidence,
version or filesystem discovery occurs here; execution stays in ``core_new``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from pdf_chunker.config import PipelineSpec, validate_options
from pdf_chunker.conversion_descriptor import ConversionDescriptor, _snapshot
from pdf_chunker.passes.chunk_options import SplitOptions
from pdf_chunker.passes.emit_jsonl import EmitConfig

_PARSERS = frozenset({"pdf_parse", "epub_parse", "text_parse"})


def _base_steps(pipeline: Sequence[str], parser: str) -> list[str]:
    steps = [parser if step in _PARSERS else step for step in pipeline]
    base = [parser, "text_clean", "split_semantic", "emit_jsonl"]
    with_headings = [parser, "text_clean", "heading_detect", "split_semantic", "emit_jsonl"]
    if steps not in (base, with_headings) or len(set(pipeline) & _PARSERS) != 1:
        raise ValueError("expected qualified base article pipeline")
    return steps


def _validate_profile(options: Mapping[str, Mapping[str, object]], steps: Sequence[str]) -> None:
    for step, config in options.items():
        if step not in {*steps, *_PARSERS, "ai_enrich"}:
            raise ValueError(f"unqualified article options: {step}")
        if step == "ai_enrich" and config.get("enabled") is not False:
            raise ValueError("base article preparation requires enrichment disabled")
        for key, value in config.items():
            if value and (
                key.startswith("interactive")
                or key.endswith("callback")
                or key in {"client", "bullet_strategy", "api_key", "fallback_api_key"}
                or (
                    step in _PARSERS
                    and key in {"exclude_pages", "exclude_spine", "footer_margin", "header_margin"}
                )
            ):
                raise ValueError(f"unqualified article setting: {step}.{key}")


def resolve_article_spec(
    spec: PipelineSpec,
    *,
    suffix: str,
    emit_defaults: EmitConfig,
    split_defaults: SplitOptions,
) -> PipelineSpec:
    """Detach and resolve a base spec, data-first and usable with ``partial``.

    The qualified composition is parser → text_clean → optional heading_detect →
    split_semantic → emit_jsonl. Proof-bound text bypasses ordinary cleaning and
    semantic heuristics. Retain even inert supplied options conservatively, but
    refuse interaction, extraction exclusions, opaque collaborators and AI work.
    Environment/CLI spec overrides must already have been loaded by the caller.
    """
    detached = json.loads(_snapshot(spec.model_dump()))
    validate_options(detached["options"])
    parser = {".md": "text_parse", ".epub": "epub_parse"}.get(suffix.lower())
    if parser is None:
        raise ValueError("resolved article source must be .md or .epub")
    steps = _base_steps(detached["pipeline"], parser)
    options = detached["options"]
    _validate_profile(options, steps)
    split = options.get("split_semantic", {})
    if "min_chunk_size" in split and split["min_chunk_size"] is None:
        raise ValueError("explicit null min_chunk_size is not executable by the runner")
    resolved_split = split_defaults.with_meta({"options": {"split_semantic": split}})
    emit = {**asdict(emit_defaults), **options.get("emit_jsonl", {})}
    if not isinstance(emit["metadata_key"], str) or emit["metadata_key"] in {"", "text"}:
        raise ValueError("article metadata_key must be a nonempty key distinct from text")
    resolved = PipelineSpec(
        pipeline=steps,
        options={
            **options,
            "split_semantic": {**split, **asdict(resolved_split)},
            "emit_jsonl": emit,
        },
    )
    validate_options(resolved.options)
    return resolved


@dataclass(frozen=True)
class PreparedArticleConversion:
    """Detached descriptor and executable spec; neither is a completion receipt."""

    descriptor: ConversionDescriptor
    spec_json: str

    def spec(self) -> PipelineSpec:
        """Return a fresh spec so caller mutations cannot alter the snapshot."""
        return PipelineSpec.model_validate(json.loads(self.spec_json))
