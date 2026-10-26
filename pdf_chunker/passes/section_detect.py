"""Opt-in structural section classification; source I/O belongs to adapters."""

from collections.abc import Mapping

from pdf_chunker.framework import Artifact, register
from pdf_chunker.sections import classify_document


class _SectionDetectPass:
    name = "section_detect"
    input_type = dict
    output_type = dict

    def __call__(self, a: Artifact[object]) -> Artifact[object]:
        artifact = a
        all_options = (artifact.meta or {}).get("options")
        options = all_options.get(self.name, {}) if isinstance(all_options, Mapping) else {}
        if not isinstance(options, Mapping):
            raise TypeError("section_detect options must be a mapping")
        if not options.get("enabled", False):
            return artifact
        doc = artifact.payload
        if not isinstance(doc, Mapping) or doc.get("type") != "page_blocks":
            return artifact
        settings = {
            k: v
            for k, v in options.items()
            if k not in {"answers", "callback", "answers_path", "questions_path", "interactive"}
        }
        result, pending, decisions = classify_document(
            doc,
            settings=settings,
            answers=options.get("answers", ()),
            callback=options.get("callback"),
        )
        return Artifact(
            payload=result,
            meta={
                **(artifact.meta or {}),
                "section_questions": pending,
                "section_decisions": decisions,
            },
        )


section_detect = register(_SectionDetectPass())
