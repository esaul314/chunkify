"""Pure, document-bound structural evidence and review decisions.

No text is removed. Roles are separate from enrichment categories. Automatic
non-body roles require independent signals; conflicting evidence stays pending.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from statistics import median
from typing import Any

from pdf_chunker.inline_styles import normalize_spans
from pdf_chunker.interactive import Decision, DecisionContext, DecisionKind

DETECTOR_VERSION = "section-v3"
ROLES = frozenset(
    {"body", "contents", "acknowledgments", "permissions", "references", "index", "unknown"}
)
_ROLE_TITLES = {
    "contents": r"(?:table of )?contents|list of (?:figures|tables)",
    "acknowledgments": r"acknowledg(?:e)?ments",
    "permissions": r"permissions|copyright|copyright acknowledg(?:e)?ments",
    "references": r"references|bibliography|works cited|further reading",
    "index": r"(?:subject |author )?index",
}
_BODY_TITLE = re.compile(r"^(?:introduction|preface|foreword|(?:part|chapter)\b.*)$", re.I)


def digest(value: Any) -> str:
    """Stable JSON identity, independent of mapping insertion order."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()


def title_role(text: str) -> str | None:
    """Recognize complete structural titles, never prose mentioning a role."""
    cleaned = re.sub(r"\s+", " ", text).strip(" \t0123456789.")
    return next(
        (role for role, pattern in _ROLE_TITLES.items() if re.fullmatch(pattern, cleaned, re.I)),
        None,
    )


def block_identity(page: Any, index: int, block: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the exact extraction block and its original coordinates."""
    return {
        "page": page,
        "block_index": block.get("_section_block_index", index),
        "text_sha256": hashlib.sha256(str(block.get("text", "")).encode()).hexdigest(),
        "bbox": list(block.get("bbox") or ()),
        **(
            {"raw_source": block["source"]["raw_source"]}
            if (block.get("source") or {}).get("raw_source")
            else {}
        ),
        **(
            {"source_page_range": list(block["source"]["page_range"])}
            if (block.get("source") or {}).get("page_range") is not None
            else {}
        ),
        **({"span": block["_section_span"]} if "_section_span" in block else {}),
        **({"kind": block["_section_kind"]} if "_section_kind" in block else {}),
        **(
            {"parent_text_sha256": block["_section_parent_sha256"]}
            if "_section_parent_sha256" in block
            else {}
        ),
    }


def _entry_count(text: str) -> int:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return sum(
        bool(re.search(r"(?:\.{2,}|\t|\s{2,})\s*(?:\d+|[ivxlcdm]+)$", line, re.I))
        or bool(re.fullmatch(r"\d+(?:[-–]\d+)?|[ivxlcdm]+", line, re.I))
        for line in lines
    )


def _support(role: str, text: str) -> list[str]:
    if role == "contents":
        return ["entry_page_numbers"] if _entry_count(text) >= 3 else []
    if role == "acknowledgments":
        return (
            ["gratitude_prose"]
            if re.search(
                r"\b(?:thank(?:s|ful)?|grateful|gratitude|owe|acknowledge|indebted)\b", text, re.I
            )
            else []
        )
    if role == "permissions":
        count = len(
            re.findall(
                r"\b(?:permission|copyright|reprint|reproduc|rights reserved|"
                r"publisher|press|inc\.|company)\w*",
                text,
                re.I,
            )
        )
        return ["rights_and_publisher_language"] if count >= 2 else []
    if role == "references":
        count = len(
            re.findall(r"(?:\(?(?:18|19|20)\d{2}[a-z]?\)?|" r"\b(?:doi|isbn|pp\.)\b)", text, re.I)
        )
        return ["citation_entries"] if count >= 3 else []
    if role == "index":
        index_entries = len(
            re.findall(r"(?m)^[^\d\n][^\n]*\s\d+(?:[–,-]\d+)*(?:,\s*\d+(?:[–-]\d+)*)*\s*$", text)
        )
        return ["alphabetical_page_entries"] if max(_entry_count(text), index_entries) >= 3 else []
    return []


def _prose(text: str) -> bool:
    return len(text.split()) >= 30 and len(re.findall(r"[.!?](?:\s|$)", text)) >= 2


def _outline_for(page: int, outline: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    # Stable last-entry precedence allows nested navigation on the same page.
    entries = [
        entry for entry in outline if isinstance(entry.get("page"), int) and entry["page"] <= page
    ]
    return (
        max(enumerate(entries), key=lambda item: (item[1]["page"], item[0]))[1] if entries else None
    )


def _typographic_title(raw: Mapping[str, Any], titles: set[str]) -> bool:
    blocks = raw.get("blocks") or ()
    sizes = [size for b in blocks for _, size in b.get("fonts", ()) if size]
    baseline = median(sizes) if sizes else 0
    font_counts: dict[str, int] = {}
    for block in blocks:
        for font, _ in block.get("fonts", ()):
            font_counts[str(font)] = font_counts.get(str(font), 0) + len(str(block.get("text", "")))
    dominant = max(font_counts, key=lambda font: font_counts[font]) if font_counts else None
    for block in blocks:
        text = re.sub(r"\s+", " ", str(block.get("text", ""))).strip().casefold()
        if text not in titles:
            continue
        fonts = block.get("fonts") or ()
        if any(
            (
                "bold" in str(font).casefold()
                or (baseline and size >= baseline * 1.15)
                or (dominant is not None and str(font) != dominant)
            )
            for font, size in fonts
        ):
            return True
    return False


def _known_title(line: str) -> bool:
    return bool(
        title_role(line) or re.fullmatch(r"introduction|preface|foreword", line.strip(), re.I)
    )


def _page_parts(
    page: Mapping[str, Any], raw: Mapping[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Partition only at exact standalone known heading lines, preserving spans.

    Mixed paragraphs with merely mentioned role words remain untouched. Multiple
    candidate boundaries carry exact extraction-block indices and source spans.
    """
    groups: list[list[dict[str, Any]]] = [[]]
    for index, block in enumerate(page.get("blocks", ())):
        text = str(block.get("text", ""))
        starts = [
            m.start()
            for m in re.finditer(r"(?m)^.+$", text)
            if _known_title(m.group().strip())
            and not re.match(r"^\s*(?:\d+|[ivxlcdm]+)\s*(?:\n|$)", text[m.end() :], re.I)
        ]
        cuts = sorted({0, len(text), *starts})
        for start, end in zip(cuts, cuts[1:], strict=False):
            piece = text[start:end]
            if not piece.strip():
                continue
            first = next((line.strip() for line in piece.splitlines() if line.strip()), "")
            if _known_title(first) and groups[-1]:
                groups.append([])
            spans = tuple(
                replace(span, start=max(span.start, start) - start, end=min(span.end, end) - start)
                for span in normalize_spans(block.get("inline_styles") or (), len(text))
                if min(span.end, end) > max(span.start, start)
            )
            groups[-1].append(
                {
                    **block,
                    "text": piece,
                    "inline_styles": list(spans),
                    "_section_block_index": index,
                    "_section_span": [start, end],
                    "_section_parent_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
    groups = [g for g in groups if g]
    if len(groups) <= 1:
        return [({**page, "blocks": groups[0] if groups else []}, dict(raw))]
    original = str(raw.get("text") or "")
    anchors = [0]
    for group in groups[1:]:
        first = next(line.strip() for line in group[0]["text"].splitlines() if line.strip())
        matches = list(re.finditer(r"(?im)^\s*" + re.escape(first) + r"\s*$", original))
        match = next((m for m in matches if m.start() > anchors[-1]), None)
        anchors.append(match.start() if match else -1)
    valid = all(anchor >= 0 for anchor in anchors)
    result = []
    for index, group in enumerate(groups):
        part_text = (
            original[
                anchors[index] : (anchors[index + 1] if index + 1 < len(anchors) else len(original))
            ]
            if valid
            else "\n".join(b["text"] for b in group)
        )
        result.append(
            (
                {**page, "blocks": group, "_section_internal_boundary": index > 0},
                {**raw, "text": part_text},
            )
        )
    return result


def page_evidence(
    page: Mapping[str, Any],
    raw: Mapping[str, Any],
    outline: Sequence[Mapping[str, Any]],
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Combine original layout, navigation, heading and continuation evidence."""
    text = str(raw.get("text") or "\n".join(str(b.get("text", "")) for b in page.get("blocks", ())))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    headings = [(line, title_role(line)) for line in lines[:8]]
    heading_roles = {role for _, role in headings if role}
    nav = _outline_for(int(page.get("page") or 0), outline)
    nav_role = title_role(str(nav.get("title", ""))) if nav else None
    candidates = set(heading_roles)
    if nav_role:
        candidates.add(nav_role)
    role_typography = {
        role
        for role in candidates
        if _typographic_title(
            raw,
            {
                re.sub(r"\s+", " ", line).casefold()
                for line, candidate in headings
                if candidate == role
            },
        )
    }
    body_heading = any(_BODY_TITLE.fullmatch(line) for line in lines[:4])
    if body_heading and _prose(text) and nav_role and nav and nav["page"] < page.get("page", 0):
        return {
            "role": "unknown",
            "signals": ["body_heading", "substantive_prose", "conflicting_outline_range"],
            "status": "pending",
            "candidate_roles": sorted({"body", nav_role}),
        }
    detected: list[tuple[str, list[str]]] = []
    for role in sorted(candidates):
        signals = _support(role, text)
        if role in heading_roles:
            signals.append("section_heading")
        if nav_role == role:
            signals.append("outline_range")
        if role in role_typography:
            signals.append("heading_typography")
        if len(signals) >= 2:
            detected.append((role, signals))
    if len(detected) == 1:
        role, signals = detected[0]
        return {
            "role": role,
            "signals": signals,
            "status": "detected",
            "candidate_roles": sorted(candidates),
        }
    if len(detected) > 1:
        return {
            "role": "unknown",
            "signals": ["conflicting_structure"],
            "status": "pending",
            "candidate_roles": sorted(candidates),
        }
    # A verified outline range continues until its next actual navigation entry.
    if (
        nav_role
        and previous
        and previous["role"] == nav_role
        and nav
        and nav["page"] < page.get("page", 0)
    ):
        support = _support(nav_role, text)
        if text.strip():
            return {
                "role": nav_role,
                "signals": ["outline_range", "section_continuity", *support],
                "status": "detected",
                "candidate_roles": [nav_role],
            }
    if candidates:
        return {
            "role": "unknown",
            "signals": ["unconfirmed_section_title"],
            "status": "pending",
            "candidate_roles": sorted(candidates),
        }
    body_heading = any(_BODY_TITLE.fullmatch(line) for line in lines[:4])
    if body_heading and _prose(text):
        return {
            "role": "body",
            "signals": [
                "body_heading" if body_heading else "heading_typography",
                "substantive_prose",
            ],
            "status": "detected",
            "candidate_roles": [],
        }
    if previous and previous["role"] in {"contents", "references", "index"}:
        support = _support(previous["role"], text)
        if support:
            return {
                "role": previous["role"],
                "signals": ["section_continuity", *support],
                "status": "detected",
                "candidate_roles": [previous["role"]],
            }
    body_heading = any(_BODY_TITLE.fullmatch(line) for line in lines[:4])
    body_nav = nav is not None and nav_role is None and _prose(text)
    if (body_heading and _prose(text)) or body_nav:
        return {
            "role": "body",
            "signals": ["body_heading" if body_heading else "body_outline", "substantive_prose"],
            "status": "detected",
            "candidate_roles": [],
        }
    if previous and previous["role"] == "body" and _prose(text):
        return {
            "role": "body",
            "signals": ["section_continuity", "substantive_prose"],
            "status": "detected",
            "candidate_roles": [],
        }
    return {"role": "unknown", "signals": [], "status": "detected", "candidate_roles": []}


def _raw_covers_extracted(raw_text: str, blocks: Sequence[Mapping[str, Any]], page: Any) -> bool:
    """Recover only when physical page text covers every original extraction token."""
    for block in blocks:
        source = block.get("source") or {}
        if any(value != page for value in (source.get("page_range") or ())):
            return False
        if any(item.get("page") != page for item in (source.get("block_provenance") or ())):
            return False
    from pdf_chunker.preservation import tokens as source_tokens

    original = iter(token.text for token in source_tokens(raw_text.casefold()))
    extracted = "\n".join(str(block.get("text", "")) for block in blocks)
    return all(
        any(candidate == token.text for candidate in original)
        for token in source_tokens(extracted.casefold())
    )


def make_question(
    kind: str,
    binding: Mapping[str, Any],
    blocks: list[dict[str, Any]],
    evidence: Mapping[str, Any],
    text: str,
    previous_text: str,
    extracted_text: str = "",
) -> dict[str, Any]:
    question = {
        "kind": kind,
        **binding,
        "blocks": blocks,
        "signals": evidence["signals"],
        "candidate_roles": evidence["candidate_roles"],
        "text": text,
        "previous_text": previous_text,
        **({"extracted_text": extracted_text} if extracted_text != text else {}),
        "question": (
            "Which structural role is supported by the original page?"
            if kind == "section_role"
            else "Does a section boundary start at the first identified block?"
        ),
        "status": "pending",
    }
    return {"question_id": digest(question), **question}


def validate_answer(question: Mapping[str, Any], answer: Mapping[str, Any]) -> dict[str, Any]:
    """One validation path for replay, agent and terminal answers; fail closed."""
    for key in (
        "question_id",
        "kind",
        "source_sha256",
        "detector_version",
        "settings_sha256",
        "blocks",
    ):
        if answer.get(key) != question.get(key):
            raise ValueError(f"stale or mismatched section answer: {key}")
    if answer.get("origin") not in {"agent", "user"}:
        raise ValueError("section answer origin must be agent or user")
    reason = answer.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("section answer requires review reasoning")
    if question["kind"] == "section_role":
        if answer.get("role") not in ROLES:
            raise ValueError("invalid section role")
    elif type(answer.get("boundary")) is not bool:
        raise ValueError("section boundary answer must be boolean")
    return dict(answer)


def answer_from_decision(question: Mapping[str, Any], decision: Decision) -> dict[str, Any] | None:
    if decision.action == "skip":
        return None
    if decision.action != "classify" or decision.remember != "once":
        raise ValueError("section review cannot delete text or learn global patterns")
    answer = {
        key: question[key]
        for key in (
            "question_id",
            "kind",
            "source_sha256",
            "detector_version",
            "settings_sha256",
            "blocks",
        )
    }
    answer.update(
        origin=decision.origin,
        reason=decision.reason,
        role=decision.section_role,
        boundary=decision.section_boundary,
    )
    return validate_answer(question, answer)


def classify_document(
    doc: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    answers: Sequence[Mapping[str, Any]] = (),
    callback: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return annotated copied blocks, pending questions and replayable decisions."""
    structure = doc.get("source_structure") or {}
    source_sha = structure.get("source_sha256")
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ValueError("section detection requires adapter-verified source SHA256")
    binding = {
        "source_sha256": source_sha,
        "detector_version": DETECTOR_VERSION,
        "settings_sha256": digest(
            {"detector": settings, "pipeline": structure.get("pipeline_settings_sha256")}
        ),
    }
    answer_map = {answer.get("question_id"): answer for answer in answers}
    if len(answer_map) != len(answers):
        raise ValueError("duplicate section answers")
    raw_pages = {p["page"]: p for p in structure.get("pages", ())}
    outline = structure.get("outline") or ()
    pages, pending, decisions, used = [], [], [], set()
    source_identities: dict[str, dict[str, Any]] = {}
    previous, previous_text, section_id = None, "", ""
    parts = []
    for page in doc.get("pages", ()):
        raw = raw_pages.get(page.get("page"), {})
        if (
            settings.get("recover_structural_text")
            and page_evidence(page, raw, outline, None)["role"] == "contents"
            and not _raw_covers_extracted(
                str(raw.get("text") or ""), page.get("blocks", ()), page.get("page")
            )
        ):
            # Check before heading partitioning: an attached following-page
            # heading must not disguise an extraction coverage conflict.
            parts.append((dict(page), dict(raw)))
        else:
            parts.extend(_page_parts(page, raw))
    for page, raw in parts:
        blocks = list(page.get("blocks", ()))
        evidence = page_evidence(page, raw, outline, previous)
        if settings.get("recover_structural_text") and evidence["role"] == "contents":
            # Original TOC text can be erased by footer patterns. Crosspage
            # extraction must stay intact until its boundaries can be reviewed.
            raw_text = str(raw.get("text") or "")
            if raw_text.strip() and not _raw_covers_extracted(raw_text, blocks, page.get("page")):
                evidence = {
                    "role": "unknown",
                    "signals": [
                        *evidence["signals"],
                        "structural_recovery_extracted_text_conflict",
                    ],
                    "status": "pending",
                    "candidate_roles": ["contents", "unknown"],
                }
            elif raw_text.strip():
                first_source = blocks[0].get("source", {}) if blocks else {}
                physical = (
                    [
                        {
                            "text": str(original.get("raw_text") or original["text"]),
                            "type": "paragraph",
                            "bbox": original.get("bbox"),
                            "source": {
                                **first_source,
                                "page": page.get("page"),
                                "method": "original_structure_recovery",
                                "raw_source": {
                                    "page": page.get("page"),
                                    "block_index": original["index"],
                                    "bbox": original.get("bbox", []),
                                    "text_sha256": hashlib.sha256(
                                        str(original.get("raw_text") or original["text"]).encode()
                                    ).hexdigest(),
                                    "span": [
                                        0,
                                        len(str(original.get("raw_text") or original["text"])),
                                    ],
                                },
                            },
                            "_section_block_index": original["index"],
                            "_section_kind": "original_physical_block",
                        }
                        for original in raw.get("blocks", ())
                    ]
                    if first_source.get("raw_source")
                    else []
                )
                blocks = physical or [
                    {
                        "text": raw_text,
                        "type": "paragraph",
                        "source": {
                            **first_source,
                            "page": page.get("page"),
                            "method": "original_structure_recovery",
                        },
                        "_section_block_index": -1,
                        "_section_kind": "original_page_text",
                    }
                ]
        identities = [block_identity(page.get("page"), i, b) for i, b in enumerate(blocks)]
        text = str(raw.get("text") or "\n".join(str(b.get("text", "")) for b in blocks))
        origin, reason, boundary = "detection", "", None
        if evidence["status"] == "pending" and blocks:
            for kind in ("section_role", "section_boundary"):
                question = make_question(
                    kind,
                    binding,
                    identities,
                    evidence,
                    text,
                    previous_text,
                    "\n".join(str(block.get("text", "")) for block in blocks),
                )
                answer = answer_map.get(question["question_id"])
                if answer is None and callback is not None:
                    context = DecisionContext(
                        kind=DecisionKind(kind),
                        curr_text=text,
                        prev_text=previous_text,
                        page=int(page.get("page") or 0),
                        confidence=0.5,
                        pattern_name="section_structure",
                        extra=question,
                    )
                    answer = answer_from_decision(question, callback(context))
                if answer is None:
                    pending.append(question)
                    if kind == "section_boundary":
                        evidence = {**evidence, "status": "pending"}
                    continue
                answer = validate_answer(question, answer)
                if question["question_id"] in answer_map:
                    used.add(question["question_id"])
                decisions.append(answer)
                origin, reason = answer["origin"], answer["reason"]
                if kind == "section_role":
                    evidence = {**evidence, "role": answer["role"], "status": "resolved"}
                else:
                    boundary = answer["boundary"]
        changed = previous is None or evidence["role"] != previous["role"]
        structural_start = page.get("_section_internal_boundary") or (
            evidence["role"] == "body"
            and (
                "body_heading" in evidence["signals"] or "heading_typography" in evidence["signals"]
            )
        )
        if boundary is False and changed:
            # Contradictory answers cannot silently blend roles.
            raise ValueError("cannot join sections with conflicting roles")
        if changed or boundary is True or structural_start or evidence["status"] == "pending":
            section_id = digest(
                {
                    **binding,
                    "start": identities[:1],
                    "page": page.get("page"),
                    "role": evidence["role"],
                }
            )[:20]
        annotated = []
        for block, identity in zip(blocks, identities, strict=True):
            provenance = identity
            if identity.get("raw_source"):
                key = digest(identity)[:24]
                if key in source_identities and source_identities[key] != identity:
                    raise ValueError("source identity digest collision")
                source_identities[key] = identity
                provenance = {
                    "id": key,
                    "page": identity["page"],
                    "block_index": identity["block_index"],
                    "bbox": identity["bbox"],
                    "text_sha256": identity["text_sha256"],
                }
            section = {
                "section_role": evidence["role"],
                "section_id": section_id,
                "section_signals": evidence["signals"],
                "section_review_status": evidence["status"],
                "section_decision_origin": origin,
                "section_source_sha256": source_sha,
                "section_detector_version": DETECTOR_VERSION,
            }
            if reason:
                section["section_review_reason"] = reason
            annotated.append(
                {
                    **block,
                    "section": section,
                    "source": {
                        **(block.get("source") or {}),
                        **section,
                        "block_provenance": [provenance],
                    },
                }
            )
        pages.append({**page, "blocks": annotated})
        previous, previous_text = evidence, text
    if set(answer_map) != used:
        raise ValueError("stale section answers do not match current source/questions")
    return (
        {
            **doc,
            "pages": pages,
            "section_detection_enabled": True,
            "source_block_identities": source_identities,
        },
        pending,
        decisions,
    )
