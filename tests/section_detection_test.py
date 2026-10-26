"""Structural roles are conservative, lossless, source-bound and interface-neutral."""

import json
from copy import deepcopy
from io import StringIO
from unittest.mock import patch

import pytest

from pdf_chunker.adapters.section_review import read_answers, terminal_review, write_questions
from pdf_chunker.framework import Artifact
from pdf_chunker.interactive import Decision
from pdf_chunker.passes.emit_jsonl import emit_jsonl
from pdf_chunker.passes.section_detect import section_detect
from pdf_chunker.passes.split_semantic import _SplitSemanticPass
from pdf_chunker.sections import DETECTOR_VERSION, classify_document

SHA = "a" * 64


def document(texts, *, outline=()):
    return {
        "type": "page_blocks",
        "source_path": "/original/book.pdf",
        "source_structure": {
            "source_sha256": SHA,
            "outline": outline,
            "pipeline_settings_sha256": "settings",
            "pages": [{"page": n, "text": text} for n, text in enumerate(texts, 1)],
        },
        "pages": [
            {
                "page": n,
                "blocks": [
                    {
                        "text": text,
                        "type": "paragraph",
                        "source": {"filename": "book.pdf", "page": n},
                    }
                ],
            }
            for n, text in enumerate(texts, 1)
        ],
    }


def section_roles(result):
    return [b["section"]["section_role"] for p in result["pages"] for b in p["blocks"]]


def answer(question, *, role=None, boundary=None, origin="agent"):
    return {
        **question,
        "origin": origin,
        "reason": "Original page and adjacent text inspected",
        "role": role,
        "boundary": boundary,
    }


def test_contents_without_outline_and_acknowledgment_outline_continuation():
    doc = document(
        [
            "Contents\nChapter One\n1\nChapter Two\n20\nChapter Three\n30",
            "Acknowledgements\nWe thank our friends. We owe a debt of gratitude.",
            "Example Press: Sample Author (2002), pp. 12-13. Other Press: Y (2011).",
            "Introduction\n" + "This explains the intellectual framework. " * 10,
        ],
        outline=[{"title": "Acknowledgements", "page": 2}, {"title": "Introduction", "page": 4}],
    )
    original = deepcopy(doc)
    result, questions, _ = classify_document(doc, settings={})
    assert section_roles(result) == ["contents", "acknowledgments", "acknowledgments", "body"]
    assert (
        result["pages"][1]["blocks"][0]["section"]["section_id"]
        == result["pages"][2]["blocks"][0]["section"]["section_id"]
    )
    assert not questions
    assert doc == original
    assert [b["text"] for p in result["pages"] for b in p["blocks"]] == [
        b["text"] for p in doc["pages"] for b in p["blocks"]
    ]


@pytest.mark.parametrize(
    "text",
    [
        "The contents of this chapter include acknowledgments of how context matters. " * 8,
        "We discuss the index of social integration. "
        "This substantive introduction examines the contents. " * 8,
        "Acknowledgments can shape political relationships. These contents make an argument. " * 8,
    ],
)
def test_prose_mentions_do_not_classify_nonbody(text):
    result, questions, _ = classify_document(document([text]), settings={})
    assert section_roles(result) == ["unknown"]
    assert not questions


@pytest.mark.parametrize(
    ("text", "role"),
    [
        (
            "Permissions\nAll rights reserved. Reproduced with permission of the publisher.",
            "permissions",
        ),
        ("References\nSmith (1998). Jones (2001). Brown (2020).", "references"),
        ("Index\nActors  12\nBehavior  45\nContext  90", "index"),
    ],
)
def test_nonbody_independent_signals(text, role):
    result, pending, _ = classify_document(document([text]), settings={})
    assert section_roles(result) == [role]
    assert not pending


def test_conflicting_structure_creates_role_and_boundary_review_not_deletion():
    doc = document(
        ["Acknowledgements\nWe thank all our friends.\n1\n20\n30"],
        outline=[{"title": "Contents", "page": 1}],
    )
    result, questions, decisions = classify_document(doc, settings={})
    assert section_roles(result) == ["unknown"]
    assert [q["kind"] for q in questions] == ["section_role", "section_boundary"]
    assert result["pages"][0]["blocks"][0]["text"] == doc["pages"][0]["blocks"][0]["text"]
    assert not decisions


def test_agent_machine_terminal_shared_validation_and_replay(tmp_path):
    doc = document(["Acknowledgements"])
    _, questions, _ = classify_document(doc, settings={"enabled": True})
    replay = [answer(q, role="acknowledgments", boundary=True) for q in questions]
    automatic, pending, saved = classify_document(doc, settings={"enabled": True}, answers=replay)
    assert not pending
    assert saved == replay

    def agent(context):
        return Decision(
            action="classify",
            section_role="acknowledgments",
            section_boundary=True,
            origin="agent",
            reason="Original page and adjacent text inspected",
        )

    callback_result, _, _ = classify_document(doc, settings={"enabled": True}, callback=agent)
    assert callback_result == automatic
    with (
        patch(
            "builtins.input",
            side_effect=[
                "acknowledgments",
                "Original page and adjacent text inspected",
                "y",
                "Original page and adjacent text inspected",
            ],
        ),
        patch("sys.stdout", new=StringIO()),
    ):
        terminal_result, pending, terminal_answers = classify_document(
            doc, settings={"enabled": True}, callback=terminal_review
        )
    assert not pending
    for page in terminal_result["pages"]:
        for block in page["blocks"]:
            block["section"]["section_decision_origin"] = "agent"
            block["source"]["section_decision_origin"] = "agent"
    assert terminal_result == automatic
    qpath = str(tmp_path / "pending.json")
    write_questions(qpath, questions)
    assert json.loads((tmp_path / "pending.json").read_text()) == questions
    with pytest.raises(FileExistsError):
        write_questions(qpath, questions)
    apath = tmp_path / "answers.json"
    apath.write_text(json.dumps(terminal_answers))
    assert read_answers(str(apath)) == terminal_answers


@pytest.mark.parametrize("mutation", ["source", "block", "settings", "detector", "extra"])
def test_stale_replay_is_rejected(mutation):
    doc = document(["Acknowledgements"])
    _, questions, _ = classify_document(doc, settings={})
    answers = [answer(q, role="acknowledgments", boundary=True) for q in questions]
    settings = {}
    if mutation == "source":
        doc["source_structure"]["source_sha256"] = "b" * 64
    if mutation == "block":
        doc["pages"][0]["blocks"][0]["text"] += " changed"
    if mutation == "settings":
        settings["threshold"] = 0.8
    if mutation == "detector":
        answers[0]["detector_version"] = "obsolete"
    if mutation == "extra":
        answers += [{**answers[0], "question_id": "unknown"}]
    with pytest.raises(ValueError, match="stale"):
        classify_document(doc, settings=settings, answers=answers)


def test_unanswered_reviews_remain_pending_without_stopping_body_inspection():
    doc = document(["Acknowledgements", "Introduction\n" + "This is substantive discussion. " * 30])
    result, pending, _ = classify_document(
        doc, settings={}, callback=lambda ctx: Decision(action="skip")
    )
    assert section_roles(result) == ["unknown", "body"]
    assert len(pending) == 2
    assert result["pages"][0]["blocks"][0]["section"]["section_review_status"] == "pending"


def test_no_implicit_global_learning_or_lossy_decision():
    with pytest.raises(ValueError, match="learn global"):
        classify_document(
            document(["Acknowledgements"]),
            settings={},
            callback=lambda ctx: Decision(action="merge", remember="always"),
        )


def test_opt_in_default_noop_and_missing_hash_refusal():
    a = Artifact(payload=document(["Contents"]))
    assert section_detect(a) is a
    with pytest.raises(ValueError, match="SHA256"):
        section_detect(
            Artifact(
                payload={"type": "page_blocks", "pages": []},
                meta={"options": {"section_detect": {"enabled": True}}},
            )
        )


def test_section_boundary_prevents_merging_overlap_and_dedupe():
    doc = document(
        [
            "Contents\nA  1\nB  2\nC  3",
            "Introduction\n" + "This substantive body explains the concept. " * 30,
        ]
    )
    annotated, _, _ = classify_document(doc, settings={})
    original = deepcopy(annotated)
    result = _SplitSemanticPass(chunk_size=40, overlap=10)(Artifact(payload=annotated))
    rows = emit_jsonl(result).payload
    assert {row["metadata"]["section_role"] for row in rows} == {"contents", "body"}
    for row in rows:
        meta = row["metadata"]
        assert meta["section_source_sha256"] == SHA
        assert meta["section_detector_version"] == DETECTOR_VERSION
        assert len(json.dumps(row, ensure_ascii=False)) <= 8000
        assert meta["block_provenance"]
        if meta["section_role"] == "body":
            assert "Contents" not in row["text"]
        else:
            assert "substantive" not in row["text"]
    assert annotated == original
    assert len({row["metadata"]["chunk_id"] for row in rows}) == len(rows)


def test_no_join_with_conflicting_roles():
    doc = document(
        [
            "Introduction\n" + "The useful body explanation has several sentences. " * 10,
            "Acknowledgements",
        ]
    )
    _, qs, _ = classify_document(doc, settings={})
    with pytest.raises(ValueError, match="conflicting roles"):
        classify_document(
            doc,
            settings={},
            answers=[answer(q, role="acknowledgments", boundary=False) for q in qs],
        )


def test_mixed_page_roles_split_at_exact_internal_heading_without_text_loss():
    text = (
        "Contents\nOne  1\nTwo  2\nThree  3\nIntroduction\n"
        + "This substantive introduction explains the framework. " * 20
    )
    doc = document([text])
    result, pending, _ = classify_document(doc, settings={})
    assert section_roles(result) == ["contents", "body"]
    assert not pending
    blocks = [block for page in result["pages"] for block in page["blocks"]]
    assert "".join(block["text"] for block in blocks) == text
    assert blocks[1]["source"]["block_provenance"][0]["span"][0] > 0
    split = _SplitSemanticPass(chunk_size=40, overlap=10)(Artifact(payload=result))
    for item in split.payload["items"]:
        if item["meta"]["section_role"] == "body":
            assert "Contents" not in item["text"]


def test_internal_boundary_question_and_replay_exact_block_span():
    text = "Contents\nA  1\nB  2\nC  3\nAcknowledgements"
    doc = document([text])
    _, pending, _ = classify_document(doc, settings={})
    assert pending
    assert all(q["blocks"][0]["span"][0] == text.index("Acknowledgements") for q in pending)
    replay = [answer(q, role="acknowledgments", boundary=True) for q in pending]
    resolved, pending, _ = classify_document(doc, settings={}, answers=replay)
    assert not pending
    assert section_roles(resolved) == ["contents", "acknowledgments"]


def test_structural_text_recovery_is_explicit_and_preserves_original_toc_text():
    doc = document(["Contents\nA  1\nB  2\nC  3"])
    doc["pages"][0]["blocks"][0]["text"] = "Contents\nA"
    result, _, _ = classify_document(doc, settings={"recover_structural_text": True})
    assert result["pages"][0]["blocks"][0]["text"] == doc["source_structure"]["pages"][0]["text"]
    assert (
        result["pages"][0]["blocks"][0]["source"]["block_provenance"][0]["kind"]
        == "original_page_text"
    )


def test_exact_span_guard_detects_missing_tail_hidden_by_overlap():
    from pdf_chunker.passes.split_semantic import _covers_source_in_order, _get_split_fn

    text = " ".join(f"word{n:04d}" for n in range(1010))
    pieces, _ = _get_split_fn(400, 100, 40, preserve_source=True)
    result = pieces(text)
    assert _covers_source_in_order(text, result)
    assert "word1009" in " ".join(result)
    assert not _covers_source_in_order(text, [text[:200], text[:200], text[:200]])


@pytest.mark.parametrize(
    ("text", "end"),
    [
        ("8(3) 352–369. Author A. Example B. and Writer C. Example reference.", 10),
        ("2481–2496. Writer A. Example B. and Scholar C. Example reference.", 2),
        ("3682–3694. Author A. Example B. and Researcher C. Example reference.", 1),
    ],
)
def test_section_font_boundary_cannot_split_inside_citation_number(text, end):
    from pdf_chunker.inline_styles import InlineStyleSpan
    from pdf_chunker.passes.split_modules.inline_headings import split_inline_heading

    block = {
        "text": text,
        "type": "paragraph",
        "section": {"section_role": "references"},
        "inline_styles": [InlineStyleSpan(start=0, end=end, style="italic")],
    }
    assert split_inline_heading(block, text) is None


@pytest.mark.parametrize("crosspage_range", [False, True])
def test_raw_contents_recovery_preserves_merged_following_body(crosspage_range):
    contents = "Contents\nChapter One\n1\nChapter Two\n20\nChapter Three\n30"
    body = "Introduction\n" + "The following page contains substantive explanatory text. " * 8
    doc = document([contents, body])
    merged = contents + "\n" + body
    doc["pages"][0]["blocks"][0]["text"] = merged
    if crosspage_range:
        doc["pages"][0]["blocks"][0]["source"]["page_range"] = [1, 2]
    doc["pages"][1]["blocks"] = []
    original = deepcopy(doc)

    result, pending, decisions = classify_document(doc, settings={"recover_structural_text": True})

    retained = result["pages"][0]["blocks"][0]
    assert retained["text"] == merged
    if crosspage_range:
        assert retained["source"]["page_range"] == [1, 2]
    assert retained["section"]["section_role"] == "unknown"
    assert retained["section"]["section_review_status"] == "pending"
    assert "structural_recovery_extracted_text_conflict" in retained["section"]["section_signals"]
    assert len(pending) == 2
    assert all(question["extracted_text"] == merged for question in pending)
    assert not decisions
    assert doc == original
    artifact = Artifact(doc)
    assert section_detect(artifact) is artifact

    # A changed extraction extent invalidates a previously reviewed answer.
    if crosspage_range:
        changed = deepcopy(doc)
        changed["pages"][0]["blocks"][0]["source"]["page_range"] = [1, 3]
        reviewed = [answer(question, role="unknown", boundary=True) for question in pending]
        with pytest.raises(ValueError, match="stale"):
            classify_document(changed, settings={"recover_structural_text": True}, answers=reviewed)


def test_stale_reference_outline_cannot_confidently_exclude_new_introduction():
    doc = document(
        [
            "References\nSample (2001). Example (2002). Invented (2003).",
            "Introduction\n"
            + (
                "This explanation cites 2001, 2002 and 2003 "
                "while developing a substantive argument. "
            )
            * 8,
        ],
        outline=[{"title": "References", "page": 1}],
    )
    result, pending, _ = classify_document(doc, settings={})
    assert section_roles(result) == ["references", "unknown"]
    assert pending
    assert result["pages"][1]["blocks"][0]["text"] == doc["pages"][1]["blocks"][0]["text"]


def test_reader_and_pipeline_profile_change_invalidates_replay():
    doc = document(
        ["Acknowledgements\nWe thank our friends.\n1\n20\n30"],
        outline=[{"title": "Contents", "page": 1}],
    )
    doc["source_structure"]["pipeline_settings_sha256"] = "first-reader-and-pipeline"
    _, pending, _ = classify_document(doc, settings={})
    assert pending
    responses = [answer(question, role="unknown", boundary=0) for question in pending]
    doc["source_structure"]["pipeline_settings_sha256"] = "different-reader-or-source-units"
    with pytest.raises(ValueError, match="stale|unmatched"):
        classify_document(doc, settings={}, answers=responses)
