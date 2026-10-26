"""CPU-only immutable section pilot preparation using canonical library passes.

Run from the converter root with PYTHONPATH=. and --output NEW directory.
Source manifest entries bind key/source/source_sha256/artifacts and artifact SHA.
No enrichment, inference, index, model or service writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import yaml

import pdf_chunker
from pdf_chunker import PipelineSpec
from pdf_chunker.adapters.ai_enrich import _load_tag_configs
from pdf_chunker.core_new import _input_artifact, _run_passes
from pdf_chunker.passes.ai_enrich import _admit_chunks_for_jsonl, _grounded_tags
from pdf_chunker.passes.emit_jsonl import _canonical_rows
from pdf_chunker.preservation import finalize_rows
from pdf_chunker.sections import digest

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
        stream.flush()
        os.fsync(stream.fileno())


def words(text):
    return Counter(re.findall(r"\w+", text.casefold()))


def pilot_settings():
    """Resolve this reviewed CPU composition without source-specific controls."""
    settings = yaml.safe_load((ROOT / "pipeline_rag.yaml").read_text())
    settings["pipeline"] = [p for p in settings["pipeline"] if p not in {"ai_enrich", "emit_jsonl"}]
    settings["pipeline"].insert(settings["pipeline"].index("split_semantic"), "section_detect")
    settings["options"]["heading_detect"] = {"enabled": True}
    settings["options"]["section_detect"] = {"enabled": True, "recover_structural_text": True}
    settings["options"].pop("ai_enrich", None)
    settings["options"].pop("emit_jsonl", None)
    return settings


def prepare(manifest_path, output, *, answers_manifest_path=None):
    assert Path(pdf_chunker.__file__).resolve().parent.parent == ROOT, "wrong converter root"
    manifest = json.loads(Path(manifest_path).read_text())
    answer_manifest = (
        json.loads(Path(answers_manifest_path).read_text()) if answers_manifest_path else {}
    )
    if not isinstance(answer_manifest, dict) or set(answer_manifest) - set(manifest):
        raise ValueError("review answer manifest contains unknown source keys")
    out = Path(output)
    out.mkdir(mode=0o700, parents=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    settings = pilot_settings()
    settings_sha = digest(settings)
    script_sha = sha(__file__)
    write(out / "settings.json", settings)
    write(out / "frozen-inputs.json", manifest)
    report = {
        "converter_root": str(ROOT),
        "converter_commit": commit,
        "settings_sha256": settings_sha,
        "script_sha256": script_sha,
        "books": {},
    }
    vocabulary = _load_tag_configs()
    for key, source in manifest.items():
        path = Path(source["source"])
        assert sha(path) == source["source_sha256"], "source drift"
        for old in source["artifacts"]:
            assert sha(old["path"]) == old["sha256"], "original artifact drift"
        book_settings = {
            **settings,
            "options": {name: dict(values) for name, values in settings["options"].items()},
        }
        answers = []
        if key in answer_manifest:
            entry = answer_manifest[key]
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise ValueError("review answer entry requires path and sha256")
            if sha(entry["path"]) != entry["sha256"]:
                raise ValueError("review answer file drift")
            answers = json.loads(Path(entry["path"]).read_text())
            if not isinstance(answers, list):
                raise ValueError("review answers must be an array")
            book_settings["options"]["section_detect"]["answers"] = answers
        write(out / f"{key}-review-answers.json", answers)
        a = _input_artifact(str(path), PipelineSpec(**book_settings))
        raw_structure = a.payload["source_structure"]
        write(out / f"{key}-input-snapshot.json", a.payload)
        write(out / f"{key}-original-structure.json", raw_structure)
        blocks = []
        for page in a.payload["pages"]:
            for index, block in enumerate(page["blocks"]):
                blocks.append(
                    {
                        "page": page["page"],
                        "block_index": index,
                        "text_sha256": hashlib.sha256(block["text"].encode()).hexdigest(),
                        "bbox": list(block.get("bbox") or ()),
                        "text": block["text"],
                        "source": block.get("source") or {},
                    }
                )
        write(out / f"{key}-extraction-blocks.json", blocks)
        # Freeze cleanup's exact text before structural detection and splitting.
        cleaned, _ = _run_passes(
            PipelineSpec(pipeline=["pdf_parse", "text_clean"], options=book_settings["options"]), a
        )
        cleaned_text = "\n".join(b["text"] for p in cleaned.payload["pages"] for b in p["blocks"])
        prefix = {**book_settings, "pipeline": settings["pipeline"][:-1]}
        classified, _ = _run_passes(PipelineSpec(**prefix), a)
        write(out / f"{key}-classified-blocks.json", classified.payload["pages"])
        result, timings = _run_passes(PipelineSpec(**book_settings), a)
        write(out / f"{key}-questions.json", result.meta.get("section_questions", []))
        write(out / f"{key}-decisions.json", result.meta.get("section_decisions", []))
        write(
            out / f"{key}-semantic-snapshot.json",
            {"chunks": result.payload, "metadata": result.meta, "timings": timings},
        )
        chunks = result.payload["items"]
        sidecar_sha = sha(out / f"{key}-extraction-blocks.json")
        for chunk in chunks:
            meta = chunk["meta"]
            meta["enrichment_status"] = "pending"
            meta["generation_provenance"] = {
                "source_sha256": source["source_sha256"],
                "converter_commit": commit,
                "settings_sha256": settings_sha,
                "review_script_sha256": script_sha,
                "block_map_sha256": sidecar_sha,
                "original_structure_sha256": sha(out / f"{key}-original-structure.json"),
            }
        before = "".join(c["text"] for c in chunks)
        admitted = [
            part
            for chunk in chunks
            for part in _admit_chunks_for_jsonl(
                [chunk],
                max_chars=7000,
                metadata_key="metadata",
                source_preservation=result.meta["preservation"],
                tag_configs={k: _grounded_tags(v, chunk["text"]) for k, v in vocabulary.items()},
            )
        ]
        rows = _canonical_rows({"type": "chunks", "items": admitted}, preserve_text=True)
        preservation, rows = finalize_rows(result.meta["preservation"], rows)
        write(out / f"{key}-preservation.json", preservation)
        assert before == "".join(row["text"] for row in rows), "admission changed text"
        artifact = out / f"{key}-cpu.jsonl"
        write(artifact, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        after_words = words(" ".join(chunk["text"] for chunk in chunks))
        missing = sorted(set(words(cleaned_text)) - set(after_words))
        roles = Counter(row["metadata"]["section_role"] for row in rows)
        coverage = [{"page": p["page"], "blocks": len(p["blocks"])} for p in a.payload["pages"]]
        max_size = max(len(json.dumps(row, ensure_ascii=False)) for row in rows)
        assert max_size <= 8000
        assert all(row["metadata"].get("block_provenance") for row in rows)
        assert sha(path) == source["source_sha256"], "source drift"
        report["books"][key] = {
            "source": str(path),
            "source_sha256": source["source_sha256"],
            "artifact": str(artifact),
            "artifact_sha256": sha(artifact),
            "records": len(rows),
            "max_serialized_chars": max_size,
            "serialized_parts_reconstruct_exact_pre_admission_text": True,
            "serialized_split_records": sum("chunk_part" in row["metadata"] for row in rows),
            "section_roles": dict(roles),
            "pending_questions": len(result.meta.get("section_questions", [])),
            "unique_normalized_words_missing_after_cleanup": missing,
            "ordered_source_preservation": preservation["accepted"],
            "preservation_pending": len(preservation["pending"]),
            "extraction_blocks": len(blocks),
            "pages": coverage,
            "enrichment_status": "pending",
            "sidecars": {p.name: sha(p) for p in out.glob(f"{key}-*.json")},
            "sections": [
                {
                    "page": row["metadata"].get("page"),
                    "role": row["metadata"]["section_role"],
                    "section_id": row["metadata"]["section_id"],
                }
                for row in rows
            ],
        }
    report["offline_accepted"] = all(
        b["ordered_source_preservation"] and not b["pending_questions"]
        for b in report["books"].values()
    )
    write(out / "cpu-acceptance.json", report)
    print(
        json.dumps(
            {
                "output": str(out),
                "accepted": report["offline_accepted"],
                "books": {
                    k: {
                        "records": v["records"],
                        "missing_words": len(v["unique_normalized_words_missing_after_cleanup"]),
                        "pending": v["pending_questions"],
                    }
                    for k, v in report["books"].items()
                },
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--answers-manifest")
    args = parser.parse_args()
    prepare(args.manifest, args.output, answers_manifest_path=args.answers_manifest)


if __name__ == "__main__":
    main()
