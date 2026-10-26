"""Independent read-only verification of immutable CPU section pilot artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def identity(entry):
    return json.dumps(entry, sort_keys=True, separators=(",", ":"))


def tokens(text):
    return re.findall(r"\w+", text.casefold())


def ordered_subset(needle, haystack):
    index = 0
    for token in haystack:
        if index < len(needle) and token == needle[index]:
            index += 1
    return index == len(needle)


def verify(directory, *, inspect_pending=False):
    directory = Path(directory)
    receipt = load(directory / "cpu-acceptance.json")
    manifest = load(directory / "frozen-inputs.json")
    from scripts.prepare_section_pilots import pilot_settings

    # Settings come from the reviewed converter composition, not invented exclusions
    # in a producer-authored receipt.
    assert load(directory / "settings.json") == pilot_settings(), "reviewed CPU settings differ"
    if not inspect_pending:
        assert receipt["offline_accepted"] is True
    result = {
        "cpu_acceptance_sha256": sha(directory / "cpu-acceptance.json"),
        "inspection_only": inspect_pending,
        "books": {},
    }
    for key, book in receipt["books"].items():
        assert sha(book["source"]) == book["source_sha256"]
        for old in manifest[key]["artifacts"]:
            assert sha(old["path"]) == old["sha256"]
        for name, wanted in book["sidecars"].items():
            assert sha(directory / name) == wanted
        artifact = Path(book["artifact"])
        assert sha(artifact) == book["artifact_sha256"]
        rows = [json.loads(line) for line in artifact.read_text().splitlines()]
        semantic = load(directory / f"{key}-semantic-snapshot.json")["chunks"]["items"]
        classified = load(directory / f"{key}-classified-blocks.json")
        raw = load(directory / f"{key}-original-structure.json")
        raw_pages = {page["page"]: page for page in raw["pages"]}
        from pdf_chunker.adapters.section_review import source_structure
        from pdf_chunker.preservation import verify_preservation

        # Re-read the original bytes with the recorded effective native reader.
        # Saved sidecars and producer acceptance booleans are never the raw authority.
        fresh = source_structure(book["source"], {})
        expected_raw = {
            key: value for key, value in raw.items() if key != "pipeline_settings_sha256"
        }
        assert json.loads(json.dumps(fresh)) == expected_raw, "original raw reader facts differ"
        from pdf_chunker.config import PipelineSpec
        from pdf_chunker.core_new import _run_passes
        from pdf_chunker.framework import Artifact

        input_document = load(directory / f"{key}-input-snapshot.json")
        assert input_document["source_path"] == book["source"], "source path differs"
        settings = load(directory / "settings.json")
        review_options = {name: dict(values) for name, values in settings["options"].items()}
        review_options["section_detect"]["answers"] = load(directory / f"{key}-review-answers.json")
        recomputed, _ = _run_passes(
            PipelineSpec(
                pipeline=settings["pipeline"][:-1],
                options=review_options,
            ),
            Artifact(payload=input_document),
        )
        assert (
            json.loads(json.dumps(recomputed.payload["pages"], default=str)) == classified
        ), "classified detection/review replay differs"
        assert (recomputed.meta or {}).get("section_questions") == load(
            directory / f"{key}-questions.json"
        ), "pending review questions differ"
        assert (recomputed.meta or {}).get("section_decisions") == load(
            directory / f"{key}-decisions.json"
        ), "review decisions differ"
        preserved = verify_preservation(
            load(directory / f"{key}-preservation.json"),
            source_structure=fresh,
            adapter_document=load(directory / f"{key}-input-snapshot.json"),
            classified_document={"type": "page_blocks", "pages": classified},
            rows=rows,
            settings=load(directory / "settings.json"),
        )
        if not inspect_pending:
            assert preserved["accepted"], "unresolved raw-to-serialized preservation"
        origins = {
            identity(entry): block
            for page in classified
            for block in page["blocks"]
            for entry in block["source"]["block_provenance"]
        }
        for chunk in semantic:
            for entry in chunk["meta"]["block_provenance"]:
                assert identity(entry) in origins, "unbound block provenance"
        # Sibling serialized records concatenate exactly, including words split
        # across records. No whitespace insertion, dedupe or clipping is allowed.
        logical = []
        cursor = 0
        for chunk in semantic:
            gathered = ""
            parts = []
            while cursor < len(rows) and len(gathered) < len(chunk["text"]):
                row = rows[cursor]
                assert row["metadata"]["section_id"] == chunk["meta"]["section_id"]
                assert row["metadata"]["section_role"] == chunk["meta"]["section_role"]
                assert len(json.dumps(row, ensure_ascii=False)) <= 8000
                gathered += row["text"]
                parts.append(row["metadata"].get("chunk_part"))
                cursor += 1
            assert gathered == chunk["text"], "serialized text/order differs"
            assert parts == [None] or parts == list(range(len(parts))), "invalid part sequence"
            logical.append(gathered)
        assert cursor == len(rows)
        section_tokens = {}
        for chunk in semantic:
            section_tokens.setdefault(chunk["meta"]["section_id"], []).extend(tokens(chunk["text"]))
        missing = []
        for origin, block in origins.items():
            needle = tokens(block["text"])
            haystack = section_tokens.get(block["section"]["section_id"], [])
            if not ordered_subset(needle, haystack):
                missing.append({"origin": json.loads(origin), "text_words": len(needle)})
        recovered = []
        for origin, block in origins.items():
            origin = json.loads(origin)
            if origin.get("kind") == "original_page_text":
                assert origin["block_index"] == -1
                assert block["text"] == raw_pages[origin["page"]]["text"]
                assert origin["text_sha256"] == hashlib.sha256(block["text"].encode()).hexdigest()
                recovered.append(origin["page"])
        result["books"][key] = {
            "records": len(rows),
            "pending_section_questions": len(load(directory / f"{key}-questions.json")),
            "ordered_source_preservation": preserved,
            "logical_chunks": len(logical),
            "bound_block_identities": len(origins),
            "ordered_block_token_coverage_missing": missing,
            "original_contents_pages_recovered": sorted(set(recovered)),
            "exact_serialized_text_order": True,
            "max_serialized_chars": max(len(json.dumps(row, ensure_ascii=False)) for row in rows),
            "all_source_and_historical_artifact_hashes_unchanged": True,
        }
    result["accepted"] = not inspect_pending and all(
        b["ordered_source_preservation"]["accepted"]
        and not b["pending_section_questions"]
        and not b["ordered_block_token_coverage_missing"]
        for b in result["books"].values()
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--inspect-pending",
        action="store_true",
        help="Recompute pending artifacts without granting CPU acceptance",
    )
    args = parser.parse_args()
    result = verify(args.directory, inspect_pending=args.inspect_pending)
    descriptor = os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "accepted": result["accepted"],
                "report": args.report,
                "books": {
                    k: {
                        "records": b["records"],
                        "missing_blocks": len(b["ordered_block_token_coverage_missing"]),
                    }
                    for k, b in result["books"].items()
                },
            }
        )
    )
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
