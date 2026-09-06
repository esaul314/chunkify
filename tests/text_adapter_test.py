from pathlib import Path

from pdf_chunker.adapters import io_text


def test_io_text_splits_transcript_cues_and_strips_webvtt(tmp_path: Path) -> None:
    transcript = tmp_path / "episode.txt"
    transcript.write_text(
        "\n".join(
            (
                "WEBVTT",
                "",
                "1",
                "00:00:01.000 --> 00:00:04.000",
                "HOST:",
                "Welcome back to the show.",
                "",
                "00:00:05.000 --> 00:00:08.000",
                "GUEST:",
                "Thanks for having me.",
            )
        ),
        encoding="utf-8",
    )

    doc = io_text.read(str(transcript))

    assert doc["type"] == "page_blocks"
    blocks = doc["pages"][0]["blocks"]
    assert [block["text"] for block in blocks] == [
        "00:00:01.000 --> 00:00:04.000 HOST: Welcome back to the show.",
        "00:00:05.000 --> 00:00:08.000 GUEST: Thanks for having me.",
    ]
    assert all("WEBVTT" not in block["text"] for block in blocks)
    assert all(block["source"]["filename"] == "episode.txt" for block in blocks)
