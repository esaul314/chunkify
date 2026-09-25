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
        "HOST: Welcome back to the show.",
        "GUEST: Thanks for having me.",
    ]
    assert all("WEBVTT" not in block["text"] for block in blocks)
    assert all(block["source"]["filename"] == "episode.txt" for block in blocks)
    assert blocks[0]["source"]["timestamp_start"] == "00:00:01.000"
    assert blocks[0]["source"]["timestamp_end"] == "00:00:04.000"
    assert blocks[1]["source"]["timestamp_start"] == "00:00:05.000"
    assert blocks[1]["source"]["timestamp_end"] == "00:00:08.000"


def test_io_text_normalizes_bom_crlf_and_speaker_labels(tmp_path: Path) -> None:
    transcript = tmp_path / "episode.txt"
    transcript.write_bytes(
        b"\xef\xbb\xbfWEBVTT\r\n\r\n"
        b"1\r\n00:00:01.000 --> 00:00:02.000\r\n"
        b"[host]:\r\nFirst line\r\nsecond line\r\n\r\n"
        b"2\r\n00:00:03.000 --> 00:00:04.000\r\n"
        b"guest:\r\nA second cue.\r\n\r\n"
        b"A plain paragraph.\r\n",
    )

    blocks = io_text.read(str(transcript))["pages"][0]["blocks"]

    assert [block["text"] for block in blocks] == [
        "[host]: First line second line",
        "guest: A second cue.",
        "A plain paragraph.",
    ]
    assert blocks[0]["source"]["location"] == "paragraph:1"
    assert blocks[1]["source"]["location"] == "paragraph:2"
    assert blocks[2]["source"]["location"] == "paragraph:3"
    assert all("\r" not in block["text"] for block in blocks)
    assert blocks[2]["source"].get("timestamp_start") is None


def test_io_text_does_not_add_timestamp_provenance_to_plain_text(tmp_path: Path) -> None:
    transcript = tmp_path / "notes.txt"
    transcript.write_text("Speaker:\nA plain note.\n", encoding="utf-8")

    block = io_text.read(str(transcript))["pages"][0]["blocks"][0]

    assert block["text"] == "Speaker: A plain note."
    assert "timestamp_start" not in block["source"]
    assert "timestamp_end" not in block["source"]
